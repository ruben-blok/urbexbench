import argparse
import json
import os
import base64
import time
import urllib.request
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError
from tqdm import tqdm
from collections import defaultdict

load_dotenv()

script_dir = Path(__file__).parent
workspace_root = script_dir.parent
models_file = workspace_root / "models.json"
test_images_dir = workspace_root / "img"
output_file = workspace_root / "results.json"
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=300) if api_key else None

MODELS_API_URL = "https://openrouter.ai/api/v1/models"
EFFORT_RANK = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5, "max": 6}
UNKNOWN_EFFORT_CANDIDATES = ["minimal", "low", "medium", "high"]
FALLBACK_REASONING_EFFORTS = {
    "google/gemini-3.1-flash-lite-preview": ["minimal"],
    "google/gemini-3.1-flash-lite": ["minimal"],
    "deepseek/deepseek-v4-flash-vision-exp": ["low"],
    "stealth/ox-alpha": ["low"],
    "google/gemini-3.7-flash": ["low"],
}
PROBE_MAX_ATTEMPTS = 3
SAVE_LOCK = threading.Lock()


def load_models():
    """Load model names from models.json"""
    if not models_file.exists():
        raise FileNotFoundError(f"models.json not found at {models_file}")
    with open(models_file, 'r') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on test images.")
    parser.add_argument(
        "--force",
        nargs="*",
        metavar="MODEL",
        help="Re-evaluate all models, or only the specified model IDs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of model runs evaluated concurrently.",
    )
    return parser.parse_args()


def encode_image_to_base64(image_path):
    """Encode image to base64 for API transmission"""
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode('utf-8')


def extract_message_cost(response):
    """Return only usage.cost (the amount OpenRouter billed).

    This deliberately does NOT fall back to total_cost or model_extra.
    It accepts both dict-style and SDK object-style responses.
    """
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None

    if isinstance(usage, dict):
        return usage.get("cost")
    return getattr(usage, "cost", None)


def fetch_catalog_reasoning():
    """Fetch reasoning metadata per model from OpenRouter's public catalog."""
    request = urllib.request.Request(MODELS_API_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    return {m["id"]: m.get("reasoning") for m in data.get("data", [])}


def sort_efforts(efforts):
    """Sort reasoning efforts ascending by known rank; unknown efforts last."""
    return sorted(set(efforts), key=lambda e: EFFORT_RANK.get(e, len(EFFORT_RANK)))


def build_run_plan(models, catalog):
    """Plan evaluation runs per model.

    Every endpoint that allows disabling reasoning gets a run with effort
    "none". Endpoints that support reasoning additionally get a run at the
    lowest enabled effort; candidates are ordered ascending so probing picks
    the cheapest accepted one.
    """
    runs = []
    for model_id in models:
        meta = catalog.get(model_id) if catalog else None
        mandatory = bool(meta and meta.get("mandatory"))

        if not mandatory:
            runs.append({"model": model_id, "candidates": ["none"]})

        if meta is not None:
            candidates = sort_efforts(
                e for e in (meta.get("supported_efforts") or []) if e != "none"
            ) or list(UNKNOWN_EFFORT_CANDIDATES)
        elif model_id in FALLBACK_REASONING_EFFORTS:
            candidates = FALLBACK_REASONING_EFFORTS[model_id]
        else:
            candidates = []

        if candidates:
            runs.append({"model": model_id, "candidates": candidates})
    return runs


def resolve_reasoning_effort(model_id, candidates):
    """Validate candidate reasoning efforts with a cheap text request.

    Returns the first accepted effort, or None when every candidate is
    genuinely unsupported. HTTP 429 (rate limit) and 5xx are transient and
    retried on the same candidate; other 4xx indicate an unsupported effort
    and advance to the next candidate. When only transient failures occur
    (e.g. a passing rate-limit storm), the lowest candidate is returned
    anyway so the run proceeds through the real evaluation.
    """
    any_unsupported = False
    for effort in candidates:
        attempt = 0
        backoff_seconds = 1
        while attempt < PROBE_MAX_ATTEMPTS:
            attempt += 1
            try:
                client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "Reply ONLY with 0."}],
                    temperature=0,
                    extra_body={"reasoning": {"effort": effort}},
                )
                return effort
            except APIStatusError as e:
                status = getattr(e, "status_code", 0) or 0
                if status == 429 or status >= 500:
                    print(
                        f"Warning: probe for {model_id} effort '{effort}' failed "
                        f"(HTTP {status}) on attempt {attempt}; retrying..."
                    )
                else:
                    print(
                        f"Info: {model_id} rejected reasoning effort '{effort}' "
                        f"(HTTP {status}); trying next candidate..."
                    )
                    any_unsupported = True
                    break
            except Exception as e:
                print(
                    f"Warning: probe for {model_id} effort '{effort}' failed "
                    f"on attempt {attempt}: {e}; retrying..."
                )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)
    if any_unsupported:
        return None
    return candidates[0] if candidates else None


MAX_CLASSIFY_ATTEMPTS = 12


def classify_image(model_id, image_path, reasoning_effort):
    """Send image to model API for classification using OpenAI SDK.

    Retries until OpenRouter returns a usable 0/1 response.
    """
    image_data = encode_image_to_base64(image_path)
    image_url = f"data:image/png;base64,{image_data}"

    attempt = 0
    backoff_seconds = 1

    while attempt < MAX_CLASSIFY_ATTEMPTS:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": "Is this location abandoned? Reply ONLY with 0 (not-abandoned) or 1 (abandoned). Target location is near center."}
                        ],
                    }
                ],
                temperature=0,
                seed=42,
                extra_body={
                    "reasoning": {
                        "effort": reasoning_effort
                    }
                }
            )

            cost = extract_message_cost(response)
            choices = getattr(response, "choices", None)
            if not choices:
                error = getattr(response, "error", None)
                print(
                    f"Warning: empty response for {Path(image_path).name} with {model_id} "
                    f"on attempt {attempt}: {error or 'no choices'}; retrying..."
                )
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30)
                continue

            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content is None:
                print(
                    f"Warning: empty content for {Path(image_path).name} with {model_id} "
                    f"on attempt {attempt}; retrying..."
                )
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30)
                continue

            content = content.strip()
            if content == '0':
                return 'not-abandoned', cost
            if content == '1':
                return 'abandoned', cost

            print(
                f"Warning: unexpected response '{content}' for {Path(image_path).name} "
                f"with {model_id} on attempt {attempt}; retrying..."
            )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)

        except APIStatusError as e:
            status = getattr(e, "status_code", 0) or 0
            if status != 429 and status < 500:
                print(
                    f"Fatal client error classifying {Path(image_path).name} with "
                    f"{model_id} (HTTP {status}); skipping image."
                )
                return None, None
            print(
                f"Error classifying {Path(image_path).name} with {model_id} "
                f"on attempt {attempt}: {e}; retrying..."
            )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)
        except Exception as e:
            print(
                f"Error classifying {Path(image_path).name} with {model_id} "
                f"on attempt {attempt}: {e}; retrying..."
            )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)

    print(
        f"Giving up on {Path(image_path).name} with {model_id} after "
        f"{MAX_CLASSIFY_ATTEMPTS} attempts; recording as unclassified."
    )
    return None, None

def load_test_images():
    """Load all test images and their ground truth labels"""
    images = {'abandoned': [], 'not-abandoned': []}
    
    if not test_images_dir.exists():
        print(f"Warning: Test images directory not found at {test_images_dir}")
        return images
    
    for label in ['abandoned', 'not-abandoned']:
        label_dir = test_images_dir / label
        if label_dir.exists():
            for image_file in sorted(label_dir.glob('*.png')):
                images[label].append(str(image_file))
    
    return images

def evaluate_model(model_id, test_images, reasoning_effort):
    """Evaluate a single model on all test images"""
    results = {
        'model': model_id,
        'reasoning_effort': reasoning_effort,
        'predictions': defaultdict(list)
    }
    
    all_images = []
    for label, images in test_images.items():
        for img_path in images:
            all_images.append((label, img_path))
    
    print(f"\nEvaluating {model_id} (effort={reasoning_effort}) on {len(all_images)} images...")
    
    for true_label, image_path in tqdm(all_images, desc=f"{model_id} (effort={reasoning_effort})"):
        prediction, cost = classify_image(model_id, image_path, reasoning_effort)
        
        if prediction is not None or cost is not None:
            results['predictions'][true_label].append({
                'image': Path(image_path).name,
                'prediction': prediction,
                'answer': true_label,
                'cost': cost
            })

    return results

def print_results(all_results):
    """Print evaluation results"""
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    for result in all_results:
        model = result['model']
        effort = result.get('reasoning_effort', 'none')
        predictions = result['predictions']
        
        valid_predictions = [
            p
            for preds in predictions.values()
            for p in preds
            if p.get('prediction') is not None
        ]
        total = len(valid_predictions)
        correct = sum(1 for p in valid_predictions if p['prediction'] == p['answer'])
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"\n{model} (effort={effort}):")
        print(f"  Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")

def run_key(result):
    """Unique key for a stored run: (model id, reasoning effort)."""
    return (result["model"], result.get("reasoning_effort", "none"))


def load_existing_results():
    """Load existing runs from results.json keyed by (model, reasoning_effort)"""
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                return {run_key(r): r for r in data.get('model_results', [])}
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    return {}


def prune_stale_results(active_models):
    """Drop stored results for models no longer listed in models.json"""
    if not output_file.exists():
        return
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        all_results = data.get('model_results', [])
        kept = [r for r in all_results if r['model'] in active_models]
        removed = len(all_results) - len(kept)
        if removed:
            with open(output_file, 'w') as f:
                json.dump({'model_results': kept}, f, indent=2)
            print(f"Pruned {removed} result entr{'y' if removed == 1 else 'ies'} for models removed from models.json.")
    except Exception as e:
        print(f"Warning: Could not prune stale results: {e}")

def save_results(all_results):
    """Save detailed results to JSON file, merging by (model, reasoning_effort)"""
    existing_results = []
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                existing_results = data.get('model_results', [])
        except Exception:
            pass
    
    results_by_run = {run_key(r): r for r in existing_results}
    for result in all_results:
        results_by_run[run_key(result)] = result
    
    try:
        with open(output_file, 'w') as f:
            json.dump({'model_results': list(results_by_run.values())}, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def execute_run(run, total_runs, progress_counter, test_images):
    """Evaluate a single planned run and persist its results thread-safely."""
    model_id = run["model"]
    effort = run["effort"]
    try:
        with SAVE_LOCK:
            print(f"\n[{progress_counter[0]}/{total_runs}] Evaluating {model_id} (effort={effort})...")
            progress_counter[0] += 1
        result = evaluate_model(model_id, test_images, effort)
        with SAVE_LOCK:
            save_results([result])
            print_results([result])
        return None
    except Exception as e:
        return f"{model_id} (effort={effort}): {e}"

def main():
    """Main evaluation pipeline"""
    args = parse_args()

    # Validate API key
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        print("Please add it to your .env file or set it as an environment variable.")
        return
    
    # Validate client initialization
    if not client:
        print("Error: Failed to initialize OpenAI client.")
        return
    
    # Load models and test images
    try:
        models = load_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    test_images = load_test_images()
    
    total_test_images = sum(len(imgs) for imgs in test_images.values())
    print(f"\n{'='*80}")
    print(f"Testing {len(models)} models on {total_test_images} test images")
    print(f"  - Abandoned: {len(test_images['abandoned'])}")
    print(f"  - Not-abandoned: {len(test_images['not-abandoned'])}")
    print(f"{'='*80}")
    
    if total_test_images == 0:
        print("Error: No test images found in", test_images_dir)
        return
    
    if not models:
        print("Error: No models found in models.json")
        return

    prune_stale_results(set(models))

    if args.force is not None:
        if len(args.force) == 0:
            forced_models = set(models)
            print("\nForce mode enabled: re-evaluating all models.")
        else:
            unknown_models = [m for m in args.force if m not in models]
            if unknown_models:
                print(f"Error: unknown model(s) in --force: {', '.join(unknown_models)}")
                return
            forced_models = set(args.force)
            print(f"\nForce mode enabled: re-evaluating {', '.join(sorted(forced_models))}.")
    else:
        forced_models = set()

    try:
        catalog = fetch_catalog_reasoning()
    except Exception as e:
        print(f"Warning: Could not fetch OpenRouter catalog ({e}); using fallback efforts.")
        catalog = {}

    runs = build_run_plan(models, catalog)

    existing = load_existing_results()
    done_keys = set() if forced_models else set(existing.keys())
    pending_runs = []
    for run in runs:
        model_id = run["model"]
        candidates = run["candidates"]
        if candidates == ["none"]:
            effort = "none"
        else:
            prior_reasoning = [
                e for (m, e) in existing.keys() if m == model_id and e != "none"
            ]
            if prior_reasoning and model_id not in forced_models:
                effort = sort_efforts(prior_reasoning)[0]
            else:
                resolved = resolve_reasoning_effort(model_id, candidates)
                if resolved is None:
                    print(f"Skipping reasoning-enabled run for {model_id}: no supported effort found.")
                    continue
                effort = resolved
        if (model_id, effort) in done_keys:
            print(f"Skipping {model_id} (effort={effort}): already evaluated.")
            continue
        done_keys.add((model_id, effort))
        pending_runs.append({"model": model_id, "effort": effort})

    if not pending_runs:
        print("\nAll planned runs have already been evaluated. Use --force to re-evaluate.")
        return

    print(f"\n{len(pending_runs)} run(s) to evaluate with {args.workers} worker(s):")
    for run in pending_runs:
        print(f"  - {run['model']} (effort={run['effort']})")

    progress_counter = [1]
    failures = []
    workers = max(1, min(args.workers, len(pending_runs)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute_run, run, len(pending_runs), progress_counter, test_images)
            for run in pending_runs
        ]
        for future in as_completed(futures):
            error = future.result()
            if error:
                print(f"Error evaluating {error}")
                failures.append(error)

    if failures:
        print(f"\nFinished with {len(failures)} failed run(s).")
    else:
        print("\nAll runs completed successfully.")

if __name__ == "__main__":
    main()
