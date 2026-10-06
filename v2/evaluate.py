"""UrbexBench v2 evaluation pipeline.

Improvements over v1:
- Pins a deterministic OpenRouter provider per model (author endpoint when
  available, otherwise the cheapest active endpoint).
- Records run metadata and per-prediction details (provider, cost, latency,
  attempts, raw content, finish reason).
- Stores one file per run in results-v2/ instead of a single shared file.
- Supports a per-class image limit for cheap smoke tests.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI
from tqdm import tqdm

from providers import provider_preferences, resolve_provider

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "models.json"
TEST_IMAGES_DIR = ROOT / "img"
RESULTS_DIR = ROOT / "results-v2"

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
CLIENT = (
    OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1", timeout=300)
    if API_KEY
    else None
)

MODELS_API_URL = "https://openrouter.ai/api/v1/models"
PROMPT = (
    "Is this location abandoned? Reply ONLY with 0 (not-abandoned) "
    "or 1 (abandoned). Target location is near center."
)
PROMPT_VERSION = 1
TEMPERATURE = 0
SEED = 42
LABELS = {"0": "not-abandoned", "1": "abandoned"}

EFFORT_RANK = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5, "max": 6}
UNKNOWN_EFFORT_CANDIDATES = ["minimal", "low", "medium", "high"]
FALLBACK_REASONING_EFFORTS = {
    "google/gemini-3.1-flash-lite-preview": ["minimal"],
    "google/gemini-3.1-flash-lite": ["minimal"],
    "deepseek/deepseek-v4-flash-vision-exp": ["low"],
    "google/gemini-3.7-flash": ["low"],
}
PROBE_MAX_ATTEMPTS = 3
DETERMINISM_PROBE_RUNS = 2
MAX_CLASSIFY_ATTEMPTS = 12
PRINT_LOCK = threading.Lock()


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def load_config() -> list:
    """Load model config entries (strings or {"id", "provider"} objects)."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"models.json not found at {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def normalize_model(entry) -> dict:
    """Normalize a config entry into {"id", "provider"}."""
    if isinstance(entry, str):
        return {"id": entry, "provider": None}
    return {"id": entry["id"], "provider": entry.get("provider")}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on test images (v2).")
    parser.add_argument(
        "--force",
        nargs="*",
        metavar="MODEL",
        help="Re-evaluate all models, or only the specified model IDs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        metavar="MODEL",
        help="Only evaluate the specified model IDs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of model runs evaluated concurrently.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Use at most N images per class (for smoke tests).",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the per-run determinism probe.",
    )
    return parser.parse_args()


def encode_image_to_base64(image_path) -> str:
    """Encode an image to base64 for API transmission."""
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")


def extract_message_cost(response) -> float:
    """Return only usage.cost (the amount OpenRouter billed), or None."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get("cost")
    return getattr(usage, "cost", None)


def fetch_catalog_reasoning() -> dict:
    """Fetch reasoning metadata per model from OpenRouter's public catalog."""
    request = urllib.request.Request(MODELS_API_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    return {m["id"]: m.get("reasoning") for m in data.get("data", [])}


def sort_efforts(efforts) -> list:
    """Sort reasoning efforts ascending by known rank; unknown efforts last."""
    return sorted(set(efforts), key=lambda e: EFFORT_RANK.get(e, len(EFFORT_RANK)))


def build_run_plan(model_ids: list, catalog: dict) -> list:
    """Plan evaluation runs per model.

    Every endpoint that allows disabling reasoning gets a "none" run.
    Endpoints that support reasoning additionally get a run at the lowest
    enabled effort.
    """
    runs = []
    for model_id in model_ids:
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


def resolve_reasoning_effort(model_id, candidates, provider_tag) -> str:
    """Validate candidate reasoning efforts with a cheap text request.

    Returns the first accepted effort, or None when every candidate is
    genuinely unsupported. HTTP 429 and 5xx are transient and retried;
    other 4xx indicate an unsupported effort and advance the candidate.
    """
    extra = {"provider": provider_preferences(provider_tag)} if provider_tag else {}
    any_unsupported = False
    for effort in candidates:
        attempt = 0
        backoff_seconds = 1
        while attempt < PROBE_MAX_ATTEMPTS:
            attempt += 1
            try:
                CLIENT.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "Reply ONLY with 0."}],
                    temperature=0,
                    extra_body={"reasoning": {"effort": effort}, **extra},
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


def parse_label(content) -> str:
    """Parse a model response into "abandoned" / "not-abandoned" / None."""
    if content is None:
        return None
    text = content.strip()
    if text in LABELS:
        return LABELS[text]

    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        for key in ("abandoned", "answer", "label", "prediction", "classification"):
            if key not in parsed:
                continue
            value = parsed[key]
            if isinstance(value, bool):
                return "abandoned" if value else "not-abandoned"
            mapped = parse_label(str(value))
            if mapped:
                return mapped
    if isinstance(parsed, bool):
        return "abandoned" if parsed else "not-abandoned"

    lowered = text.lower()
    if re.search(r"\bnot[\s-]?abandoned\b", lowered):
        return "not-abandoned"
    if re.search(r"\babandoned\b", lowered):
        return "abandoned"
    match = re.search(r"(?<!\d)([01])(?!\d)", text)
    if match:
        return LABELS[match.group(1)]
    return None


def classify_image(model_id, image_path, reasoning_effort, provider_tag) -> dict:
    """Classify one image, retrying until a usable label is returned.

    Returns a prediction dict, or a dict with prediction None on failure.
    """
    image_url = "data:image/png;base64," + encode_image_to_base64(image_path)
    attempt = 0
    backoff_seconds = 1
    last_raw = None
    last_finish = None

    while attempt < MAX_CLASSIFY_ATTEMPTS:
        attempt += 1
        started = time.monotonic()
        try:
            response = CLIENT.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
                temperature=TEMPERATURE,
                seed=SEED,
                extra_body={
                    "reasoning": {"effort": reasoning_effort},
                    "provider": provider_preferences(provider_tag),
                },
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            choices = getattr(response, "choices", None)
            if not choices:
                print(f"Warning: empty response for {Path(image_path).name} attempt {attempt}; retrying...")
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30)
                continue

            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None) if message is not None else None
            last_raw = content
            last_finish = getattr(choices[0], "finish_reason", None)
            label = parse_label(content)
            if label:
                serialized = response.model_dump()
                return {
                    "prediction": label,
                    "cost": extract_message_cost(response),
                    "provider": serialized.get("provider"),
                    "latency_ms": latency_ms,
                    "attempts": attempt,
                    "raw_content": content,
                    "finish_reason": last_finish,
                }

            print(
                f"Warning: unexpected response {content!r} for {Path(image_path).name} "
                f"attempt {attempt}; retrying..."
            )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)

        except APIStatusError as e:
            status = getattr(e, "status_code", 0) or 0
            if status != 429 and status < 500:
                print(
                    f"Fatal client error classifying {Path(image_path).name} "
                    f"(HTTP {status}); skipping image."
                )
                return {"prediction": None, "error": f"HTTP {status}", "attempts": attempt}
            print(f"Error classifying {Path(image_path).name} attempt {attempt}: {e}; retrying...")
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)
        except Exception as e:
            print(f"Error classifying {Path(image_path).name} attempt {attempt}: {e}; retrying...")
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)

    return {
        "prediction": None,
        "error": "max attempts reached",
        "attempts": MAX_CLASSIFY_ATTEMPTS,
        "raw_content": last_raw,
        "finish_reason": last_finish,
    }


def probe_determinism(model_id, effort, provider_tag, image_path) -> dict:
    """Check that the pinned provider is stable at temperature 0."""
    outputs = []
    for _ in range(DETERMINISM_PROBE_RUNS):
        result = classify_image(model_id, image_path, effort, provider_tag)
        outputs.append(result.get("prediction"))
    stable = outputs[0] is not None and len(set(outputs)) == 1
    return {"stable": stable, "outputs": outputs}


def load_test_images(limit=None) -> dict:
    """Load test images and their ground-truth labels."""
    images = {"abandoned": [], "not-abandoned": []}
    if not TEST_IMAGES_DIR.exists():
        print(f"Warning: Test images directory not found at {TEST_IMAGES_DIR}")
        return images

    for label in ("abandoned", "not-abandoned"):
        label_dir = TEST_IMAGES_DIR / label
        if label_dir.exists():
            files = sorted(label_dir.glob("*.png"))
            if limit is not None:
                files = files[:limit]
            images[label] = [str(f) for f in files]
    return images


def dataset_hash(test_images: dict) -> str:
    """Stable hash of the evaluated dataset (labels, names and sizes)."""
    digest = hashlib.sha256()
    for label in sorted(test_images):
        for path in sorted(test_images[label]):
            p = Path(path)
            digest.update(f"{label}/{p.name}:{p.stat().st_size}".encode())
    return digest.hexdigest()


def run_file(model_id: str, effort: str) -> Path:
    """Path of the stored run file for a (model, effort) pair."""
    safe_model = model_id.replace("/", "__")
    return RESULTS_DIR / f"{safe_model}__{effort}.json"


def load_existing_runs() -> dict:
    """Load stored runs keyed by (model, reasoning_effort)."""
    runs = {}
    if not RESULTS_DIR.exists():
        return runs
    for path in RESULTS_DIR.glob("*.json"):
        try:
            with open(path, "r") as f:
                run = json.load(f)
            runs[(run["model"], run.get("reasoning_effort", "none"))] = run
        except Exception as e:
            print(f"Warning: could not read {path.name}: {e}")
    return runs


def save_run(run: dict) -> None:
    """Persist a run to its own JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = run_file(run["model"], run["reasoning_effort"])
    with open(path, "w") as f:
        json.dump(run, f, indent=2)


def evaluate_run(model_id, effort, provider_tag, test_images, use_probe) -> dict:
    """Evaluate one (model, effort) run with a pinned provider."""
    all_images = [
        (label, path)
        for label, paths in test_images.items()
        for path in paths
    ]

    determinism = None
    if use_probe and all_images:
        probe_image = all_images[0][1]
        determinism = probe_determinism(model_id, effort, provider_tag, probe_image)

    started_at = now_iso()
    predictions = {"abandoned": [], "not-abandoned": []}
    provider_counts = {}

    with PRINT_LOCK:
        print(f"\nEvaluating {model_id} (effort={effort}, provider={provider_tag}) on {len(all_images)} images...")

    for true_label, image_path in tqdm(
        all_images, desc=f"{model_id} ({effort})", leave=False
    ):
        result = classify_image(model_id, image_path, effort, provider_tag)
        provider_counts[result.get("provider") or provider_tag] = (
            provider_counts.get(result.get("provider") or provider_tag, 0) + 1
        )
        predictions[true_label].append(
            {"image": Path(image_path).name, "answer": true_label, **result}
        )

    return {
        "model": model_id,
        "reasoning_effort": effort,
        "provider": {"requested": provider_tag, "used": provider_counts},
        "determinism": determinism,
        "run": {
            "started_at": started_at,
            "finished_at": now_iso(),
            "dataset_hash": dataset_hash(test_images),
            "total_images": len(all_images),
            "prompt": PROMPT,
            "prompt_version": PROMPT_VERSION,
            "temperature": TEMPERATURE,
            "seed": SEED,
        },
        "predictions": predictions,
    }


def run_score(run: dict) -> tuple:
    """(correct, total) for a stored run."""
    predictions = run["predictions"]
    total = 0
    correct = 0
    for preds in predictions.values():
        for p in preds:
            if p.get("prediction") is None:
                continue
            total += 1
            correct += p["prediction"] == p["answer"]
    return correct, total


def print_run_result(run: dict) -> None:
    """Print accuracy for a single run."""
    correct, total = run_score(run)
    accuracy = (correct / total * 100) if total else 0
    print("\n" + "=" * 80)
    print(f"{run['model']} (effort={run['reasoning_effort']}):")
    print(f"  Provider: {run['provider']}")
    print(f"  Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")


def main():
    args = parse_args()

    if not API_KEY or not CLIENT:
        print("Error: OPENROUTER_API_KEY not found. Add it to .env or the environment.")
        return

    try:
        config = [normalize_model(e) for e in load_config()]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.models:
        unknown = [m for m in args.models if m not in {c["id"] for c in config}]
        if unknown:
            print(f"Error: unknown model(s) in --models: {', '.join(unknown)}")
            return
        config = [c for c in config if c["id"] in set(args.models)]

    model_ids = [c["id"] for c in config]
    overrides = {c["id"]: c["provider"] for c in config if c["provider"]}

    test_images = load_test_images(args.limit)
    total_images = sum(len(v) for v in test_images.values())
    print(f"\n{'=' * 80}")
    print(f"Testing {len(model_ids)} models on {total_images} test images")
    print(f"  - Abandoned: {len(test_images['abandoned'])}")
    print(f"  - Not-abandoned: {len(test_images['not-abandoned'])}")
    print(f"  - Results dir: {RESULTS_DIR}")
    print(f"{'=' * 80}")

    if total_images == 0:
        print("Error: no test images found in", TEST_IMAGES_DIR)
        return
    if not model_ids:
        print("Error: no models found in models.json")
        return

    if args.force is not None:
        if len(args.force) == 0:
            forced_models = set(model_ids)
            print("\nForce mode enabled: re-evaluating all models.")
        else:
            unknown = [m for m in args.force if m not in set(model_ids)]
            if unknown:
                print(f"Error: unknown model(s) in --force: {', '.join(unknown)}")
                return
            forced_models = set(args.force)
            print(f"\nForce mode enabled: re-evaluating {', '.join(sorted(forced_models))}.")
    else:
        forced_models = set()

    try:
        catalog = fetch_catalog_reasoning()
    except Exception as e:
        print(f"Warning: could not fetch OpenRouter catalog ({e}); using fallback efforts.")
        catalog = {}

    runs = build_run_plan(model_ids, catalog)
    existing = load_existing_runs()
    done_keys = {key for key in existing if key[0] not in forced_models}

    provider_cache = {}
    effort_cache = {}
    pending = []
    for run in runs:
        model_id = run["model"]
        candidates = run["candidates"]

        if model_id not in provider_cache:
            provider_cache[model_id] = resolve_provider(model_id, overrides.get(model_id))
        provider = provider_cache[model_id]
        if provider is None:
            print(f"Skipping {model_id}: no active provider endpoint found.")
            continue
        provider_tag = provider["tag"]
        print(
            f"Provider for {model_id}: {provider_tag} "
            f"(source={provider['source']})"
        )

        if candidates == ["none"]:
            effort = "none"
        else:
            prior_reasoning = [e for (m, e) in existing if m == model_id and e != "none"]
            cache_key = (model_id, provider_tag)
            if cache_key in effort_cache:
                effort = effort_cache[cache_key]
            elif prior_reasoning and model_id not in forced_models:
                effort = sort_efforts(prior_reasoning)[0]
                effort_cache[cache_key] = effort
            else:
                resolved = resolve_reasoning_effort(model_id, candidates, provider_tag)
                if resolved is None:
                    print(f"Skipping reasoning-enabled run for {model_id}: no supported effort found.")
                    continue
                effort = resolved
                effort_cache[cache_key] = effort

        if (model_id, effort) in done_keys:
            print(f"Skipping {model_id} (effort={effort}): already evaluated.")
            continue
        done_keys.add((model_id, effort))
        pending.append({"model": model_id, "effort": effort, "provider": provider_tag})

    if not pending:
        print("\nAll planned runs have already been evaluated. Use --force to re-evaluate.")
        return

    print(f"\n{len(pending)} run(s) to evaluate with {args.workers} worker(s):")
    for run in pending:
        print(f"  - {run['model']} (effort={run['effort']}, provider={run['provider']})")

    failures = []
    workers = max(1, min(args.workers, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_run,
                run["model"],
                run["effort"],
                run["provider"],
                test_images,
                not args.no_probe,
            ): run
            for run in pending
        }
        for future in as_completed(futures):
            run = futures[future]
            try:
                result = future.result()
                save_run(result)
                with PRINT_LOCK:
                    print_run_result(result)
            except Exception as e:
                failures.append(f"{run['model']} (effort={run['effort']}): {e}")
                print(f"Error evaluating {failures[-1]}")

    if failures:
        print(f"\nFinished with {len(failures)} failed run(s).")
    else:
        print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
