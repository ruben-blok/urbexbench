import argparse
import json
import os
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

load_dotenv()

script_dir = Path(__file__).parent
workspace_root = script_dir.parent
models_file = workspace_root / "models.json"
test_images_dir = workspace_root / "img"
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1") if api_key else None


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


def classify_image(model_id, image_path):
    """Send image to model API for classification using OpenAI SDK.

    Retries until OpenRouter returns a usable 0/1 response.
    """
    image_data = encode_image_to_base64(image_path)
    image_url = f"data:image/png;base64,{image_data}"

    attempt = 0
    backoff_seconds = 1

    while True:
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
                        "effort": "none"
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

        except Exception as e:
            print(
                f"Error classifying {Path(image_path).name} with {model_id} "
                f"on attempt {attempt}: {e}; retrying..."
            )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)

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

def evaluate_model(model_id, test_images):
    """Evaluate a single model on all test images"""
    results = {
        'model': model_id,
        'predictions': defaultdict(list)
    }
    
    all_images = []
    for label, images in test_images.items():
        for img_path in images:
            all_images.append((label, img_path))
    
    print(f"\nEvaluating {model_id} on {len(all_images)} images...")
    
    for true_label, image_path in tqdm(all_images, desc=model_id):
        prediction, cost = classify_image(model_id, image_path)
        
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
        
        print(f"\n{model}:")
        print(f"  Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")

def load_existing_results():
    """Load existing results from results.json if it exists"""
    output_file = workspace_root / "results.json"
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                return {r['model'] for r in data.get('model_results', [])}
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    return set()

def save_results(all_results):
    """Save detailed results to JSON file"""
    output_file = workspace_root / "results.json"
    
    existing_results = []
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                existing_results = data.get('model_results', [])
        except Exception:
            pass
    
    results_by_model = {r['model']: r for r in existing_results}
    for result in all_results:
        results_by_model[result['model']] = result
    
    try:
        with open(output_file, 'w') as f:
            json.dump({'model_results': list(results_by_model.values())}, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

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

    force_models = args.force
    if force_models is not None:
        if len(force_models) == 0:
            models_to_evaluate = models
            print("\nForce mode enabled: re-evaluating all models.")
        else:
            unknown_models = [m for m in force_models if m not in models]
            if unknown_models:
                print(f"Error: unknown model(s) in --force: {', '.join(unknown_models)}")
                return

            forced_set = set(force_models)
            models_to_evaluate = [m for m in models if m in forced_set]
            print(f"\nForce mode enabled: re-evaluating {', '.join(models_to_evaluate)}.")
    else:
        existing_models = load_existing_results()
        if existing_models:
            print(f"\nFound existing results for: {', '.join(existing_models)}")

        models_to_evaluate = [m for m in models if m not in existing_models]
        if not models_to_evaluate:
            print("\nAll models have already been evaluated. Use --force to re-evaluate.")
            return

        print(f"\nEvaluating {len(models_to_evaluate)} new model(s)...")

    if not models_to_evaluate:
        print("\nNo matching models found to evaluate.")
        return
    
    for i, model_id in enumerate(models_to_evaluate, 1):
        try:
            print(f"\n[{i}/{len(models_to_evaluate)}] Evaluating {model_id}...")
            result = evaluate_model(model_id, test_images)
            save_results([result])
            print_results([result])
        except Exception as e:
            print(f"Error evaluating model {model_id}: {e}")
            continue

if __name__ == "__main__":
    main()
