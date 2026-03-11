import json
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

load_dotenv()

# Configuration
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR  # root of the repository
MODELS_FILE = WORKSPACE_ROOT / "models.json"
TEST_IMAGES_DIR = WORKSPACE_ROOT / "img"
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenAI client with OpenRouter endpoint
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1") if API_KEY else None

def load_models():
    """Load model names from models.json"""
    if not MODELS_FILE.exists():
        raise FileNotFoundError(f"models.json not found at {MODELS_FILE}")
    with open(MODELS_FILE, 'r') as f:
        return json.load(f)

def encode_image_to_base64(image_path):
    """Encode image to base64 for API transmission"""
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode('utf-8')


def classify_image(model_id, image_path):
    """Send image to model API for classification using OpenAI SDK"""
    try:
        image_data = encode_image_to_base64(image_path)
        image_url = f"data:image/png;base64,{image_data}"

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Is this location abandoned? Reply ONLY with a number 0 or 1 (0 = not abandoned, 1 = abandoned). Target location is near center."}
                    ],
                }
            ],
            extra_body={
                "reasoning": {
                    "enabled": "true"
                }
            }
        )

        content = response.choices[0].message.content

        if content is None:
            return None

        content = content.strip()
        if '0' in content:
            return 'not-abandoned'
        elif '1' in content:
            return 'abandoned'
        return None

    except Exception as e:
        if '429' in str(e).lower() or 'rate limit' in str(e).lower():
            pass
        else:
            print(f"Error classifying {Path(image_path).name} with {model_id}: {e}")
        return None

def load_test_images():
    """Load all test images and their ground truth labels"""
    images = {'abandoned': [], 'not-abandoned': []}
    
    if not TEST_IMAGES_DIR.exists():
        print(f"Warning: Test images directory not found at {TEST_IMAGES_DIR}")
        return images
    
    for label in ['abandoned', 'not-abandoned']:
        label_dir = TEST_IMAGES_DIR / label
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
        prediction = classify_image(model_id, image_path)
        
        if prediction is not None:
            results['predictions'][true_label].append({
                'image': Path(image_path).name,
                'prediction': prediction,
                'answer': true_label
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
        
        total = sum(len(preds) for preds in predictions.values())
        correct = sum(1 for preds in predictions.values() for p in preds if p['prediction'] == p['answer'])
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"\n{model}:")
        print(f"  Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")

def load_existing_results():
    """Load existing results from results.json if it exists"""
    output_file = WORKSPACE_ROOT / "results.json"
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
    output_file = WORKSPACE_ROOT / "results.json"
    
    existing_results = []
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                existing_results = data.get('model_results', [])
        except Exception:
            pass
    
    existing_models = {r['model'] for r in existing_results}
    for result in all_results:
        if result['model'] not in existing_models:
            existing_results.append(result)
    
    try:
        with open(output_file, 'w') as f:
            json.dump({'model_results': existing_results}, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main evaluation pipeline"""
    # Validate API key
    if not API_KEY:
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
        print("Error: No test images found in", TEST_IMAGES_DIR)
        return
    
    if not models:
        print("Error: No models found in models.json")
        return
    
    existing_models = load_existing_results()
    if existing_models:
        print(f"\nFound existing results for: {', '.join(existing_models)}")
    
    models_to_evaluate = [m for m in models if m not in existing_models]
    if not models_to_evaluate:
        print("\nAll models have already been evaluated. Use --force to re-evaluate.")
        return
    
    print(f"\nEvaluating {len(models_to_evaluate)} new model(s)...")
    
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