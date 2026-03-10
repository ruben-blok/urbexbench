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
SCRIPT_DIR = Path(__file__).parent.absolute()
WORKSPACE_ROOT = SCRIPT_DIR.parent
MODELS_FILE = WORKSPACE_ROOT / "models.json"
TEST_IMAGES_DIR = WORKSPACE_ROOT / "images" / "test"
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
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with open(image_path, 'rb') as image_file:
            return base64.standard_b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise IOError(f"Failed to read image {image_path}: {e}")

def get_image_media_type(image_path):
    """Determine the media type based on file extension"""
    ext = Path(image_path).suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return media_types.get(ext, 'image/jpeg')

def classify_image(model_id, image_path):
    """Send image to model API for classification using OpenAI SDK"""
    try:
        image_data = encode_image_to_base64(image_path)
        media_type = get_image_media_type(image_path)
        
        # Build a data URL for the image so the responses API can ingest it
        image_url = f"data:{media_type};base64,{image_data}"

        # Use OpenAI SDK to send the request
        response = client.responses.create(
            model=model_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url},
                        {"type": "input_text", "text": "Is this location abandoned? Reply ONLY with '0' (not abandoned) or '1' (abandoned)."}
                    ],
                }
            ],
        )

        # Extract textual output from the response
        content = None
        if hasattr(response, "output_text") and response.output_text:
            content = response.output_text
        else:
            # Fallback: inspect output list
            for item in getattr(response, "output", []):
                if item.get("type") == "output_text":
                    content = item.get("text")
                    break

        if content is None:
            return None

        content = content.strip()
        # Look for 0 or 1 in the response
        if '0' in content:
            return 'not-abandoned'
        elif '1' in content:
            return 'abandoned'
        else:
            print(f"Warning: Invalid response '{content}' from {model_id} (expected 0 or 1)")
            return None

    except FileNotFoundError as e:
        print(f"File error: {e}")
        return None
    except Exception as e:
        error_msg = str(e).lower()
        if '429' in error_msg or 'rate limit' in error_msg:
            # Don't print every rate limit error
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
            for image_file in sorted(label_dir.glob('*')):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    images[label].append(str(image_file))
    
    return images

def evaluate_model(model_id, test_images):
    """Evaluate a single model on all test images"""
    results = {
        'model': model_id,
        'predictions': defaultdict(list),
        'total_processed': 0,
        'total_valid': 0,
        'invalid_responses': 0,
        'invalid_images': [],
        'accuracy': 0,
        'correct': 0,
        'per_class': {}
    }
    
    all_images = []
    for label, images in test_images.items():
        for img_path in images:
            all_images.append((label, img_path))
    
    print(f"\nEvaluating {model_id} on {len(all_images)} images...")
    
    for true_label, image_path in tqdm(all_images, desc=model_id):
        prediction = classify_image(model_id, image_path)
        results['total_processed'] += 1
        
        if prediction is None:
            # Track invalid response
            results['invalid_responses'] += 1
            results['invalid_images'].append(image_path)
        else:
            # Valid response - track prediction
            results['total_valid'] += 1
            results['predictions'][true_label].append({
                'image': image_path,
                'prediction': prediction,
                'correct': prediction == true_label
            })
            
            if prediction == true_label:
                results['correct'] += 1
    
    # Calculate accuracy based on valid responses only
    if results['total_valid'] > 0:
        results['accuracy'] = results['correct'] / results['total_valid']
    
    for label in ['abandoned', 'not-abandoned']:
        class_predictions = results['predictions'][label]
        if class_predictions:
            correct = sum(1 for p in class_predictions if p['correct'])
            results['per_class'][label] = {
                'accuracy': correct / len(class_predictions),
                'total': len(class_predictions),
                'correct': correct
            }
    
    return results

def print_results(all_results):
    """Print evaluation results in a formatted table"""
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Sort by accuracy
    sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'Model':<50} {'Accuracy':<12} {'Valid/Total':<15}")
    print("-"*80)
    
    for result in sorted_results:
        accuracy_pct = result['accuracy'] * 100
        valid = result['total_valid']
        total = result['total_processed']
        invalid = result['invalid_responses']
        status = f" ({invalid} invalid)" if invalid > 0 else ""
        print(f"{result['model']:<50} {accuracy_pct:>6.2f}%       {result['correct']}/{valid} ({total}){status}")
    
    print("\n" + "="*80)
    print("PER-CLASS BREAKDOWN")
    print("="*80)
    
    for result in sorted_results:
        print(f"\n{result['model']}:")
        if result['invalid_responses'] > 0:
            print(f"  Invalid responses: {result['invalid_responses']}/{result['total_processed']}")
        for label, metrics in result['per_class'].items():
            acc = metrics['accuracy'] * 100
            print(f"  {label:<20} {acc:>6.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # Calculate overall statistics
    total_processed = sum(r['total_processed'] for r in all_results)
    total_valid = sum(r['total_valid'] for r in all_results)
    total_correct = sum(r['correct'] for r in all_results)
    total_invalid = sum(r['invalid_responses'] for r in all_results)
    total_wrong = total_valid - total_correct
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS (ACROSS ALL MODELS)")
    print("="*80)
    print(f"Total predictions processed: {total_processed}")
    print(f"  ✓ Correct:   {total_correct:>6} ({(total_correct/total_processed)*100:>6.2f}%)")
    print(f"  ✗ Wrong:     {total_wrong:>6} ({(total_wrong/total_processed)*100:>6.2f}%)")
    print(f"  ⚠ Invalid:   {total_invalid:>6} ({(total_invalid/total_processed)*100:>6.2f}%)")

def save_results(all_results):
    """Save detailed results to JSON file"""
    output_file = WORKSPACE_ROOT / "results.json"
    
    # Convert defaultdict to regular dict for JSON serialization
    serializable_results = []
    for result in all_results:
        r = dict(result)
        r['predictions'] = {k: v for k, v in r['predictions'].items()}
        serializable_results.append(r)
    
    # Calculate overall statistics
    total_processed = sum(r['total_processed'] for r in all_results)
    total_valid = sum(r['total_valid'] for r in all_results)
    total_correct = sum(r['correct'] for r in all_results)
    total_invalid = sum(r['invalid_responses'] for r in all_results)
    total_wrong = total_valid - total_correct
    
    overall_stats = {
        'total_processed': total_processed,
        'total_correct': total_correct,
        'correct_percentage': (total_correct / total_processed * 100) if total_processed > 0 else 0,
        'total_wrong': total_wrong,
        'wrong_percentage': (total_wrong / total_processed * 100) if total_processed > 0 else 0,
        'total_invalid': total_invalid,
        'invalid_percentage': (total_invalid / total_processed * 100) if total_processed > 0 else 0
    }
    
    output_data = {
        'overall_statistics': overall_stats,
        'model_results': serializable_results
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
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
    
    # Evaluate each model
    all_results = []
    for i, model_id in enumerate(models, 1):
        try:
            print(f"\n[{i}/{len(models)}] Evaluating {model_id}...")
            result = evaluate_model(model_id, test_images)
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating model {model_id}: {e}")
            continue
    
    # Print and save results
    if all_results:
        print_results(all_results)
        save_results(all_results)
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    main()