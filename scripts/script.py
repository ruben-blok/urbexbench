import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

model_id = "google/gemma-3-4b-it"
api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

def process_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')

def classify_image(image_path):
    image_data = process_image_to_base64(image_path)
    image_url = f"data:image/png;base64,{image_data}"

    response = client.chat.completions.create(
        model=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Is this location abandoned? Reply ONLY with 0 (not-abandoned) or 1 (abandoned). Target location is near center."}
            ],
        }],
        extra_body={"reasoning": {"enabled": True}}
    )
    
    content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
    return 'abandoned' if '1' in content else 'not-abandoned' if '0' in content else None

def main():
    test_dir = os.path.join(os.path.dirname(__file__), "..", "img")
    predictions = {'abandoned': [], 'not-abandoned': []}
    
    # Voor elk label
    for label in predictions:
        label_dir = os.path.join(test_dir, label)
        if os.path.exists(label_dir):
            # Voor elke image
            for image_file in sorted(os.listdir(label_dir)):
                if image_file.endswith('.png'):
                    image_path = os.path.join(label_dir, image_file)

                    # Maak prediction
                    prediction = classify_image(image_path)
                    print(f"Done processing: {image_file}")

                    # Sla de prediction op
                    if prediction:
                        predictions[label].append({'image': image_file, 'prediction': prediction})

    # Print resultaten
    total = sum(len(p) for p in predictions.values())
    correct = sum(1 for label, preds in predictions.items() for p in preds if p['prediction'] == label)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{model_id}:")
    print(f"  Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()