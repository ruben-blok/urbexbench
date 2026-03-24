import json
from pathlib import Path

def calculate_stats():
    results_file = Path(__file__).parent.parent / "results.json"
    
    if not results_file.exists():
        print(f"Error: results.json not found at {results_file}")
        return
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    model_results = data.get('model_results', [])
    
    if not model_results:
        print("No model results found.")
        return
    
    print("\n" + "=" * 70)
    print("MODEL ACCURACY STATISTICS")
    print("=" * 70)
    print(f"{'Model':<45} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 70)
    
    stats = []
    for result in model_results:
        model = result['model']
        predictions = result['predictions']
        
        total = sum(len(preds) for preds in predictions.values())
        correct = sum(
            1 for preds in predictions.values() 
            for p in preds if p['prediction'] == p['answer']
        )
        accuracy = (correct / total * 100) if total > 0 else 0
        
        stats.append({
            'model': model,
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        })
    
    stats.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for s in stats:
        print(f"{s['model']:<45} {s['correct']:<10} {s['total']:<10} {s['accuracy']:.2f}%")
    
    print("-" * 70)
    print(f"Total models evaluated: {len(stats)}")

if __name__ == "__main__":
    calculate_stats()