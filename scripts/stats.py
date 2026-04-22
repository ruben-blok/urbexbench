import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "results.json"
PRICE_FILE = ROOT / "price.json"
OUTPUT_FILE = ROOT / "accuracy_vs_cost.svg"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_cost(value):
    if value is None:
        return "N/A"
    return f"{value:.2e}"


def build_scatter_svg(stats, output_file: Path):
    if not stats:
        print("No models with a known cost found; skipping plot generation.")
        return

    width = 1200
    height = 800
    left = 110
    right = 40
    top = 80
    bottom = 120
    plot_width = width - left - right
    plot_height = height - top - bottom

    costs = [s["cost"] for s in stats]
    accuracies = [s["accuracy"] for s in stats]

    x_min = min(costs)
    x_max = max(costs)
    x_range = x_max - x_min
    if x_range == 0:
        x_range = x_min if x_min > 0 else 1e-6
    x_padding = max(x_range * 0.1, 1e-6)
    x_min = max(0, x_min - x_padding)
    x_max = x_max + x_padding

    y_min = 0
    y_max = 100

    def x_to_px(value):
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value):
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_height

    def short_label(model):
        return model.split("/")[-1]

    def x_tick_labels():
        return [x_min + i * (x_max - x_min) / 4 for i in range(5)]

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.2f}" y="42" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="#111827">'
        'Model Accuracy vs Cost</text>',
        f'<text x="{width / 2:.2f}" y="68" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#4b5563">'
        'Only models with a known cost are plotted</text>',
    ]

    # Grid and axes.
    for tick in range(0, 101, 20):
        y = y_to_px(tick)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">'
            f'{tick}%</text>'
        )

    for tick in x_tick_labels():
        x = x_to_px(tick)
        svg.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">'
            f'{escape(format_cost(tick))}</text>'
        )

    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<text x="{width / 2:.2f}" y="{height - 28}" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#111827">Cost</text>',
            (
                f'<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
                'font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#111827" '
                'transform="rotate(-90 28 '
                f'{top + plot_height / 2:.2f})">Accuracy (%)</text>'
            ),
        ]
    )

    for stat in stats:
        x = x_to_px(stat["cost"])
        y = y_to_px(stat["accuracy"])
        label = escape(short_label(stat["model"]))
        tooltip = escape(
            f"{stat['model']} | accuracy {stat['accuracy']:.2f}% | cost {format_cost(stat['cost'])}"
        )
        if x > left + plot_width - 140:
            label_x = x - 12
            anchor = "end"
        else:
            label_x = x + 12
            anchor = "start"
        label_y = y - 10 if y > top + 24 else y + 16
        svg.append(
            f'<g><title>{tooltip}</title><circle cx="{x:.2f}" cy="{y:.2f}" r="7" '
            'fill="#2563eb" stroke="#ffffff" stroke-width="2"/>'
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#111827">'
            f'{label}</text></g>'
        )

    svg.append(
        f'<text x="{left}" y="{height - 58}" font-family="Arial, Helvetica, sans-serif" '
        'font-size="12" fill="#6b7280">'
        f'Plotted {len(stats)} models with a known cost</text>'
    )
    svg.append('</svg>')

    output_file.write_text("\n".join(svg), encoding="utf-8")
    print(f"Saved scatter plot to {output_file}")


def calculate_stats():
    if not RESULTS_FILE.exists():
        print(f"Error: results.json not found at {RESULTS_FILE}")
        return

    if not PRICE_FILE.exists():
        print(f"Error: price.json not found at {PRICE_FILE}")
        return

    try:
        data = load_json(RESULTS_FILE)
        prices = load_json(PRICE_FILE)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return

    model_results = data.get("model_results", [])

    if not model_results:
        print("No model results found.")
        return

    print("\n" + "=" * 92)
    print("MODEL ACCURACY STATISTICS")
    print("=" * 92)
    print(f"{'Model':<45} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Cost':<12}")
    print("-" * 92)

    stats = []
    for result in model_results:
        model = result["model"]
        predictions = result["predictions"]

        total = sum(len(preds) for preds in predictions.values())
        correct = sum(
            1 for preds in predictions.values()
            for p in preds if p["prediction"] == p["answer"]
        )
        accuracy = (correct / total * 100) if total > 0 else 0

        cost = prices.get(model)
        has_cost = isinstance(cost, (int, float)) and cost > 0

        stats.append({
            "model": model,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "cost": cost,
            "has_cost": has_cost,
        })

    stats.sort(key=lambda x: x["accuracy"], reverse=True)

    for s in stats:
        cost_text = format_cost(s["cost"]) if s["has_cost"] else "N/A"
        print(
            f"{s['model']:<45} {s['correct']:<10} {s['total']:<10} "
            f"{s['accuracy']:.2f}% {cost_text:<12}"
        )

    print("-" * 92)
    print(f"Total models evaluated: {len(stats)}")

    plotted_stats = [s for s in stats if s["has_cost"]]
    skipped = len(stats) - len(plotted_stats)

    if skipped:
        print(f"Models without a known cost: {skipped}")

    build_scatter_svg(plotted_stats, OUTPUT_FILE)


if __name__ == "__main__":
    calculate_stats()
