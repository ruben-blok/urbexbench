import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "results.json"
OUTPUT_FILE = ROOT / "accuracy_vs_cost.svg"
X_AXIS_SCALE = 10000


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_cost(value):
    if value is None:
        return "N/A"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def average_message_cost(predictions):
    costs = []
    for preds in predictions.values():
        for prediction in preds:
            cost = prediction.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                costs.append(float(cost))

    if not costs:
        return None

    return sum(costs) / len(costs)


def nice_number(value, round_up=False):
    if value <= 0:
        return 1

    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)

    if round_up:
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
    else:
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10

    return nice_fraction * (10 ** exponent)


def build_scatter_svg(stats, output_file: Path):
    if not stats:
        print("No models with a known average cost found; skipping plot generation.")
        return

    width = 1200
    height = 800
    left = 110
    right = 40
    top = 80
    bottom = 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_mid = left + plot_width / 2
    y_mid = top + plot_height / 2

    costs = [s["avg_cost"] * X_AXIS_SCALE for s in stats]
    accuracies = [s["accuracy"] for s in stats]

    x_min = min(costs)
    x_max = max(costs)
    x_range = x_max - x_min
    if x_range == 0:
        x_range = x_min if x_min > 0 else 1e-6
    x_padding = max(x_range * 0.1, 1e-6)
    x_min = max(0, x_min - x_padding)
    x_max = x_max + x_padding
    x_tick_step = nice_number((x_max - x_min) / 8, round_up=True)
    x_min = math.floor(x_min / x_tick_step) * x_tick_step
    x_max = math.ceil(x_max / x_tick_step) * x_tick_step

    y_min = 0
    y_max = 100

    def x_to_px(value):
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value):
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_height

    def short_label(model):
        return model.split("/")[-1]

    def x_tick_labels():
        tick = x_min
        labels = []
        while tick <= x_max + (x_tick_step / 1000):
            labels.append(round(tick, 10))
            tick = round(tick + x_tick_step, 10)
        return labels

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.2f}" y="42" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="#111827">'
        f'Model Accuracy vs Average Cost / Message (x{X_AXIS_SCALE})</text>',
    ]

    # Quadrant backgrounds.
    svg.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_width / 2:.2f}" height="{plot_height / 2:.2f}" fill="#dcfce7"/>',
            f'<rect x="{x_mid:.2f}" y="{top}" width="{plot_width / 2:.2f}" height="{plot_height / 2:.2f}" fill="#f9fafb"/>',
            f'<rect x="{left}" y="{y_mid:.2f}" width="{plot_width / 2:.2f}" height="{plot_height / 2:.2f}" fill="#f9fafb"/>',
            f'<rect x="{x_mid:.2f}" y="{y_mid:.2f}" width="{plot_width / 2:.2f}" height="{plot_height / 2:.2f}" fill="#f9fafb"/>',
            f'<text x="{left + 16}" y="{top + 24}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#166534">Most attractive quadrant</text>',
        ]
    )

    # Axis tick labels.
    for tick in range(0, 101, 20):
        y = y_to_px(tick)
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">'
            f'{tick}%</text>'
        )

    for tick in x_tick_labels():
        x = x_to_px(tick)
        svg.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">'
            f'{escape(format_cost(tick))}</text>'
        )

    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{x_mid:.2f}" y1="{top}" x2="{x_mid:.2f}" y2="{top + plot_height}" stroke="#d1d5db" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{y_mid:.2f}" x2="{left + plot_width}" y2="{y_mid:.2f}" stroke="#d1d5db" stroke-width="1.5"/>',
            f'<text x="{width / 2:.2f}" y="{height - 28}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#111827">Average Cost / Message (x{X_AXIS_SCALE})</text>',
            (
                f'<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
                'font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#111827" '
                'transform="rotate(-90 28 '
                f'{top + plot_height / 2:.2f})">Accuracy (%)</text>'
            ),
        ]
    )

    for stat in stats:
        scaled_cost = stat["avg_cost"] * X_AXIS_SCALE
        x = x_to_px(scaled_cost)
        y = y_to_px(stat["accuracy"])
        label = escape(short_label(stat["model"]))
        tooltip = escape(
            f"{stat['model']} | accuracy {stat['accuracy']:.2f}% | avg cost {format_cost(stat['avg_cost'])} | x{X_AXIS_SCALE} {format_cost(scaled_cost)}"
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

    svg.append('</svg>')

    output_file.write_text("\n".join(svg), encoding="utf-8")
    print(f"Saved scatter plot to {output_file}")


def calculate_stats():
    if not RESULTS_FILE.exists():
        print(f"Error: results.json not found at {RESULTS_FILE}")
        return

    try:
        data = load_json(RESULTS_FILE)
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
    print(f"{'Model':<45} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Avg Cost / Msg':<16}")
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

        avg_cost = average_message_cost(predictions)
        has_avg_cost = avg_cost is not None

        stats.append({
            "model": model,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "avg_cost": avg_cost,
            "has_avg_cost": has_avg_cost,
        })

    stats.sort(key=lambda x: x["accuracy"], reverse=True)

    for s in stats:
        cost_text = format_cost(s["avg_cost"]) if s["has_avg_cost"] else "N/A"
        print(
            f"{s['model']:<45} {s['correct']:<10} {s['total']:<10} "
            f"{s['accuracy']:.2f}% {cost_text:<12}"
        )

    print("-" * 92)
    print(f"Total models evaluated: {len(stats)}")

    plotted_stats = [s for s in stats if s["has_avg_cost"]]
    skipped = len(stats) - len(plotted_stats)

    if skipped:
        print(f"Models without saved cost data: {skipped}")

    build_scatter_svg(plotted_stats, OUTPUT_FILE)


if __name__ == "__main__":
    calculate_stats()
