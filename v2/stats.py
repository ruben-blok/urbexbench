"""UrbexBench v2 statistics and chart generation.

Reads every run file from results-v2/, prints a ranked leaderboard and
writes an accuracy-vs-cost scatter plot.
"""

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RESULTS_DIR = ROOT / "results-v2"
OUTPUT_FILE = RESULTS_DIR / "accuracy_vs_cost.svg"
X_AXIS_SCALE = 1000
COLOR_REASONING_OFF = "#2563eb"
COLOR_REASONING_ON = "#16a34a"


def load_runs():
    """Load all run files from results-v2/."""
    runs = []
    if not RESULTS_DIR.exists():
        return runs
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception as e:
            print(f"Warning: could not read {path.name}: {e}")
    return runs


def format_cost(value):
    if value is None:
        return "N/A"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_percentage(value):
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text else "0"


def average_message_cost(predictions):
    costs = [
        float(p["cost"])
        for preds in predictions.values()
        for p in preds
        if isinstance(p.get("cost"), (int, float)) and not isinstance(p.get("cost"), bool)
    ]
    return sum(costs) / len(costs) if costs else None


def nice_number(value, round_up=False):
    if value <= 0:
        return 1
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if round_up:
        nice_fraction = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    else:
        nice_fraction = 1 if fraction < 1.5 else 2 if fraction < 3 else 5 if fraction < 7 else 10
    return nice_fraction * (10 ** exponent)


def build_scatter_svg(stats, output_file: Path):
    """Render the accuracy-vs-cost scatter plot as an SVG file."""
    if not stats:
        print("No models with a known average cost found; skipping plot generation.")
        return

    width, height = 1200, 800
    left, right, top, bottom = 110, 40, 80, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_mid = left + plot_width / 2
    y_mid = top + plot_height / 2

    costs = [s["avg_cost"] * X_AXIS_SCALE for s in stats]
    accuracies = [s["accuracy"] for s in stats]

    x_min, x_max = min(costs), max(costs)
    x_range = x_max - x_min or (x_min if x_min > 0 else 1e-6)
    x_padding = max(x_range * 0.1, 1e-6)
    x_min, x_max = max(0, x_min - x_padding), x_max + x_padding
    x_tick_step = nice_number((x_max - x_min) / 8, round_up=True)
    x_min = math.floor(x_min / x_tick_step) * x_tick_step
    x_max = math.ceil(x_max / x_tick_step) * x_tick_step

    y_min, y_max = min(accuracies), max(accuracies)
    y_range = y_max - y_min
    y_padding = y_range * 0.1 if y_range else max(abs(y_max) * 0.1, 1.0)
    y_min, y_max = max(0, y_min - y_padding), min(100, y_max + y_padding)
    y_tick_step = nice_number((y_max - y_min) / 8, round_up=True)
    y_min = max(0, math.floor(y_min / y_tick_step) * y_tick_step)
    y_max = min(100, math.ceil(y_max / y_tick_step) * y_tick_step)

    def x_to_px(value):
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value):
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_height

    def short_label(model):
        return model.split("/")[-1]

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
        f'<rect x="{left}" y="{top}" width="{plot_width / 2:.2f}" height="{plot_height / 2:.2f}" fill="#dcfce7"/>',
        f'<rect x="{x_mid:.2f}" y="{top}" width="{plot_width / 2:.2f}" height="{plot_height / 2:.2f}" fill="#f9fafb"/>',
        f'<rect x="{left}" y="{y_mid:.2f}" width="{plot_width / 2:.2f}" height="{plot_height / 2:.2f}" fill="#f9fafb"/>',
        f'<rect x="{x_mid:.2f}" y="{y_mid:.2f}" width="{plot_width / 2:.2f}" height="{plot_height / 2:.2f}" fill="#f9fafb"/>',
        f'<text x="{left + 16}" y="{top + 24}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#166534">Most attractive quadrant</text>',
    ]

    legend_x = left + plot_width - 236
    svg.append(f'<circle cx="{legend_x}" cy="{top + 18}" r="6" fill="{COLOR_REASONING_OFF}"/>')
    svg.append(f'<text x="{legend_x + 12}" y="{top + 22}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#111827">Reasoning off</text>')
    svg.append(f'<circle cx="{legend_x}" cy="{top + 40}" r="6" fill="{COLOR_REASONING_ON}"/>')
    svg.append(f'<text x="{legend_x + 12}" y="{top + 44}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#111827">Reasoning on (lowest effort)</text>')

    tick = y_min
    while tick <= y_max + (y_tick_step / 1000):
        y = y_to_px(tick)
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">'
            f'{format_percentage(tick)}%</text>'
        )
        tick = round(tick + y_tick_step, 10)

    tick = x_min
    while tick <= x_max + (x_tick_step / 1000):
        x = x_to_px(tick)
        svg.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">'
            f'{escape(format_cost(tick))}</text>'
        )
        tick = round(tick + x_tick_step, 10)

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>')
    svg.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>')
    svg.append(f'<line x1="{x_mid:.2f}" y1="{top}" x2="{x_mid:.2f}" y2="{top + plot_height}" stroke="#d1d5db" stroke-width="1.5"/>')
    svg.append(f'<line x1="{left}" y1="{y_mid:.2f}" x2="{left + plot_width}" y2="{y_mid:.2f}" stroke="#d1d5db" stroke-width="1.5"/>')
    svg.append(f'<text x="{width / 2:.2f}" y="{height - 28}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#111827">Average Cost / Message (x{X_AXIS_SCALE})</text>')
    svg.append(
        f'<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#111827" '
        f'transform="rotate(-90 28 {top + plot_height / 2:.2f})">Accuracy (%)</text>'
    )

    for stat in stats:
        scaled_cost = stat["avg_cost"] * X_AXIS_SCALE
        x, y = x_to_px(scaled_cost), y_to_px(stat["accuracy"])
        effort = stat["reasoning_effort"]
        point_color = COLOR_REASONING_OFF if effort == "none" else COLOR_REASONING_ON
        label = escape(f"{short_label(stat['model'])} ({effort})")
        tooltip = escape(
            f"{stat['model']} (effort={effort}) | provider {stat.get('provider')} | "
            f"accuracy {stat['accuracy']:.2f}% | avg cost {format_cost(stat['avg_cost'])}"
        )
        if x > left + plot_width - 140:
            label_x, anchor = x - 12, "end"
        else:
            label_x, anchor = x + 12, "start"
        label_y = y - 10 if y > top + 24 else y + 16
        svg.append(
            f'<g><title>{tooltip}</title><circle cx="{x:.2f}" cy="{y:.2f}" r="7" '
            f'fill="{point_color}" stroke="#ffffff" stroke-width="2"/>'
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#111827">'
            f'{label}</text></g>'
        )

    svg.append("</svg>")
    output_file.write_text("\n".join(svg), encoding="utf-8")
    print(f"\nSaved scatter plot to {output_file}")


def calculate_stats():
    runs = load_runs()
    if not runs:
        print(f"No run files found in {RESULTS_DIR}.")
        return

    stats = []
    for run in runs:
        predictions = run["predictions"]
        correct = total = 0
        for preds in predictions.values():
            for p in preds:
                if p.get("prediction") is None:
                    continue
                total += 1
                correct += p["prediction"] == p["answer"]
        accuracy = (correct / total * 100) if total else 0
        provider = run.get("provider", {})
        used = provider.get("used") or {}
        provider_label = ",".join(sorted(used)) if used else provider.get("requested")
        stats.append(
            {
                "model": run["model"],
                "reasoning_effort": run.get("reasoning_effort", "none"),
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "avg_cost": average_message_cost(predictions),
                "provider": provider_label,
                "determinism": run.get("determinism"),
            }
        )

    stats.sort(key=lambda x: x["accuracy"], reverse=True)

    print("\n" + "=" * 118)
    print("URBEXBENCH V2 LEADERBOARD")
    print("=" * 118)
    print(f"{'Rank':<5} {'Model':<42} {'Effort':<9} {'Correct':<10} {'Total':<8} {'Accuracy':<10} {'Avg $/msg':<12} {'Provider':<16}")
    print("-" * 118)
    for rank, s in enumerate(stats, 1):
        avg_cost = format_cost(s["avg_cost"]) if s["avg_cost"] is not None else "N/A"
        print(
            f"{rank:<5} {s['model']:<42} {s['reasoning_effort']:<9} {s['correct']:<10} "
            f"{s['total']:<8} {s['accuracy']:.2f}%     {avg_cost:<12} {str(s['provider']):<16}"
        )
    print("-" * 118)
    print(f"Total evaluated runs: {len(stats)}")

    unstable = [s for s in stats if s["determinism"] and not s["determinism"].get("stable")]
    if unstable:
        print("\nWarning: non-deterministic providers detected (temperature 0 output varied):")
        for s in unstable:
            print(f"  - {s['model']} ({s['reasoning_effort']}): {s['determinism'].get('outputs')}")

    print("\nMarkdown leaderboard:")
    print("| Rank | Model | Effort | Accuracy | Provider |")
    print("|------|-------|--------|----------|----------|")
    for rank, s in enumerate(stats, 1):
        print(f"| {rank} | {s['model']} | {s['reasoning_effort']} | {s['accuracy']:.1f}% | {s['provider']} |")

    plotted = [s for s in stats if s["avg_cost"] is not None]
    if len(plotted) != len(stats):
        print(f"\nRuns without saved cost data (not plotted): {len(stats) - len(plotted)}")
    build_scatter_svg(plotted, OUTPUT_FILE)


if __name__ == "__main__":
    calculate_stats()
