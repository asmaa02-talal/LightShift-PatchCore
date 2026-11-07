
"""Generate illumination robustness plots for LightShift-PatchCore."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("results/illumination_evaluation/baseline/carpet")
INPUT_FILE = ROOT / "illumination_summary.json"
OUTPUT_DIR = ROOT / "plots"

TRANSFORMS = [
    "brightness",
    "contrast",
    "gamma",
    "color_temperature",
]

DISPLAY_NAMES = {
    "brightness": "Brightness",
    "contrast": "Contrast",
    "gamma": "Gamma",
    "color_temperature": "Color Temperature",
}


def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Summary file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_metric(results, metric, ylabel, title, filename):
    plt.figure(figsize=(10, 6))

    for transform in TRANSFORMS:
        selected = [
            row for row in results
            if row["transform"] == transform
        ]

        selected.sort(key=lambda row: row["severity"])

        severities = [row["severity"] for row in selected]
        values = [row[metric] for row in selected]

        plt.plot(
            severities,
            values,
            marker="o",
            linewidth=2,
            markersize=6,
            label=DISPLAY_NAMES[transform],
        )

    plt.xlabel("Illumination severity", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()

    if len(results) != 20:
        raise ValueError(
            f"Expected 20 experiments, found {len(results)}"
        )

    plot_metric(
        results=results,
        metric="fpr",
        ylabel="False Positive Rate (FPR)",
        title="PatchCore Baseline - FPR under Illumination Changes",
        filename="fpr_by_severity.png",
    )

    plot_metric(
        results=results,
        metric="f1",
        ylabel="F1-score",
        title="PatchCore Baseline - F1-score under Illumination Changes",
        filename="f1_by_severity.png",
    )

    print("\nIllumination plots generated successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
