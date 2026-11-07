import json
from pathlib import Path

ROOT = Path("results/illumination_evaluation/baseline/carpet")

TRANSFORMS = [
    "brightness",
    "contrast",
    "gamma",
    "color_temperature",
]

rows = []

for transform in TRANSFORMS:
    for severity in range(1, 6):
        path = ROOT / transform / f"severity_{severity}.json"

        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        rows.append({
            "transform": transform,
            "severity": severity,
            "factor": result["transform_value"],
            "image_auroc": result["image_auroc"],
            "f1": result["f1"],
            "precision": result["precision"],
            "recall": result["recall"],
            "fpr": result["fpr"],
            "pixel_auroc": result["pixel_auroc"],
        })

print("\n--- ILLUMINATION SUMMARY: BOTTLE / BASELINE ---\n")

header = (
    f"{'Transform':<20} {'S':<3} {'Factor':<9} "
    f"{'AUROC':<9} {'F1':<9} {'Precision':<11} "
    f"{'Recall':<9} {'FPR':<9} {'Pixel AUROC':<12}"
)
print(header)
print("-" * len(header))

for row in rows:
    print(
        f"{row['transform']:<20} "
        f"{row['severity']:<3} "
        f"{row['factor']:<9.2f} "
        f"{row['image_auroc']:<9.4f} "
        f"{row['f1']:<9.4f} "
        f"{row['precision']:<11.4f} "
        f"{row['recall']:<9.4f} "
        f"{row['fpr']:<9.2%} "
        f"{row['pixel_auroc']:<12.4f}"
    )

output = ROOT / "illumination_summary.json"

with output.open("w", encoding="utf-8") as file:
    json.dump(rows, file, indent=4)

print(f"\nSummary saved to: {output}")
print(f"Total experiments: {len(rows)}")