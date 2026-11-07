
"""Evaluate PatchCore robustness under illumination changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from lightshift.data.dataset import MVTecDataset
from lightshift.data.illumination import (
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    adjust_color_temperature,
)
from lightshift.models.patchcore import build_patchcore


SEVERITY_VALUES = {
    "brightness": [0.7, 0.8, 0.9, 1.1, 1.3],
    "contrast": [0.7, 0.8, 0.9, 1.1, 1.3],
    "gamma": [0.7, 0.85, 1.15, 1.3, 1.5],
    "color_temperature": [0.1, 0.2, 0.3, 0.4, 0.5],
}


def apply_illumination(image, transform, severity):
    """Apply one illumination transformation."""

    value = SEVERITY_VALUES[transform][severity - 1]

    if transform == "brightness":
        return adjust_brightness(image, value)

    if transform == "contrast":
        return adjust_contrast(image, value)

    if transform == "gamma":
        return adjust_gamma(image, value)

    if transform == "color_temperature":
        return adjust_color_temperature(image, value)

    raise ValueError(f"Unknown transformation: {transform}")


def predict_samples(model, samples, batch_size=4):
    scores = []
    labels = []
    masks = []
    anomaly_maps = []

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]

        images = [
            torch.from_numpy(sample["image"].copy())
            .permute(2, 0, 1)
            .float() / 255.0
            for sample in batch_samples
        ]

        batch = torch.stack(images)

        with torch.inference_mode():
            output = model(batch)

        scores.extend(
            output.pred_score.detach().cpu().reshape(-1).tolist()
        )

        anomaly_maps.extend(
            output.anomaly_map.detach().cpu().numpy()
        )

        labels.extend(sample["label"] for sample in batch_samples)
        masks.extend(sample["mask"] for sample in batch_samples)

    return (
        np.asarray(scores),
        np.asarray(labels),
        np.asarray(masks),
        np.asarray(anomaly_maps),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--category",
        choices=["bottle", "carpet", "hazelnut"],
        required=True,
    )

    parser.add_argument(
        "--transform",
        choices=list(SEVERITY_VALUES.keys()),
        required=True,
    )

    parser.add_argument(
        "--severity",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
    )

    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    torch.set_num_threads(4)

    # Load the Baseline checkpoint.
    checkpoint_path = (
        Path("results/patchcore")
        / args.category
        / "patchcore_baseline.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = build_patchcore()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = MVTecDataset(
        root="data/raw/MVTecAD",
        category=args.category,
    )

    # 1. Determine threshold using ORIGINAL validation normals.
    validation_samples = dataset.load_split("validation")

    validation_scores, _, _, _ = predict_samples(
        model,
        validation_samples,
        args.batch_size,
    )

    threshold = float(np.quantile(validation_scores, 0.95))

    # 2. Load official test images.
    test_samples = dataset.load_test()

    # 3. Apply illumination transformation to test images only.
    transformed_samples = []

    for sample in test_samples:
        transformed = {
            **sample,
            "image": apply_illumination(
                sample["image"],
                args.transform,
                args.severity,
            ),
        }

        transformed_samples.append(transformed)

    # 4. Evaluate using the unchanged validation threshold.
    scores, labels, masks, anomaly_maps = predict_samples(
        model,
        transformed_samples,
        args.batch_size,
    )

    predictions = (scores > threshold).astype(int)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else None

    results = {
        "category": args.category,
        "method": "baseline",
        "transform": args.transform,
        "severity": args.severity,
        "transform_value": SEVERITY_VALUES[args.transform][args.severity - 1],
        "threshold_from_original_validation": threshold,
        "image_count": len(labels),
        "normal_count": int((labels == 0).sum()),
        "anomaly_count": int((labels == 1).sum()),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "fpr": fpr,
        "confusion_matrix": matrix.tolist(),
    }

    pixel_labels = masks.reshape(-1)
    pixel_scores = anomaly_maps.reshape(-1)

    if np.unique(pixel_labels).size == 2:
        results["pixel_auroc"] = float(
            roc_auc_score(pixel_labels, pixel_scores)
        )
    else:
        results["pixel_auroc"] = None

    # 5. Save without overwriting Baseline original results.
    output_dir = (
        Path("results/illumination_evaluation")
        / "baseline"
        / args.category
        / args.transform
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"severity_{args.severity}.json"

    output_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\n--- ILLUMINATION EVALUATION ---")

    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
