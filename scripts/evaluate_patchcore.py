"""Evaluate PatchCore on the untouched official MVTec test split."""

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
from lightshift.models.patchcore import build_patchcore


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

        scores.extend(output.pred_score.detach().cpu().reshape(-1).tolist())
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
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(4)

    checkpoint_path = (
        Path("results/patchcore")
        / args.category
        / "patchcore_baseline.pt"
    )

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

    # Validation normals are used only to determine a provisional threshold.
    validation_samples = dataset.load_split("validation")

    validation_scores, _, _, _ = predict_samples(
        model,
        validation_samples,
        args.batch_size,
    )

    threshold = float(np.quantile(validation_scores, 0.95))

    # The official test set is evaluated only after fixing the threshold.
    test_samples = dataset.load_test()

    scores, labels, masks, anomaly_maps = predict_samples(
        model,
        test_samples,
        args.batch_size,
    )

    predictions = (scores > threshold).astype(int)

    results = {
        "category": args.category,
        "threshold": threshold,
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
        "confusion_matrix": confusion_matrix(
            labels, predictions
        ).tolist(),
    }

    pixel_labels = masks.reshape(-1)
    pixel_scores = anomaly_maps.reshape(-1)

    if np.unique(pixel_labels).size == 2:
        results["pixel_auroc"] = float(
            roc_auc_score(pixel_labels, pixel_scores)
        )
    else:
        results["pixel_auroc"] = None

    output_dir = Path("results/evaluation") / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "baseline_metrics.json"
    output_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\n--- PATCHCORE EVALUATION ---")
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
