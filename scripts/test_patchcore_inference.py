"""Smoke test: reload PatchCore and predict official test images."""

import torch

from lightshift.data.dataset import MVTecDataset
from lightshift.models.patchcore import build_patchcore


def main():
    torch.set_num_threads(4)

    checkpoint_path = (
        "results/patchcore/bottle/patchcore_baseline.pt"
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
        category="bottle",
    )

    samples = dataset.load_test()

    normal = next(s for s in samples if s["label"] == 0)
    anomaly = next(s for s in samples if s["label"] == 1)

    def predict(sample):
        image = torch.from_numpy(sample["image"].copy())
        image = image.permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0)

        with torch.no_grad():
            return model(image)

    normal_result = predict(normal)
    anomaly_result = predict(anomaly)

    print("Checkpoint loaded successfully.")
    print("Normal image:", normal["path"])
    print("Anomaly image:", anomaly["path"])

    print("\nNormal prediction:")
    print("Score:", normal_result.pred_score)
    print("Anomaly map:", normal_result.anomaly_map.shape)

    print("\nAnomaly prediction:")
    print("Score:", anomaly_result.pred_score)
    print("Anomaly map:", anomaly_result.anomaly_map.shape)


if __name__ == "__main__":
    main()
