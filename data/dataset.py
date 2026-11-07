
"""MVTec AD dataset loader with explicit, leakage-safe splits."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def validate_split_ids(
    train_ids: set[str],
    calibration_ids: set[str],
    validation_ids: set[str],
    test_ids: set[str],
) -> None:
    """Prevent data leakage between dataset splits."""
    groups = {
        "train": train_ids,
        "calibration": calibration_ids,
        "validation": validation_ids,
        "test": test_ids,
    }

    names = list(groups)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = groups[left_name] & groups[right_name]
            if overlap:
                raise ValueError(
                    f"Data leakage between {left_name} and "
                    f"{right_name}: {sorted(overlap)[:5]}"
                )


class MVTecDataset:
    """Load one category of the MVTec AD dataset."""

    def __init__(
        self,
        root: str | Path,
        category: str,
        image_size: tuple[int, int] = (256, 256),
        splits_root: str | Path = "data/splits",
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.image_size = image_size
        self.category_path = self.root / category
        self.splits_root = Path(splits_root)

        if not self.category_path.is_dir():
            raise FileNotFoundError(
                f"Category not found: {self.category_path}"
            )

    def _load_ids(self, split: str) -> list[str]:
        """Read image paths from a split file."""
        split_file = self.splits_root / self.category / f"{split}.txt"

        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        return [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve and validate a path relative to the dataset root."""
        path = self.root / relative_path

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        return path

    def _load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                Image.Resampling.BILINEAR,
            )
            return np.asarray(image, dtype=np.uint8)

    def _load_mask(self, path: Path | None) -> np.ndarray:
        if path is None:
            return np.zeros(self.image_size, dtype=np.uint8)

        if not path.is_file():
            raise FileNotFoundError(f"Missing mask: {path}")

        with Image.open(path) as mask:
            mask = mask.convert("L")
            mask = mask.resize(
                (self.image_size[1], self.image_size[0]),
                Image.Resampling.NEAREST,
            )
            return (np.asarray(mask) > 0).astype(np.uint8)

    def load_split(self, split: str) -> list[dict[str, Any]]:
        """Load one of the predefined normal-image splits."""
        if split not in {"train", "validation", "calibration"}:
            raise ValueError(f"Unsupported split: {split}")

        samples = []

        for relative_path in self._load_ids(split):
            path = self._resolve_path(relative_path)

            samples.append({
                "image": self._load_image(path),
                "label": 0,
                "mask": np.zeros(self.image_size, dtype=np.uint8),
                "path": str(path),
                "category": self.category,
            })

        if not samples:
            raise RuntimeError(f"Empty split: {split}")

        return samples

    def load_train(self) -> list[dict[str, Any]]:
        """Load only the training images."""
        return self.load_split("train")

    def load_test(self) -> list[dict[str, Any]]:
        """Load the untouched official MVTec test set."""
        test_dir = self.category_path / "test"
        samples = []

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            is_anomaly = defect_type != "good"

            for path in sorted(defect_dir.glob("*.png")):
                mask_path = None

                if is_anomaly:
                    mask_path = (
                        self.category_path
                        / "ground_truth"
                        / defect_type
                        / f"{path.stem}_mask.png"
                    )

                samples.append({
                    "image": self._load_image(path),
                    "label": int(is_anomaly),
                    "mask": self._load_mask(mask_path),
                    "path": str(path),
                    "category": self.category,
                    "defect_type": defect_type,
                })

        if not samples:
            raise RuntimeError(f"No test images found: {test_dir}")

        return samples
