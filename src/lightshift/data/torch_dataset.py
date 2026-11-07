"""PyTorch dataset and Anomalib batch collation."""

import torch
from anomalib.data import ImageBatch
from torch.utils.data import Dataset


class MVTecTorchDataset(Dataset):
    """Convert MVTec samples into PyTorch tensors."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = torch.from_numpy(sample["image"].copy())
        image = image.permute(2, 0, 1).float() / 255.0

        mask = torch.from_numpy(sample["mask"].copy()).float()

        return {
            "image": image,
            "gt_label": torch.tensor(sample["label"], dtype=torch.long),
            "gt_mask": mask,
            "image_path": sample["path"],
        }


def anomalib_collate_fn(samples):
    """Build an ImageBatch from individual samples."""
    return ImageBatch(
        image=torch.stack([sample["image"] for sample in samples]),
        gt_label=torch.stack([sample["gt_label"] for sample in samples]),
        gt_mask=torch.stack([sample["gt_mask"] for sample in samples]),
        image_path=[sample["image_path"] for sample in samples],
    )
