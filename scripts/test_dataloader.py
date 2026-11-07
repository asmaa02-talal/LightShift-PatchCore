
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from lightshift.data.dataset import MVTecDataset


class MVTecTorchDataset(Dataset):
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
            "mask": mask,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "path": sample["path"],
        }


root = Path("data/raw/MVTecAD")

for category in ["bottle", "carpet", "hazelnut"]:
    dataset = MVTecDataset(root, category)
    samples = dataset.load_train()

    torch_dataset = MVTecTorchDataset(samples)
    loader = DataLoader(torch_dataset, batch_size=8, shuffle=True)

    batch = next(iter(loader))

    print(f"\n--- {category.upper()} ---")
    print("Images:", len(torch_dataset))
    print("Batch shape:", batch["image"].shape)
    print("Mask shape:", batch["mask"].shape)
    print("Labels:", batch["label"].shape)
    print("PyTorch device:", "CPU")
