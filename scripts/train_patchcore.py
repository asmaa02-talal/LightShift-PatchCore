"""Train the PatchCore baseline on an explicit MVTec training split."""

import argparse
from pathlib import Path

import torch
from anomalib.engine import Engine
from torch.utils.data import DataLoader

from lightshift.data.dataset import MVTecDataset
from lightshift.data.torch_dataset import (
    MVTecTorchDataset,
    anomalib_collate_fn,
)
from lightshift.models.patchcore import build_patchcore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(4)

    dataset = MVTecDataset(
        root="data/raw/MVTecAD",
        category=args.category,
    )

    train_samples = dataset.load_train()
    train_dataset = MVTecTorchDataset(train_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=anomalib_collate_fn,
    )

    print(f"Category: {args.category}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print("Device: CPU")

    model = build_patchcore()

    engine = Engine(
        default_root_dir="results/patchcore",
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    print("\nStarting PatchCore training...")

    engine.fit(
        model=model,
        train_dataloaders=train_loader,
    )

    memory_bank = model.model.memory_bank

    if memory_bank is None or memory_bank.numel() == 0:
        raise RuntimeError("PatchCore memory bank was not created.")

    output_dir = Path("results/patchcore") / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "patchcore_baseline.pt"

    torch.save(
        {
            "category": args.category,
            "state_dict": model.state_dict(),
            "memory_bank": memory_bank.cpu(),
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
            "train_images": len(train_dataset),
        },
        checkpoint_path,
    )

    print("\nTraining completed.")
    print(f"Memory bank shape: {tuple(memory_bank.shape)}")
    print(f"Memory bank vectors: {memory_bank.shape[0]}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
