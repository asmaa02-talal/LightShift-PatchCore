"""Train illumination-augmented PatchCore with a fixed memory-bank budget."""

import argparse
from pathlib import Path

import numpy as np
import torch
from anomalib.engine import Engine
from torch.utils.data import DataLoader

from lightshift.data.dataset import MVTecDataset
from lightshift.data.augmented_dataset import IlluminationAugmentedDataset
from lightshift.data.torch_dataset import (
    MVTecTorchDataset,
    anomalib_collate_fn,
)
from lightshift.models.coreset import FixedBudgetPatchcore


MEMORY_BUDGETS = {
    "bottle": 14950,
    "carpet": 20070,
    "hazelnut": 27955,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle",
                        choices=MEMORY_BUDGETS.keys())
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = MVTecDataset(
        root="data/raw/MVTecAD",
        category=args.category,
    )

    train_samples = dataset.load_train()

    augmented_samples = IlluminationAugmentedDataset(
        samples=train_samples,
        seed=args.seed,
        augmentations_per_image=1,
    )

    train_dataset = MVTecTorchDataset(
        [augmented_samples[i] for i in range(len(augmented_samples))]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=anomalib_collate_fn,
    )

    target_budget = MEMORY_BUDGETS[args.category]

    print(f"Category: {args.category}")
    print(f"Original training images: {len(train_samples)}")
    print(f"Training views: {len(train_dataset)}")
    print(f"Target memory bank budget: {target_budget}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print("Device: CPU")

    model = FixedBudgetPatchcore(
        target_memory_bank_size=target_budget,
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_processor=True,
        post_processor=True,
        evaluator=True,
        visualizer=False,
    )

    engine = Engine(
        default_root_dir="results/patchcore_b1",
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    print("\nStarting B1 training...")

    engine.fit(
        model=model,
        train_dataloaders=train_loader,
    )

    memory_bank = model.model.memory_bank

    if memory_bank is None or memory_bank.numel() == 0:
        raise RuntimeError("B1 memory bank was not created.")

    if memory_bank.shape[0] != target_budget:
        raise RuntimeError(
            f"Expected {target_budget} vectors, got {memory_bank.shape[0]}"
        )

    output_dir = Path("results/patchcore_b1") / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "patchcore_b1.pt"

    torch.save(
        {
            "category": args.category,
            "state_dict": model.state_dict(),
            "memory_bank": memory_bank.cpu(),
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "num_neighbors": 9,
            "target_memory_bank_size": target_budget,
            "train_images": len(train_samples),
            "training_views": len(train_dataset),
            "seed": args.seed,
            "method": "b1_invariant_memory",
        },
        checkpoint_path,
    )

    print("\nB1 training completed.")
    print(f"Memory bank shape: {tuple(memory_bank.shape)}")
    print(f"Memory bank vectors: {memory_bank.shape[0]}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
