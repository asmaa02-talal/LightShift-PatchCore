
from pathlib import Path

from lightshift.data.dataset import MVTecDataset

ROOT = Path("data/raw/MVTecAD")
CATEGORIES = ["bottle", "carpet", "hazelnut"]

for category in CATEGORIES:
    dataset = MVTecDataset(ROOT, category)

    train = dataset.load_train()
    test = dataset.load_test()

    normal = sum(sample["label"] == 0 for sample in test)
    anomalous = sum(sample["label"] == 1 for sample in test)
    masks_missing = sum(
        sample["label"] == 1 and not sample["mask"].any()
        for sample in test
    )

    print(f"\n--- {category.upper()} ---")
    print(f"Training normal images : {len(train)}")
    print(f"Test normal images     : {normal}")
    print(f"Test anomalous images  : {anomalous}")
    print(f"Empty anomaly masks    : {masks_missing}")

    if train:
        print(f"Image shape            : {train[0]['image'].shape}")
