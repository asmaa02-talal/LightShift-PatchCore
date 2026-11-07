from pathlib import Path
import random
import json

ROOT = Path("data/raw/MVTecAD")
OUTPUT = Path("data/splits")

CATEGORIES = ["bottle", "carpet", "hazelnut"]

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
CAL_RATIO = 0.20

random.seed(SEED)


def create_category_splits(category: str):
    train_dir = ROOT / category / "train" / "good"
    images = sorted(train_dir.glob("*.png"))

    if not images:
        raise RuntimeError(f"No images found in {train_dir}")

    images = [str(path.relative_to(ROOT)) for path in images]
    random.shuffle(images)

    total = len(images)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)

    train = images[:n_train]
    validation = images[n_train:n_train + n_val]
    calibration = images[n_train + n_val:]

    output_dir = OUTPUT / category
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train,
        "validation": validation,
        "calibration": calibration,
    }

    for name, paths in splits.items():
        output_file = output_dir / f"{name}.txt"
        output_file.write_text(
            "\n".join(paths) + "\n",
            encoding="utf-8",
        )

    all_paths = train + validation + calibration

    assert len(all_paths) == total
    assert len(set(all_paths)) == total

    return {
        "category": category,
        "total": total,
        "train": len(train),
        "validation": len(validation),
        "calibration": len(calibration),
        "seed": SEED,
    }


def main():
    results = []

    for category in CATEGORIES:
        result = create_category_splits(category)
        results.append(result)

        print(f"\n--- {category.upper()} ---")
        print("Train:", result["train"])
        print("Validation:", result["validation"])
        print("Calibration:", result["calibration"])
        print("Total:", result["total"])

    OUTPUT.mkdir(parents=True, exist_ok=True)

    metadata = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VAL_RATIO,
        "calibration_ratio": CAL_RATIO,
        "categories": results,
    }

    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nAll splits created successfully.")


if __name__ == "__main__":
    main()
