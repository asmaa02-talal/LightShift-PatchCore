
"""Multi-seed benchmark for Anomalib and matrix-based KCenterGreedy."""

from __future__ import annotations

import time

import numpy as np
import torch
from anomalib.models.components.sampling import KCenterGreedy


SEEDS = [42, 123, 2026]

CANDIDATE_COUNT = 10_000
EMBEDDING_DIM = 1_536
CORESET_SIZE = 1_000
INITIAL_INDEX = 1234


def greedy_anomalib(features, coreset_size, initial_idx):
    """Original Anomalib KCenterGreedy."""

    sampler = KCenterGreedy(
        embedding=torch.empty(features.shape[0], 2),
        sampling_ratio=coreset_size / features.shape[0],
    )

    sampler.features = features
    sampler.n_observations = features.shape[0]
    sampler.coreset_size = coreset_size
    sampler.reset_distances()

    selected = [torch.tensor(initial_idx, device=features.device)]
    idx = selected[0]

    for _ in range(coreset_size - 1):
        sampler.update_distances(idx)
        idx = sampler.get_new_idx()

        sampler.min_distances.scatter_(
            0, idx.unsqueeze(0).unsqueeze(1), 0.0
        )
        selected.append(idx)

    return torch.stack(selected).cpu().tolist()


def greedy_matmul(features, coreset_size, initial_idx):
    """KCenterGreedy using squared distances and matrix multiplication."""

    n = features.shape[0]
    norms = (features ** 2).sum(dim=1)

    min_distances = torch.full(
        (n,),
        float("inf"),
        dtype=features.dtype,
        device=features.device,
    )

    selected = [initial_idx]
    idx = initial_idx

    for _ in range(coreset_size - 1):
        center = features[idx]
        center_norm = torch.dot(center, center)

        distances = (
            norms + center_norm - 2.0 * torch.mv(features, center)
        )

        distances.clamp_(min=0.0)

        min_distances = torch.minimum(min_distances, distances)
        min_distances[idx] = 0.0

        idx = int(torch.argmax(min_distances).item())
        selected.append(idx)

    return selected


def coverage_metrics(features, selected_indices):
    """Measure geometric coverage using Euclidean distances."""

    centers = features[selected_indices]

    candidate_norms = (features ** 2).sum(dim=1)
    center_norms = (centers ** 2).sum(dim=1)

    block_size = 512
    min_distances = []

    for start in range(0, features.shape[0], block_size):
        block = features[start:start + block_size]
        block_norms = candidate_norms[start:start + block_size]

        distances = (
            block_norms[:, None]
            + center_norms[None, :]
            - 2.0 * (block @ centers.T)
        )

        distances.clamp_(min=0.0)
        min_distances.append(distances.min(dim=1).values)

    min_distances = torch.cat(min_distances).sqrt()

    return {
        "mean": min_distances.mean().item(),
        "p95": torch.quantile(min_distances, 0.95).item(),
        "max": min_distances.max().item(),
    }


def run_seed(seed):
    """Run both algorithms with identical data and initialization."""

    print("\n" + "=" * 55)
    print(f"SEED: {seed}")
    print("=" * 55)

    torch.manual_seed(seed)
    np.random.seed(seed)

    embeddings = torch.randn(
        CANDIDATE_COUNT,
        EMBEDDING_DIM,
    )

    projection_sampler = KCenterGreedy(
        embedding=embeddings,
        sampling_ratio=CORESET_SIZE / CANDIDATE_COUNT,
    )

    projection_sampler.model.fit(embeddings)
    features = projection_sampler.model.transform(embeddings)

    # Same initial index for both algorithms.
    initial_idx = INITIAL_INDEX

    # Original Anomalib.
    start = time.perf_counter()

    original_indices = greedy_anomalib(
        features,
        CORESET_SIZE,
        initial_idx,
    )

    original_time = time.perf_counter() - start

    # Matrix multiplication.
    start = time.perf_counter()

    matrix_indices = greedy_matmul(
        features,
        CORESET_SIZE,
        initial_idx,
    )

    matrix_time = time.perf_counter() - start

    original_set = set(original_indices)
    matrix_set = set(matrix_indices)

    common = len(original_set & matrix_set)
    jaccard = common / len(original_set | matrix_set)

    matching_positions = sum(
        a == b
        for a, b in zip(original_indices, matrix_indices)
    )

    # Coverage.
    original_coverage = coverage_metrics(
        features,
        original_indices,
    )

    matrix_coverage = coverage_metrics(
        features,
        matrix_indices,
    )

    print("Projected dimensions:", features.shape[1])
    print("Anomalib time:", round(original_time, 3), "s")
    print("Matrix time:", round(matrix_time, 3), "s")
    print("Speedup:", round(original_time / matrix_time, 2), "x")

    print("\nSelection agreement:")
    print("Matching positions:", matching_positions, "/", CORESET_SIZE)
    print("Common indices:", common, "/", CORESET_SIZE)
    print("Jaccard:", round(jaccard, 6))

    print("\nGeometric coverage:")

    for metric in ["mean", "p95", "max"]:
        print(
            f"{metric}: "
            f"Anomalib={original_coverage[metric]:.6f} | "
            f"Matrix={matrix_coverage[metric]:.6f}"
        )

    return {
        "seed": seed,
        "original_time": original_time,
        "matrix_time": matrix_time,
        "speedup": original_time / matrix_time,
        "jaccard": jaccard,
        "matching_positions": matching_positions,
        "original_coverage": original_coverage,
        "matrix_coverage": matrix_coverage,
    }


def main():
    torch.set_num_threads(4)

    results = []

    for seed in SEEDS:
        results.append(run_seed(seed))

    print("\n" + "=" * 55)
    print("FINAL MULTI-SEED SUMMARY")
    print("=" * 55)

    print(
        "Average Anomalib time:",
        round(np.mean([r["original_time"] for r in results]), 3),
        "s",
    )

    print(
        "Average Matrix time:",
        round(np.mean([r["matrix_time"] for r in results]), 3),
        "s",
    )

    print(
        "Average speedup:",
        round(np.mean([r["speedup"] for r in results]), 2),
        "x",
    )

    print(
        "Average Jaccard:",
        round(np.mean([r["jaccard"] for r in results]), 6),
    )

    print(
        "Average matching positions:",
        round(np.mean([r["matching_positions"] for r in results]), 2),
        "/",
        CORESET_SIZE,
    )

    print("=" * 55)


if __name__ == "__main__":
    main()
