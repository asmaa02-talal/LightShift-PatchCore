
"""Fixed-budget KCenterGreedy with optimized matrix distances."""

from __future__ import annotations

import torch
from anomalib.models.components.sampling import KCenterGreedy
from anomalib.models import Patchcore


def select_fixed_budget_coreset(
    embeddings: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    """Select exactly target_size embeddings using matrix-based KCenterGreedy."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")

    if target_size <= 0:
        raise ValueError("target_size must be positive")

    candidate_count = embeddings.shape[0]

    if target_size > candidate_count:
        raise ValueError("Target budget exceeds candidate count")

    if target_size == candidate_count:
        return embeddings.clone()

    print(
        f"[CORESET] Candidates: {candidate_count:,} | "
        f"Target budget: {target_size:,}",
        flush=True,
    )

    # Same projection method as Anomalib.
    sampling_ratio = (target_size + 0.5) / candidate_count

    sampler = KCenterGreedy(
        embedding=embeddings,
        sampling_ratio=sampling_ratio,
    )

    print("[CORESET] Fitting random projection...", flush=True)

    sampler.model.fit(embeddings)
    features = sampler.model.transform(embeddings)

    print(
        f"[CORESET] Projected features: {tuple(features.shape)}",
        flush=True,
    )

    # Same random initialization principle as Anomalib.
    idx = torch.randint(
        high=candidate_count,
        size=(1,),
        device=features.device,
    ).item()

    feature_norms = torch.sum(features * features, dim=1)

    min_distances = torch.full(
        (candidate_count,),
        float("inf"),
        dtype=features.dtype,
        device=features.device,
    )

    selected_indices = [idx]

    print("[CORESET] Starting matrix-based selection...", flush=True)

    for iteration in range(target_size - 1):
        center = features[idx]
        center_norm = torch.dot(center, center)

        distances = (
            feature_norms
            + center_norm
            - 2.0 * torch.mv(features, center)
        )

        distances.clamp_(min=0.0)

        min_distances = torch.minimum(min_distances, distances)
        min_distances[idx] = 0.0

        idx = int(torch.argmax(min_distances).item())
        selected_indices.append(idx)

        if (iteration + 1) % 1000 == 0:
            print(
                f"[CORESET] Selected: {iteration + 1:,}/"
                f"{target_size - 1:,}",
                flush=True,
            )

    indices = torch.tensor(
        selected_indices,
        dtype=torch.long,
        device=embeddings.device,
    )

    memory_bank = embeddings[indices]

    print(
        f"[CORESET] Selection completed: {len(memory_bank):,} embeddings",
        flush=True,
    )

    return memory_bank


class FixedBudgetPatchcore(Patchcore):
    """PatchCore with an exact memory-bank budget."""

    def __init__(self, target_memory_bank_size: int, **kwargs):
        super().__init__(**kwargs)

        if target_memory_bank_size <= 0:
            raise ValueError("target_memory_bank_size must be positive")

        self.target_memory_bank_size = target_memory_bank_size

    def fit(self) -> None:
        """Build the memory bank with an exact number of embeddings."""

        embedding_store = self.model.embedding_store

        print(
            f"[B1] Embedding store contains {len(embedding_store)} batches.",
            flush=True,
        )

        if len(embedding_store) == 0:
            raise ValueError("Embedding store is empty")

        embeddings = torch.vstack(embedding_store)

        print(
            f"[B1] Total candidate embeddings: {embeddings.shape[0]:,}",
            flush=True,
        )

        memory_bank = select_fixed_budget_coreset(
            embeddings,
            self.target_memory_bank_size,
        )

        self.model.memory_bank = memory_bank
        embedding_store.clear()

        print(
            f"[B1] Memory bank created: {tuple(self.model.memory_bank.shape)}",
            flush=True,
        )

        print(
            f"[B1] Remaining embedding batches: {len(embedding_store)}",
            flush=True,
        )

        if self.model.memory_bank.shape[0] != self.target_memory_bank_size:
            raise RuntimeError("Memory bank size does not match target budget")

        print("[B1] Memory bank validation successful.", flush=True)
