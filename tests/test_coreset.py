import numpy as np
import pytest
import torch

from lightshift.models.coreset import select_fixed_budget_coreset


def test_selects_exact_target_size():
    embeddings = torch.randn(100, 16)

    torch.manual_seed(42)
    np.random.seed(42)

    coreset = select_fixed_budget_coreset(embeddings, 10)

    assert coreset.shape == (10, 16)


def test_full_budget_returns_copy():
    embeddings = torch.randn(20, 8)

    coreset = select_fixed_budget_coreset(embeddings, 20)

    assert coreset.shape == embeddings.shape
    assert torch.equal(coreset, embeddings)
    assert coreset.data_ptr() != embeddings.data_ptr()


def test_rejects_non_2d_embeddings():
    with pytest.raises(ValueError):
        select_fixed_budget_coreset(torch.randn(10), 5)


@pytest.mark.parametrize("target_size", [0, -1])
def test_rejects_invalid_target_size(target_size):
    with pytest.raises(ValueError):
        select_fixed_budget_coreset(torch.randn(20, 8), target_size)


def test_rejects_budget_larger_than_candidates():
    with pytest.raises(ValueError):
        select_fixed_budget_coreset(torch.randn(10, 8), 11)


def test_selection_is_reproducible_with_fixed_seeds():
    embeddings = torch.randn(100, 16)

    torch.manual_seed(42)
    np.random.seed(42)
    first = select_fixed_budget_coreset(embeddings, 10)

    torch.manual_seed(42)
    np.random.seed(42)
    second = select_fixed_budget_coreset(embeddings, 10)

    assert torch.equal(first, second)
