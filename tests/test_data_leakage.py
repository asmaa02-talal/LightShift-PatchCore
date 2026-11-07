import pytest

from lightshift.data.dataset import validate_split_ids


def test_split_validation_accepts_disjoint_sets():
    validate_split_ids({"a"}, {"b"}, {"c"}, {"d"})


def test_split_validation_rejects_overlap():
    with pytest.raises(ValueError, match="Fuite de données"):
        validate_split_ids({"same"}, {"same"}, set(), set())
