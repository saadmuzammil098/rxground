import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from retrieve import _l2_squared_to_cosine_similarity


def test_zero_distance_is_identical():
    assert _l2_squared_to_cosine_similarity(0.0) == pytest.approx(1.0)


def test_max_l2_distance_for_unit_vectors_is_opposite():
    # for unit vectors, l2_squared maxes out at 4 (vectors pointing opposite ways)
    assert _l2_squared_to_cosine_similarity(4.0) == pytest.approx(-1.0)


def test_orthogonal_vectors():
    # for unit vectors, l2_squared = 2 means cosine similarity 0
    assert _l2_squared_to_cosine_similarity(2.0) == pytest.approx(0.0)
