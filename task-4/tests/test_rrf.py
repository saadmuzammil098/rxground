from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_retrieve import reciprocal_rank_fusion  # noqa: E402


def test_top_ranked_in_both_lists_scores_highest():
    dense = ["a", "b", "c"]
    bm25 = ["a", "c", "b"]
    scores = reciprocal_rank_fusion(dense, bm25, k=60)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_id_only_in_one_list_still_scored():
    dense = ["a", "b"]
    bm25 = ["c"]
    scores = reciprocal_rank_fusion(dense, bm25, k=60)
    assert set(scores) == {"a", "b", "c"}
    assert scores["c"] == 1 / 61


def test_id_in_both_lists_beats_id_in_one_list_even_at_worse_rank():
    # "a" is rank 5 in both lists, "b" is rank 1 in one list and absent
    # from the other, appearing in both lists should still win here
    # because RRF sums contributions across every list an id appears in.
    dense = ["x1", "x2", "x3", "x4", "a"]
    bm25 = ["x5", "x6", "x7", "x8", "a"]
    scores_a = reciprocal_rank_fusion(dense, bm25, k=60)["a"]

    dense_b = ["b", "x2", "x3", "x4", "x9"]
    bm25_b = []
    scores_b = reciprocal_rank_fusion(dense_b, bm25_b, k=60)["b"]

    assert scores_a > scores_b


def test_empty_lists_produce_empty_scores():
    assert reciprocal_rank_fusion([], []) == {}
