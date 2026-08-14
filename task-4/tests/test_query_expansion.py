from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from query_expansion import expand_query  # noqa: E402


def test_brand_name_gets_generic_added():
    expanded = expand_query("What are the contraindications for Lipitor?")
    assert "atorvastatin" in expanded.lower()


def test_generic_name_gets_brand_added():
    expanded = expand_query("maximum dose of atorvastatin")
    assert "Lipitor" in expanded


def test_query_with_both_names_is_unchanged():
    query = "Lipitor atorvastatin contraindications"
    assert expand_query(query) == query


def test_query_with_no_known_drug_name_is_unchanged():
    query = "what is the weather today"
    assert expand_query(query) == query


def test_multi_word_brand_name_matches():
    expanded = expand_query("Advil Dual Action with Acetaminophen dosing")
    assert "ibuprofen" in expanded.lower()
