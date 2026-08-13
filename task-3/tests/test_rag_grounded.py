from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402,F401
from citation_schema import Citation  # noqa: E402
from rag_grounded import _extract_json, _validate_claim_citations  # noqa: E402
from retrieve_with_ids import RetrievedChunk  # noqa: E402


def _chunk(**overrides):
    defaults = dict(
        chunk_id="set-1:contraindications:0",
        text="Lipitor is contraindicated in active liver disease.",
        brand_name="Lipitor",
        generic_name="atorvastatin",
        section="contraindications",
        set_id="set-1",
        similarity=0.9,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def test_extract_json_from_bare_object():
    raw = '{"refused": false, "claims": []}'
    assert _extract_json(raw) == raw


def test_extract_json_strips_markdown_fence_and_prose():
    raw = 'Sure, here is the answer:\n```json\n{"refused": true, "refusal_reason": "x"}\n```\nHope that helps!'
    assert _extract_json(raw) == '{"refused": true, "refusal_reason": "x"}'


def test_extract_json_with_no_object_raises():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


def test_valid_citation_matches_retrieved_chunk():
    retrieved_by_id = {"set-1:contraindications:0": _chunk()}
    citations = [Citation(set_id="set-1", section="contraindications", chunk_id="set-1:contraindications:0")]
    valid, reason, ids = _validate_claim_citations(citations, retrieved_by_id)
    assert valid is True
    assert reason is None
    assert ids == ["set-1:contraindications:0"]


def test_citation_to_nonexistent_chunk_id_is_invalid():
    retrieved_by_id = {"set-1:contraindications:0": _chunk()}
    citations = [Citation(set_id="set-1", section="contraindications", chunk_id="fabricated-chunk-id")]
    valid, reason, _ = _validate_claim_citations(citations, retrieved_by_id)
    assert valid is False
    assert "not among the retrieved chunks" in reason


def test_citation_with_mismatched_section_is_invalid():
    retrieved_by_id = {"set-1:contraindications:0": _chunk()}
    citations = [Citation(set_id="set-1", section="dosage_and_administration", chunk_id="set-1:contraindications:0")]
    valid, reason, _ = _validate_claim_citations(citations, retrieved_by_id)
    assert valid is False
    assert "does not match" in reason
