from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citation_schema import Citation, Claim, GroundedResponse  # noqa: E402


def _citation(**overrides):
    defaults = {"set_id": "set-1", "section": "contraindications", "chunk_id": "set-1:contraindications:0"}
    defaults.update(overrides)
    return Citation(**defaults)


def test_valid_non_refused_response():
    resp = GroundedResponse(
        refused=False,
        claims=[Claim(text="Lipitor is contraindicated in active liver disease.", citations=[_citation()])],
    )
    assert resp.refused is False
    assert len(resp.claims) == 1


def test_valid_refused_response():
    resp = GroundedResponse(refused=True, refusal_reason="not covered by the indexed labels")
    assert resp.claims == []


def test_refused_response_with_claims_is_rejected():
    with pytest.raises(ValidationError):
        GroundedResponse(
            refused=True,
            refusal_reason="not covered",
            claims=[Claim(text="something", citations=[_citation()])],
        )


def test_non_refused_response_with_no_claims_is_rejected():
    with pytest.raises(ValidationError):
        GroundedResponse(refused=False, claims=[])


def test_refused_response_without_reason_is_rejected():
    with pytest.raises(ValidationError):
        GroundedResponse(refused=True, refusal_reason=None)


def test_claim_without_citation_is_rejected():
    with pytest.raises(ValidationError):
        Claim(text="unsupported claim", citations=[])


def test_citation_requires_non_empty_fields():
    with pytest.raises(ValidationError):
        Citation(set_id="", section="dosage_and_administration", chunk_id="x:y:0")
