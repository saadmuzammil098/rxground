from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groundedness import answer_groundedness, claim_groundedness  # noqa: E402


def test_fully_supported_claim_scores_high():
    claim = "Lipitor is contraindicated in active liver disease."
    source = ["Lipitor is contraindicated in patients with active liver disease or unexplained transaminase elevations."]
    score = claim_groundedness(claim, source)
    assert score > 0.8


def test_fabricated_claim_scores_low():
    claim = "Lipitor causes permanent blindness in most patients within a week."
    source = ["Lipitor is contraindicated in patients with active liver disease."]
    score = claim_groundedness(claim, source)
    assert score < 0.3


def test_empty_claim_scores_zero():
    assert claim_groundedness("", ["some source text"]) == 0.0


def test_no_source_scores_zero():
    assert claim_groundedness("furosemide treats edema", []) == 0.0


def test_answer_groundedness_averages_claims():
    pairs = [
        ("furosemide treats edema", ["furosemide is indicated for the treatment of edema"]),
        ("furosemide cures cancer", ["furosemide is indicated for the treatment of edema"]),
    ]
    score = answer_groundedness(pairs)
    high = claim_groundedness(*pairs[0])
    low = claim_groundedness(*pairs[1])
    assert abs(score - (high + low) / 2) < 1e-9


def test_answer_groundedness_of_refusal_is_one():
    assert answer_groundedness([]) == 1.0
