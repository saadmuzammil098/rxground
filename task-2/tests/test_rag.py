import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag
from retrieve import RetrievedChunk


def test_extract_citations_matches_requested_format():
    text = "Warfarin interacts with many drugs. (Warfarin Sodium, drug_interactions)"
    citations = rag.extract_citations(text)
    assert citations == [("Warfarin Sodium", "drug_interactions")]


def test_extract_citations_ignores_source_style_brackets():
    text = "Contraindicated in liver failure [see Contraindications (4)]."
    assert rag.extract_citations(text) == []


def test_extract_source_style_citations():
    text = "Risk factors include age [see Warnings and Precautions (5.1)] and renal impairment."
    found = rag.extract_source_style_citations(text)
    assert found == ["[see Warnings and Precautions (5.1)]"]


def _fake_chunk(similarity: float, brand: str = "TestDrug") -> RetrievedChunk:
    return RetrievedChunk(
        text="some label text",
        brand_name=brand,
        generic_name="testinib",
        section="dosage_and_administration",
        set_id="fake-set-id",
        similarity=similarity,
    )


def test_below_threshold_skips_generation_entirely(monkeypatch):
    calls = []

    def fake_retrieve(question, top_k=5):
        return [_fake_chunk(0.5)]

    def fake_generator(system_prompt, user_prompt):
        calls.append(1)
        return "should not be called"

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "GENERATORS", {"fake": fake_generator})

    result = rag.answer_question("some question", provider="fake", threshold=0.75)

    assert result.passed_similarity_gate is False
    assert result.stated_not_covered is True
    assert "not covered" in result.answer.lower()
    assert calls == []  # the generator must never be invoked below the gate


def test_above_threshold_calls_generator_and_parses_citation(monkeypatch):
    def fake_retrieve(question, top_k=5):
        return [_fake_chunk(0.9)]

    def fake_generator(system_prompt, user_prompt):
        return "Take one tablet daily. (TestDrug, dosage_and_administration)"

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "GENERATORS", {"fake": fake_generator})

    result = rag.answer_question("some question", provider="fake", threshold=0.75)

    assert result.passed_similarity_gate is True
    assert result.stated_not_covered is False
    assert result.citations == [("TestDrug", "dosage_and_administration")]


def test_generation_error_is_surfaced_not_raised(monkeypatch):
    def fake_retrieve(question, top_k=5):
        return [_fake_chunk(0.9)]

    def failing_generator(system_prompt, user_prompt):
        raise rag.GenerationError("provider unreachable")

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "GENERATORS", {"fake": failing_generator})

    result = rag.answer_question("some question", provider="fake", threshold=0.75)

    assert "generation failed" in result.answer
    assert result.passed_similarity_gate is True
