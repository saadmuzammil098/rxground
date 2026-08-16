from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import service  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from rag_grounded import ClaimResult, GroundedAnswer  # noqa: E402

client = TestClient(service.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_check_interaction_refused(monkeypatch):
    monkeypatch.setattr(
        service,
        "answer_question",
        lambda question, provider="ollama": GroundedAnswer(
            question=question,
            provider="none (similarity gate)",
            refused=True,
            refusal_reason="not covered by the indexed labels",
        ),
    )
    resp = client.post(
        "/check_interaction", json={"drug_a": "foo", "drug_b": "bar"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["claims"] == []


def test_check_interaction_grounded_answer(monkeypatch):
    monkeypatch.setattr(
        service,
        "answer_question",
        lambda question, provider="ollama": GroundedAnswer(
            question=question,
            provider="ollama",
            refused=False,
            claims=[
                ClaimResult(
                    text="Warfarin and Zocor together can increase INR.",
                    cited_chunk_ids=["set-1:drug_interactions:5"],
                    citations_valid=True,
                    invalid_reason=None,
                    groundedness=0.8,
                )
            ],
            groundedness_score=0.8,
        ),
    )
    resp = client.post(
        "/check_interaction",
        json={"drug_a": "Warfarin Sodium", "drug_b": "ZOCOR"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert len(body["claims"]) == 1
    assert body["claims"][0]["cited_chunk_ids"] == ["set-1:drug_interactions:5"]
    assert body["groundedness_score"] == 0.8


def test_check_interaction_builds_the_expected_question(monkeypatch):
    captured = {}

    def fake_answer_question(question, provider="ollama"):
        captured["question"] = question
        captured["provider"] = provider
        return GroundedAnswer(question=question, provider=provider, refused=True, refusal_reason="x")

    monkeypatch.setattr(service, "answer_question", fake_answer_question)
    client.post(
        "/check_interaction",
        json={"drug_a": "aspirin", "drug_b": "warfarin", "provider": "gemini"},
    )
    assert "aspirin" in captured["question"]
    assert "warfarin" in captured["question"]
    assert captured["provider"] == "gemini"
