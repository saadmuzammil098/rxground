"""Deliberate failure exercise: run the same real questions through the
citation-enforced pipeline and through an unenforced pipeline (same
retrieval, same retrieved context, citation requirement removed from the
prompt entirely) and compare what actually comes back.

The question of interest is q18, a false-premise question asking about a
drug interaction that the Lipitor label does not describe. The
citation-enforced path can only answer with claims it can tie to a real
retrieved chunk_id, so a nonexistent interaction has nowhere to attach a
citation. The unenforced path has no such constraint.

Usage:
    ../.venv/bin/python failure_exercise.py [provider]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import paths  # noqa: F401
from generate import GENERATORS
from prompts import SYSTEM_PROMPT_ENFORCED, SYSTEM_PROMPT_UNENFORCED, build_user_prompt
from rag_grounded import answer_question
from retrieve_with_ids import retrieve

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

QUESTIONS = [
    "Does Lipitor (atorvastatin) have a documented interaction with levothyroxine that causes serotonin syndrome?",
    "What are the contraindications for Lipitor (atorvastatin)?",
    "What is the pediatric use guidance for Neurontin (gabapentin)?",
    "What is the recommended dose of aspirin for a patient with atrial fibrillation?",
]


def run(provider: str = "ollama") -> list[dict]:
    """Both prompt variants are run against the exact same retrieved
    context (retrieved once per question), the similarity gate that the
    real pipeline uses in answer_question() is deliberately bypassed here
    for both variants, otherwise a low-similarity question would be
    refused before either prompt variant ever got a chance to answer,
    which would tell us nothing about what the unenforced prompt does
    when it IS given a chance to fabricate.
    """
    results = []
    for question in QUESTIONS:
        chunks = retrieve(question, top_k=5)
        user_prompt = build_user_prompt(question, chunks)

        enforced_answer = answer_question(question, provider=provider, enforce_citations=True)
        unenforced_raw = GENERATORS[provider](SYSTEM_PROMPT_UNENFORCED, user_prompt)

        results.append(
            {
                "question": question,
                "best_similarity": round(chunks[0].similarity, 3) if chunks else 0.0,
                "enforced": {
                    "refused": enforced_answer.refused,
                    "refusal_reason": enforced_answer.refusal_reason,
                    "claims": [
                        {"text": c.text, "citations_valid": c.citations_valid, "cited_chunk_ids": c.cited_chunk_ids}
                        for c in enforced_answer.claims
                    ],
                    "raw_response": enforced_answer.raw_response,
                },
                "unenforced": {
                    "raw_response": unenforced_raw,
                },
            }
        )

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "failure_exercise.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print("=" * 80)
        print(r["question"])
        print("-- enforced --")
        print("refused:", r["enforced"]["refused"], r["enforced"]["refusal_reason"])
        for c in r["enforced"]["claims"]:
            print(" claim:", c["text"], "| valid citation:", c["citations_valid"])
        print("-- unenforced --")
        print(r["unenforced"]["raw_response"][:500])
    return results


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    run(provider)
