"""Runs the full citation-enforced RAG pipeline (rag_grounded.answer_question)
against every query in data/eval_queries.json and writes a real, measured
results table to outputs/eval_results.json and outputs/eval_results.md.

Usage:
    ../.venv/bin/python run_eval.py [provider]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import paths  # noqa: F401
from rag_grounded import answer_question

DATA_PATH = Path(__file__).resolve().parent / "data" / "eval_queries.json"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def _claim_citation_validity(claims: list) -> float:
    if not claims:
        return 1.0
    valid = sum(1 for c in claims if c.citations_valid)
    return valid / len(claims)


def run(provider: str = "ollama") -> dict:
    queries = json.loads(DATA_PATH.read_text())
    rows = []

    for q in queries:
        result = answer_question(q["question"], provider=provider)
        citation_validity = _claim_citation_validity(result.claims)

        refusal_correct = None
        if q["expected_refusal"] is not None:
            refusal_correct = result.refused == q["expected_refusal"]

        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected_refusal": q["expected_refusal"],
                "actual_refused": result.refused,
                "refusal_correct": refusal_correct,
                "refusal_reason": result.refusal_reason,
                "parse_error": result.parse_error,
                "num_claims": len(result.claims),
                "all_citations_valid": result.all_citations_valid,
                "citation_validity": round(citation_validity, 3),
                "groundedness_score": round(result.groundedness_score, 3),
                "best_similarity": round(result.best_similarity, 3),
            }
        )

    scored_refusals = [r for r in rows if r["refusal_correct"] is not None]
    non_refused = [r for r in rows if not r["actual_refused"]]

    summary = {
        "provider": provider,
        "num_queries": len(rows),
        "refusal_correctness_rate": (
            round(sum(1 for r in scored_refusals if r["refusal_correct"]) / len(scored_refusals), 3)
            if scored_refusals
            else None
        ),
        "mean_citation_validity_on_answered": (
            round(sum(r["citation_validity"] for r in non_refused) / len(non_refused), 3)
            if non_refused
            else None
        ),
        "mean_groundedness_score": round(sum(r["groundedness_score"] for r in rows) / len(rows), 3),
        "parse_failures": sum(1 for r in rows if r["parse_error"]),
    }

    OUTPUTS_DIR.mkdir(exist_ok=True)
    output = {"summary": summary, "rows": rows}
    (OUTPUTS_DIR / "eval_results.json").write_text(json.dumps(output, indent=2))

    md_lines = [
        "| id | category | expected refusal | actual refused | refusal correct | # claims | citations valid | groundedness | best sim |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['id']} | {r['category']} | {r['expected_refusal']} | {r['actual_refused']} | "
            f"{r['refusal_correct']} | {r['num_claims']} | {r['citation_validity']} | "
            f"{r['groundedness_score']} | {r['best_similarity']} |"
        )
    (OUTPUTS_DIR / "eval_results.md").write_text("\n".join(md_lines) + "\n")

    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    run(provider)
