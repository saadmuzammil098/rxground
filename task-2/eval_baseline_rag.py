"""Runs real pharmacist-style questions through the baseline RAG pipeline
and records what actually happened, citations included or not, refused or
answered, against Ollama live. This is the evidence behind the README's
"answers cite the actual label text, refuses plainly when a drug is not
indexed" claim, not a predicted outcome.

Usage:
    ../.venv/bin/python eval_baseline_rag.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag import answer_question

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class EvalCase:
    question: str
    expect_covered: bool
    expected_brand_substring: str | None = None


CASES = [
    EvalCase(
        "What is the maximum dose of the metformin combination ZITUVIMET for a patient with renal impairment",
        expect_covered=True,
        expected_brand_substring="ZITUVIMET",
    ),
    EvalCase(
        "What are the contraindications for Lipitor, atorvastatin",
        expect_covered=True,
        expected_brand_substring="Lipitor",
    ),
    EvalCase(
        "What drug interactions does warfarin sodium have",
        expect_covered=True,
        expected_brand_substring="Warfarin",
    ),
    EvalCase(
        "Is lisinopril, Zestril, safe to take during pregnancy",
        expect_covered=True,
        expected_brand_substring="Zestril",
    ),
    EvalCase(
        "What happens in an overdose of amoxicillin, Amoxil",
        expect_covered=True,
        expected_brand_substring="Amoxil",
    ),
    EvalCase(
        "What is the boxed warning for the metformin combination ZITUVIMET",
        expect_covered=True,
        expected_brand_substring="ZITUVIMET",
    ),
    EvalCase(
        "What warnings and precautions exist for simvastatin, ZOCOR",
        expect_covered=True,
        expected_brand_substring="ZOCOR",
    ),
    EvalCase(
        "What is albuterol, VENTOLIN HFA, indicated to treat",
        expect_covered=True,
        expected_brand_substring="VENTOLIN",
    ),
    EvalCase(
        "What is the recommended dose of aspirin for a heart attack",
        expect_covered=False,
    ),
    EvalCase(
        "Is prednisone safe for a diabetic patient",
        expect_covered=False,
    ),
]


def run(provider: str = "ollama") -> list[dict]:
    rows = []
    for case in CASES:
        result = answer_question(case.question, provider=provider)
        has_citation = len(result.citations) > 0
        citation_matches_expected = (
            any(case.expected_brand_substring.lower() in brand.lower() for brand, _ in result.citations)
            if case.expected_brand_substring
            else None
        )
        row = {
            "question": case.question,
            "expect_covered": case.expect_covered,
            "actual_passed_similarity_gate": result.passed_similarity_gate,
            "actual_stated_not_covered": result.stated_not_covered,
            "best_similarity": round(result.best_similarity, 4),
            "answer": result.answer,
            "citations": result.citations,
            "has_citation": has_citation,
            "citation_matches_expected_drug": citation_matches_expected,
            "source_style_citations": result.source_style_citations,
        }
        rows.append(row)
        print(f"Q: {case.question[:70]}")
        print(f"   covered={result.passed_similarity_gate} similarity={result.best_similarity:.3f} citations={result.citations}")
        if result.source_style_citations:
            print(f"   source-style citations: {result.source_style_citations}")
        print(f"   A: {result.answer[:200]}")
        print()
    return rows


def main() -> None:
    rows = run()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "eval_results.json").write_text(json.dumps(rows, indent=2))

    n = len(rows)
    gate_correct = sum(1 for r in rows if r["actual_passed_similarity_gate"] == r["expect_covered"])
    covered_rows = [r for r in rows if r["expect_covered"]]
    cited_correctly = sum(1 for r in covered_rows if r["citation_matches_expected_drug"])
    any_citation_style = sum(1 for r in covered_rows if r["citation_matches_expected_drug"] or r["source_style_citations"])
    not_covered_rows = [r for r in rows if not r["expect_covered"]]
    correctly_refused = sum(1 for r in not_covered_rows if r["actual_stated_not_covered"] or not r["actual_passed_similarity_gate"])

    print("=== Summary ===")
    print(f"similarity gate matched expectation: {gate_correct}/{n}")
    print(f"in-index questions with the exact requested (Brand, section) citation format: {cited_correctly}/{len(covered_rows)}")
    print(f"in-index questions with any traceable citation (requested format or the label's own [see ...] style): {any_citation_style}/{len(covered_rows)}")
    print(f"out-of-index questions correctly refused: {correctly_refused}/{len(not_covered_rows)}")
    print(f"\nwrote {OUTPUTS_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
