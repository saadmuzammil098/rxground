"""Measures, with real numbers, whether hybrid search plus reranking beats
naive dense-only vector search on this document set, the roadmap's
explicit done-when for this task.

18 queries: the same 12 pharmacist-style questions task-1 used to compare
chunking strategies (general retrieval quality), plus 6 new
exact-dosage-sensitive questions designed to stress the specific weakness
hybrid search exists to fix, every drug's dosage_and_administration
section reads in almost the same shape ("recommended starting dose is
X mg once daily"), so a purely semantic search can retrieve the wrong
drug's dosing section for a query that names one specific drug.

Usage:
    ../.venv/bin/python eval_retrieval_comparison.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import paths  # noqa: F401
from advanced_retrieve import retrieve
from hybrid_retrieve import _dense_ranked_ids, _chunk_by_id  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class EvalQuery:
    query: str
    expected_set_id: str
    expected_section: str
    category: str


# The original 12 from task-1's eval_retrieval.py, general retrieval quality.
GENERAL_QUERIES = [
    EvalQuery(
        "What is the dosage of the metformin combination ZITUVIMET for a patient with reduced kidney function",
        "0098dec4-f0e5-45d5-8aa4-5d0faf9ab142", "dosage_and_administration", "general",
    ),
    EvalQuery(
        "What are the contraindications for Lipitor, atorvastatin",
        "a60cc18b-0631-4cf0-b021-9f52224ece65", "contraindications", "general",
    ),
    EvalQuery(
        "What drug interactions does warfarin sodium have",
        "0cbce382-9c88-4f58-ae0f-532a841e8f95", "drug_interactions", "general",
    ),
    EvalQuery(
        "Is lisinopril, Zestril, safe to take during pregnancy",
        "838c2d78-d2d8-4981-9ec9-e50ef9e1a5d8", "pregnancy", "general",
    ),
    EvalQuery(
        "What happens in an overdose of amoxicillin, Amoxil",
        "ec7dd735-dc92-3d29-e053-2a95a90af549", "overdosage", "general",
    ),
    EvalQuery(
        "What is the boxed warning for the sitagliptin and metformin combination ZITUVIMET",
        "0098dec4-f0e5-45d5-8aa4-5d0faf9ab142", "boxed_warning", "general",
    ),
    EvalQuery(
        "What are the common adverse reactions and side effects of sertraline, Zoloft",
        "fda754f6-d0f3-4dce-a17a-927d64f912f7", "adverse_reactions", "general",
    ),
    EvalQuery(
        "How should furosemide, Lasix, be dosed for a patient",
        "2c9b4d8f-0770-482d-a9e6-9c616a440b1a", "dosage_and_administration", "general",
    ),
    EvalQuery(
        "Can children safely take gabapentin, Neurontin, what is the pediatric use guidance",
        "97935fd9-1d4a-43b6-a5d9-de994591187b", "pediatric_use", "general",
    ),
    EvalQuery(
        "What warnings and precautions exist for simvastatin, ZOCOR",
        "8f55d5de-5a4f-4a39-8c84-c53976dd6af9", "warnings_and_cautions", "general",
    ),
    EvalQuery(
        "What dosing considerations exist for elderly geriatric patients taking losartan, COZAAR",
        "9949448f-c3b9-44ee-94ed-c1aca8c90f39", "geriatric_use", "general",
    ),
    EvalQuery(
        "What is albuterol, VENTOLIN HFA, indicated to treat",
        "2ed73618-be3a-4331-9509-6401258f791f", "indications_and_usage", "general",
    ),
]

# New, deliberately exact-dosage-sensitive: every one of these is a
# "recommended starting dose is X mg once daily"-shaped question against a
# named drug, exactly the pattern where dense-only search can confuse one
# drug's dosing section for another's.
EXACT_DOSAGE_QUERIES = [
    EvalQuery(
        "Norvasc amlodipine 5 mg once daily starting dose",
        "7367289c-b0b0-466a-83e2-558e2985c29f", "dosage_and_administration", "exact_dosage",
    ),
    EvalQuery(
        "ZOCOR simvastatin 80 mg maximum daily dose",
        "8f55d5de-5a4f-4a39-8c84-c53976dd6af9", "dosage_and_administration", "exact_dosage",
    ),
    EvalQuery(
        "Lipitor atorvastatin 10 mg starting dose",
        "a60cc18b-0631-4cf0-b021-9f52224ece65", "dosage_and_administration", "exact_dosage",
    ),
    EvalQuery(
        "COZAAR losartan potassium 50 mg once daily",
        "9949448f-c3b9-44ee-94ed-c1aca8c90f39", "dosage_and_administration", "exact_dosage",
    ),
    EvalQuery(
        "Lasix furosemide 40 mg intravenous dose",
        "2c9b4d8f-0770-482d-a9e6-9c616a440b1a", "dosage_and_administration", "exact_dosage",
    ),
    EvalQuery(
        "Zestril lisinopril 10 mg initial dose for hypertension",
        "838c2d78-d2d8-4981-9ec9-e50ef9e1a5d8", "dosage_and_administration", "exact_dosage",
    ),
]

QUERIES = GENERAL_QUERIES + EXACT_DOSAGE_QUERIES


def _naive_dense_top3(query: str) -> list[tuple[str, str]]:
    ids = _dense_ranked_ids(query, top_k=3)
    return [(_chunk_by_id(cid).set_id, _chunk_by_id(cid).section.value) for cid in ids]


def _hybrid_rerank_top3(query: str) -> list[tuple[str, str]]:
    result = retrieve(query, top_k=3, candidate_pool=20)
    return [(c.set_id, c.section) for c in result.chunks]


def evaluate(search_fn, label: str) -> dict:
    rows = []
    top1_hits = 0
    top3_hits = 0
    category_hits: dict[str, list[int]] = {"general": [0, 0], "exact_dosage": [0, 0]}

    for q in QUERIES:
        top3 = search_fn(q.query)
        expected = (q.expected_set_id, q.expected_section)
        top1_hit = top3[0] == expected if top3 else False
        top3_hit = expected in top3
        top1_hits += int(top1_hit)
        top3_hits += int(top3_hit)
        category_hits[q.category][0] += int(top1_hit)
        category_hits[q.category][1] += 1
        rows.append(
            {
                "query": q.query,
                "category": q.category,
                "expected": {"set_id": q.expected_set_id, "section": q.expected_section},
                "top_3_returned": [{"set_id": s, "section": sec} for s, sec in top3],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
            }
        )

    n = len(QUERIES)
    return {
        "label": label,
        "n_queries": n,
        "top1_accuracy": top1_hits / n,
        "top3_accuracy": top3_hits / n,
        "exact_dosage_top1_accuracy": category_hits["exact_dosage"][0] / category_hits["exact_dosage"][1],
        "general_top1_accuracy": category_hits["general"][0] / category_hits["general"][1],
        "rows": rows,
    }


def main() -> None:
    naive = evaluate(_naive_dense_top3, "naive_dense_only")
    hybrid = evaluate(_hybrid_rerank_top3, "hybrid_bm25_dense_plus_rerank")

    print(f"naive dense-only:        top1={naive['top1_accuracy']:.3f} top3={naive['top3_accuracy']:.3f} "
          f"exact_dosage_top1={naive['exact_dosage_top1_accuracy']:.3f}")
    print(f"hybrid + rerank:         top1={hybrid['top1_accuracy']:.3f} top3={hybrid['top3_accuracy']:.3f} "
          f"exact_dosage_top1={hybrid['exact_dosage_top1_accuracy']:.3f}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "eval_results.json").write_text(
        json.dumps({"naive_dense_only": naive, "hybrid_bm25_dense_plus_rerank": hybrid}, indent=2)
    )
    print(f"\nwrote {OUTPUTS_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
