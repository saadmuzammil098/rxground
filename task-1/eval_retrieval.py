"""Measures, with real numbers, whether section-aware chunking retrieves
better than naive fixed-size chunking on this document set.

12 hand-written pharmacist-style questions, each with a known-correct
(drug, section) answer taken directly from the real labels fetched by
fetch_labels.py. For each query, embeds it with the same
BAAI/bge-base-en-v1.5 model, runs a top-3 nearest-neighbor search against
both Chroma collections built by index.py, and scores whether the correct
section actually comes back, at rank 1 and within the top 3.

Usage:
    ../.venv/bin/python eval_retrieval.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from index import NAIVE_COLLECTION, SECTION_AWARE_COLLECTION, get_client, get_embedder

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class EvalQuery:
    query: str
    expected_set_id: str
    expected_section: str


QUERIES = [
    EvalQuery(
        "What is the dosage of the metformin combination ZITUVIMET for a patient with reduced kidney function",
        "0098dec4-f0e5-45d5-8aa4-5d0faf9ab142",
        "dosage_and_administration",
    ),
    EvalQuery(
        "What are the contraindications for Lipitor, atorvastatin",
        "a60cc18b-0631-4cf0-b021-9f52224ece65",
        "contraindications",
    ),
    EvalQuery(
        "What drug interactions does warfarin sodium have",
        "0cbce382-9c88-4f58-ae0f-532a841e8f95",
        "drug_interactions",
    ),
    EvalQuery(
        "Is lisinopril, Zestril, safe to take during pregnancy",
        "838c2d78-d2d8-4981-9ec9-e50ef9e1a5d8",
        "pregnancy",
    ),
    EvalQuery(
        "What happens in an overdose of amoxicillin, Amoxil",
        "ec7dd735-dc92-3d29-e053-2a95a90af549",
        "overdosage",
    ),
    EvalQuery(
        "What is the boxed warning for the sitagliptin and metformin combination ZITUVIMET",
        "0098dec4-f0e5-45d5-8aa4-5d0faf9ab142",
        "boxed_warning",
    ),
    EvalQuery(
        "What are the common adverse reactions and side effects of sertraline, Zoloft",
        "fda754f6-d0f3-4dce-a17a-927d64f912f7",
        "adverse_reactions",
    ),
    EvalQuery(
        "How should furosemide, Lasix, be dosed for a patient",
        "2c9b4d8f-0770-482d-a9e6-9c616a440b1a",
        "dosage_and_administration",
    ),
    EvalQuery(
        "Can children safely take gabapentin, Neurontin, what is the pediatric use guidance",
        "97935fd9-1d4a-43b6-a5d9-de994591187b",
        "pediatric_use",
    ),
    EvalQuery(
        "What warnings and precautions exist for simvastatin, ZOCOR",
        "8f55d5de-5a4f-4a39-8c84-c53976dd6af9",
        "warnings_and_cautions",
    ),
    EvalQuery(
        "What dosing considerations exist for elderly geriatric patients taking losartan, COZAAR",
        "9949448f-c3b9-44ee-94ed-c1aca8c90f39",
        "geriatric_use",
    ),
    EvalQuery(
        "What is albuterol, VENTOLIN HFA, indicated to treat",
        "2ed73618-be3a-4331-9509-6401258f791f",
        "indications_and_usage",
    ),
]


def _run_query(collection, embedder, query: str, top_k: int = 3):
    embedding = embedder.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    metadatas = results["metadatas"][0]
    return [(m["set_id"], m["section"]) for m in metadatas]


def evaluate(collection_name: str) -> dict:
    client = get_client()
    collection = client.get_collection(collection_name)
    embedder = get_embedder()

    rows = []
    top1_hits = 0
    top3_hits = 0
    for eval_query in QUERIES:
        top_results = _run_query(collection, embedder, eval_query.query, top_k=3)
        expected = (eval_query.expected_set_id, eval_query.expected_section)
        top1_hit = top_results[0] == expected if top_results else False
        top3_hit = expected in top_results
        top1_hits += int(top1_hit)
        top3_hits += int(top3_hit)
        rows.append(
            {
                "query": eval_query.query,
                "expected": {"set_id": eval_query.expected_set_id, "section": eval_query.expected_section},
                "top_3_returned": [{"set_id": s, "section": sec} for s, sec in top_results],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
            }
        )

    n = len(QUERIES)
    return {
        "collection": collection_name,
        "n_queries": n,
        "top1_accuracy": top1_hits / n,
        "top3_accuracy": top3_hits / n,
        "rows": rows,
    }


def main() -> None:
    section_aware_results = evaluate(SECTION_AWARE_COLLECTION)
    naive_results = evaluate(NAIVE_COLLECTION)

    print(f"section-aware: top1={section_aware_results['top1_accuracy']:.3f} top3={section_aware_results['top3_accuracy']:.3f}")
    print(f"naive fixed-size: top1={naive_results['top1_accuracy']:.3f} top3={naive_results['top3_accuracy']:.3f}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "eval_results.json").write_text(
        json.dumps({"section_aware": section_aware_results, "naive_fixed_size": naive_results}, indent=2)
    )
    print(f"\nwrote {OUTPUTS_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
