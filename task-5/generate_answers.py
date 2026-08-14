"""Runs the current best retrieval (task-4's hybrid search + rerank) plus
a plain-text generation prompt over the golden set, and writes the raw
(question, answer, contexts, reference) rows RAGAS needs to
outputs/raw_answers.json.

Deliberately separate from run_ragas_eval.py: generating the answers
requires Ollama (or Groq/Gemini) and the local index, scoring them with
RAGAS requires a whole separate, heavy LLM-judge dependency chain
(langchain, datasets). Splitting them means a RAGAS re-score against
already-generated answers does not need to re-run generation.

Usage:
    ../.venv/bin/python generate_answers.py [provider]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import paths  # noqa: F401
from advanced_retrieve import retrieve
from generate import GENERATORS

DATA_PATH = Path(__file__).resolve().parent / "data" / "golden_set.json"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

SYSTEM_PROMPT = (
    "You are RxGround, a clinical drug-reference assistant for a pharmacist. Answer the "
    "question using only the label excerpts given below, in your own words, plain prose, no "
    "citation formatting needed. If the excerpts do not answer the question, say so plainly."
)


def _build_user_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts) if contexts else "(no label excerpts retrieved)"
    return f"Label excerpts:\n{context_block}\n\nQuestion: {question}"


def run(provider: str = "ollama") -> list[dict]:
    golden_set = json.loads(DATA_PATH.read_text())
    generator = GENERATORS[provider]

    rows = []
    for item in golden_set:
        result = retrieve(item["question"], top_k=5)
        contexts = [c.text for c in result.chunks]
        answer = generator(SYSTEM_PROMPT, _build_user_prompt(item["question"], contexts))
        rows.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "reference": item["reference_answer"],
                "expected_set_id": item["expected_set_id"],
                "expected_section": item["expected_section"],
                "retrieved_set_ids": [c.set_id for c in result.chunks],
                "retrieved_sections": [c.section for c in result.chunks],
            }
        )
        print(f"answered: {item['question'][:70]}...")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "raw_answers.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUTPUTS_DIR / 'raw_answers.json'}")
    return rows


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    run(provider)
