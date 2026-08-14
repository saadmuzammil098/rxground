"""Scores outputs/raw_answers.json (written by generate_answers.py) with
RAGAS: faithfulness, answer relevancy, context precision, context recall.

Uses Ollama (qwen2.5:7b) as the RAGAS judge LLM and BAAI/bge-base-en-v1.5
(the same embedding model task-1 already uses for the index) as the
RAGAS embeddings, via ragas's LangChain wrappers, consistent with this
project's provider-agnostic, no-OpenAI-required approach everywhere else.

Usage:
    ../.venv/bin/python run_ragas_eval.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

# ragas 0.3.9 (and every 0.4.x release as of this writing) unconditionally
# imports ChatVertexAI from langchain_community.chat_models.vertexai at
# import time. langchain-community has since removed that submodule as
# part of its own "sunset" migration to standalone integration packages
# (see the deprecation warning it prints on import), so this import fails
# on any currently-installable langchain-community version, independent
# of which ragas version is pinned, a real upstream incompatibility, not
# a mistake in this project's own pinning. This project never uses Vertex
# AI (Ollama, Groq, and Gemini are the three providers everywhere in this
# repo), so a stub module satisfying the import is a safe, minimal
# workaround, not a functional stand-in for real Vertex AI support.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - unused stub, see comment above
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Vertex AI is not used anywhere in this project, this is an import-only stub")

    _vertexai_stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

RAW_ANSWERS_PATH = Path(__file__).resolve().parent / "outputs" / "raw_answers.json"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

JUDGE_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def build_dataset() -> Dataset:
    rows = json.loads(RAW_ANSWERS_PATH.read_text())
    return Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["reference"],
            }
            for r in rows
        ]
    )


def run() -> dict:
    dataset = build_dataset()

    judge_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))

    # ragas's default RunConfig fires up to 16 judge calls concurrently
    # (max_workers=16). A single local Ollama server processes requests
    # serially, not in parallel, so most of those 16 just queue behind
    # each other, several ended up waiting past the default 180s timeout
    # before Ollama even started them, and were silently scored as
    # None/NaN rather than as a real faithfulness or precision value (see
    # this task's README for the real numbers that produced, most rows
    # came back None on the first run). max_workers=2 and a longer
    # timeout let every judge call actually finish instead of timing out
    # in a queue.
    run_config = RunConfig(timeout=900, max_workers=2)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    result_df = result.to_pandas()
    per_row = json.loads(result_df.to_json(orient="records"))

    summary = {
        "n_queries": len(per_row),
        "mean_faithfulness": float(result_df["faithfulness"].mean()),
        "mean_answer_relevancy": float(result_df["answer_relevancy"].mean()),
        "mean_context_precision": float(result_df["context_precision"].mean()),
        "mean_context_recall": float(result_df["context_recall"].mean()),
    }

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "ragas_results.json").write_text(
        json.dumps({"summary": summary, "rows": per_row}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
