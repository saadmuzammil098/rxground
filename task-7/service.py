"""Minimal FastAPI wrapper around task-3's citation-enforced RAG pipeline.

Not part of RxGround's original 6-task roadmap (a bonus task, same
shape as FleetPulse's task-11), added so CareThread's capstone (Phase 7)
can make a genuine live HTTP call into RxGround for drug-interaction
checks instead of CareThread's own task-1 static stub table, the
cross-project integration the roadmap's CareThread Task 3 asks for.

Deliberately thin: this file adds zero new retrieval or generation
logic, it exposes `task-3/rag_grounded.py::answer_question` (already
citation-enforced, already refusal-capable when nothing relevant is
indexed) over HTTP, unmodified. `sys.path` is extended by hand here
rather than via a shared `paths.py`, naming this file's own helper
`paths` would collide with `task-3/paths.py` the instant both get
imported in the same process, the identical `src`-package collision
FleetPulse's and CareThread's READMEs already document, just with a
different module name.

Run locally (no Terraform/Lambda deployment, this is a same-machine
service call, not a hosted one, see task-7/README.md for why):
    uvicorn task-7.service:app --port 8100
"""
from __future__ import annotations

import sys
from pathlib import Path

_TASK3_DIR = Path(__file__).resolve().parent.parent / "task-3"
if str(_TASK3_DIR) not in sys.path:
    sys.path.append(str(_TASK3_DIR))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from rag_grounded import answer_question  # noqa: E402

app = FastAPI(title="RxGround Query Service", version="0.1.0")


class InteractionRequest(BaseModel):
    drug_a: str
    drug_b: str
    provider: str = "ollama"


class Claim(BaseModel):
    text: str
    cited_chunk_ids: list[str]
    groundedness: float


class InteractionResponse(BaseModel):
    question: str
    refused: bool
    refusal_reason: str | None = None
    claims: list[Claim]
    groundedness_score: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/check_interaction", response_model=InteractionResponse)
def check_interaction(request: InteractionRequest) -> InteractionResponse:
    question = (
        f"Is there a known interaction between {request.drug_a} and "
        f"{request.drug_b}? If so, what is the clinical significance?"
    )
    result = answer_question(question, provider=request.provider)

    return InteractionResponse(
        question=question,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        claims=[
            Claim(
                text=c.text,
                cited_chunk_ids=c.cited_chunk_ids,
                groundedness=c.groundedness,
            )
            for c in result.claims
        ],
        groundedness_score=result.groundedness_score,
    )
