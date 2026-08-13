"""Citation-enforced generation on top of task-1's index and task-2's
provider-agnostic generators.

Pipeline for one question:
1. retrieve() the top-k chunks from task-1's section-aware Chroma index
   (reused unchanged, see retrieve_with_ids.py).
2. A similarity gate, same threshold and rationale as task-2's, skips
   generation entirely when nothing relevant was retrieved.
3. If the gate passes, ask the provider for a structured JSON answer
   (prompts.SYSTEM_PROMPT_ENFORCED), one claim per factual statement, each
   claim citing a (set_id, section, chunk_id) it came from.
4. Parse the JSON and validate it against schema.GroundedResponse. A
   response that is not valid JSON, or does not match the schema (a claim
   with no citation, for example), is NOT passed through as if it were a
   real answer, it fails closed to a refusal, this is the whole point of
   validating before returning anything to the caller.
5. Cross-check every citation's chunk_id against the chunk_ids actually
   retrieved in step 1. Pydantic can confirm a citation has the right
   shape, it cannot confirm the LLM did not invent a chunk_id that looks
   real, only this repo-level check can.
6. Score groundedness per claim with a lexical overlap heuristic against
   the text of its cited chunk(s) (groundedness.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import paths  # noqa: F401
from generate import GENERATORS, GenerationError
from groundedness import answer_groundedness
from prompts import NOT_COVERED_PHRASE, SYSTEM_PROMPT_ENFORCED, SYSTEM_PROMPT_UNENFORCED, build_user_prompt
from pydantic import ValidationError
from citation_schema import GroundedResponse
from retrieve_with_ids import RetrievedChunk, retrieve

NOT_COVERED_SIMILARITY_THRESHOLD = 0.75


@dataclass
class ClaimResult:
    text: str
    cited_chunk_ids: list[str]
    citations_valid: bool
    invalid_reason: str | None
    groundedness: float


@dataclass
class GroundedAnswer:
    question: str
    provider: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    best_similarity: float = 0.0
    passed_similarity_gate: bool = False
    refused: bool = False
    refusal_reason: str | None = None
    parse_error: str | None = None
    claims: list[ClaimResult] = field(default_factory=list)
    all_citations_valid: bool = True
    groundedness_score: float = 1.0
    raw_response: str = ""


def _extract_json(raw: str) -> str:
    """Model output sometimes wraps the JSON in markdown fences or a
    stray sentence. Take the substring between the first '{' and the
    last '}' rather than assuming the whole response is bare JSON.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return raw[start : end + 1]


def _validate_claim_citations(
    claim_citations: list, retrieved_by_id: dict[str, RetrievedChunk]
) -> tuple[bool, str | None, list[str]]:
    cited_chunk_ids = [c.chunk_id for c in claim_citations]
    for citation in claim_citations:
        chunk = retrieved_by_id.get(citation.chunk_id)
        if chunk is None:
            return False, f"chunk_id '{citation.chunk_id}' was not among the retrieved chunks", cited_chunk_ids
        if chunk.set_id != citation.set_id or chunk.section != citation.section:
            return (
                False,
                f"citation set_id/section does not match retrieved chunk '{citation.chunk_id}'",
                cited_chunk_ids,
            )
    return True, None, cited_chunk_ids


def answer_question(
    question: str,
    provider: str = "ollama",
    top_k: int = 5,
    threshold: float = NOT_COVERED_SIMILARITY_THRESHOLD,
    enforce_citations: bool = True,
) -> GroundedAnswer:
    chunks = retrieve(question, top_k=top_k)
    best_similarity = chunks[0].similarity if chunks else 0.0

    if best_similarity < threshold:
        return GroundedAnswer(
            question=question,
            provider="none (similarity gate)",
            retrieved=chunks,
            best_similarity=best_similarity,
            passed_similarity_gate=False,
            refused=True,
            refusal_reason=(
                f"closest indexed chunk scored {best_similarity:.3f} cosine similarity, below "
                f"the {threshold:.2f} confidence threshold, {NOT_COVERED_PHRASE}"
            ),
            groundedness_score=1.0,
        )

    system_prompt = SYSTEM_PROMPT_ENFORCED if enforce_citations else SYSTEM_PROMPT_UNENFORCED
    user_prompt = build_user_prompt(question, chunks)
    generator = GENERATORS[provider]
    try:
        raw = generator(system_prompt, user_prompt)
    except GenerationError as exc:
        return GroundedAnswer(
            question=question,
            provider=provider,
            retrieved=chunks,
            best_similarity=best_similarity,
            passed_similarity_gate=True,
            refused=True,
            refusal_reason=f"generation failed: {exc}",
            groundedness_score=1.0,
        )

    if not enforce_citations:
        # Unenforced path returns plain prose on purpose, for the failure
        # exercise, it is never schema-validated or citation-checked.
        return GroundedAnswer(
            question=question,
            provider=provider,
            retrieved=chunks,
            best_similarity=best_similarity,
            passed_similarity_gate=True,
            refused=False,
            raw_response=raw,
        )

    retrieved_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    try:
        parsed = GroundedResponse.model_validate(json.loads(_extract_json(raw)))
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        return GroundedAnswer(
            question=question,
            provider=provider,
            retrieved=chunks,
            best_similarity=best_similarity,
            passed_similarity_gate=True,
            refused=True,
            refusal_reason="model output failed schema validation, failing closed rather than returning an ungrounded answer",
            parse_error=str(exc),
            all_citations_valid=False,
            groundedness_score=0.0,
            raw_response=raw,
        )

    if parsed.refused:
        return GroundedAnswer(
            question=question,
            provider=provider,
            retrieved=chunks,
            best_similarity=best_similarity,
            passed_similarity_gate=True,
            refused=True,
            refusal_reason=parsed.refusal_reason,
            raw_response=raw,
        )

    claim_results: list[ClaimResult] = []
    all_valid = True
    for claim in parsed.claims:
        valid, reason, cited_ids = _validate_claim_citations(claim.citations, retrieved_by_id)
        cited_texts = [retrieved_by_id[cid].text for cid in cited_ids if cid in retrieved_by_id]
        score = answer_groundedness([(claim.text, cited_texts)]) if valid else 0.0
        if not valid:
            all_valid = False
        claim_results.append(
            ClaimResult(
                text=claim.text,
                cited_chunk_ids=cited_ids,
                citations_valid=valid,
                invalid_reason=reason,
                groundedness=score,
            )
        )

    overall_groundedness = (
        sum(c.groundedness for c in claim_results) / len(claim_results) if claim_results else 1.0
    )

    return GroundedAnswer(
        question=question,
        provider=provider,
        retrieved=chunks,
        best_similarity=best_similarity,
        passed_similarity_gate=True,
        refused=False,
        claims=claim_results,
        all_citations_valid=all_valid,
        groundedness_score=overall_groundedness,
        raw_response=raw,
    )
