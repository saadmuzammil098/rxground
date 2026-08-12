"""Orchestrates retrieve, augment, and generate into one answer.

Two independent layers decide whether a question is "not covered":

1. A similarity gate, before any LLM call. If the best retrieved chunk
   scores below NOT_COVERED_SIMILARITY_THRESHOLD, generation is skipped
   entirely and a deterministic refusal is returned, this is cheap, fast,
   and cannot be talked out of by a persuasive-sounding question.
2. The LLM's own instruction-following (prompts.SYSTEM_PROMPT), for the
   case where retrieval clears the gate but the specific detail asked for
   still is not actually in the retrieved excerpts.

NOT_COVERED_SIMILARITY_THRESHOLD = 0.75 is not a guess, it comes from a
real measurement (see README "Retrieval confidence threshold"): 3
real in-index questions scored a minimum top-1 cosine similarity of 0.789
against this index, 4 real out-of-index questions (asking about drugs
never indexed here) scored a maximum top-1 similarity of 0.716. 0.75 sits
in that real gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import paths  # noqa: F401
from generate import GENERATORS, GenerationError
from prompts import NOT_COVERED_PHRASE, SYSTEM_PROMPT, build_user_prompt
from retrieve import RetrievedChunk, retrieve

NOT_COVERED_SIMILARITY_THRESHOLD = 0.75

CITATION_PATTERN = re.compile(r"\(([^,()]+),\s*([a-z_]+)\)")
# The label text itself uses its own internal cross-reference convention,
# "[see Warnings and Precautions (5.1)]", and models sometimes echo that
# style instead of the (Brand, section_name) format instructed in the
# prompt. Still a real, traceable pointer back to a label section, just
# not in the requested format, so it is tracked separately rather than
# silently counted as "no citation at all."
SOURCE_STYLE_CITATION_PATTERN = re.compile(r"\[see [^\]]+\]", re.IGNORECASE)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    provider: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    best_similarity: float = 0.0
    passed_similarity_gate: bool = False
    stated_not_covered: bool = False
    citations: list[tuple[str, str]] = field(default_factory=list)
    source_style_citations: list[str] = field(default_factory=list)


def extract_citations(answer_text: str) -> list[tuple[str, str]]:
    return [(brand.strip(), section.strip()) for brand, section in CITATION_PATTERN.findall(answer_text)]


def extract_source_style_citations(answer_text: str) -> list[str]:
    """Citations in the label's own bracket convention rather than the
    prompt's requested format, still traceable, just not format-compliant.
    """
    return SOURCE_STYLE_CITATION_PATTERN.findall(answer_text)


def answer_question(
    question: str,
    provider: str = "ollama",
    top_k: int = 5,
    threshold: float = NOT_COVERED_SIMILARITY_THRESHOLD,
) -> RAGAnswer:
    chunks: list[RetrievedChunk] = retrieve(question, top_k=top_k)
    best_similarity = chunks[0].similarity if chunks else 0.0

    if best_similarity < threshold:
        return RAGAnswer(
            question=question,
            answer=(
                f"This question is {NOT_COVERED_PHRASE}. The closest indexed label section "
                f"scored {best_similarity:.3f} cosine similarity, below the "
                f"{threshold:.2f} confidence threshold, so no answer is generated rather than "
                f"risking one built on an unrelated label."
            ),
            provider="none (similarity gate)",
            retrieved=chunks,
            best_similarity=best_similarity,
            passed_similarity_gate=False,
            stated_not_covered=True,
            citations=[],
        )

    generator = GENERATORS[provider]
    user_prompt = build_user_prompt(question, chunks)
    try:
        answer_text = generator(SYSTEM_PROMPT, user_prompt)
    except GenerationError as exc:
        return RAGAnswer(
            question=question,
            answer=f"generation failed: {exc}",
            provider=provider,
            retrieved=chunks,
            best_similarity=best_similarity,
            passed_similarity_gate=True,
            stated_not_covered=False,
            citations=[],
        )

    return RAGAnswer(
        question=question,
        answer=answer_text,
        provider=provider,
        retrieved=chunks,
        best_similarity=best_similarity,
        passed_similarity_gate=True,
        stated_not_covered=NOT_COVERED_PHRASE in answer_text.lower(),
        citations=extract_citations(answer_text),
        source_style_citations=extract_source_style_citations(answer_text),
    )
