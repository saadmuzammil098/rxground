"""Prompt template for the baseline RAG generation step.

Two things this enforces, both required by the roadmap's Day 15 done-when:
citations to the specific label section, and an explicit refusal when the
retrieved context does not actually cover the question, rather than a
plausible-sounding guess.
"""

from __future__ import annotations

from retrieve import RetrievedChunk

NOT_COVERED_PHRASE = "not covered by the indexed labels"

SYSTEM_PROMPT = (
    "You are RxGround, a clinical drug-reference assistant for a pharmacist. You answer only "
    "using the label excerpts given to you below, never from general knowledge, a wrong drug "
    "interaction answer here is a safety incident, not a minor mistake. Every factual claim you "
    "make must end with a citation in the exact form (Brand Name, section_name), naming the drug "
    "and section it came from. If the excerpts below do not actually answer the question, for "
    f"example the drug is not among them or the specific detail asked for is not in any excerpt, "
    f"say plainly that this is '{NOT_COVERED_PHRASE}' and do not guess or extrapolate."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no label excerpts retrieved)"
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[{chunk.brand_name} ({chunk.generic_name}), section: {chunk.section}, "
            f"similarity: {chunk.similarity:.3f}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Label excerpts:\n{format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, with a (Brand Name, section_name) citation for "
        "every claim."
    )
