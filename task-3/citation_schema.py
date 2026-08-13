"""Pydantic models for the structured, citation-enforced generation output.

task-2's citation check was a regex over free-text (Brand, section) pairs,
which task-2's own README documents as unreliable, the model often cited
in a different real format that the regex missed. task-3 asks the LLM to
return structured JSON instead, one claim per factual statement, each
claim carrying one or more citations that name the exact retrieved chunk
it came from (set_id, section, chunk_id). GroundedResponse validates the
shape (a non-refused answer must have at least one claim, every claim must
carry at least one citation), a second, non-Pydantic check in rag.py then
confirms each cited chunk_id actually was one of the chunks retrieved for
this question, since Pydantic alone cannot know that, it only knows the
LLM produced a string that looks like a chunk id.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    set_id: str = Field(min_length=1, description="openFDA set_id, the drug label id")
    section: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class Claim(BaseModel):
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class GroundedResponse(BaseModel):
    refused: bool
    refusal_reason: str | None = None
    claims: list[Claim] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refusal_and_claims_are_consistent(self) -> "GroundedResponse":
        if self.refused and self.claims:
            raise ValueError("a refused response must not carry claims")
        if not self.refused and not self.claims:
            raise ValueError("a non-refused response must carry at least one claim")
        if self.refused and not self.refusal_reason:
            raise ValueError("a refused response must state a refusal_reason")
        return self
