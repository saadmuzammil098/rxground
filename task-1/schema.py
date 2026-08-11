"""Pydantic models for one drug label chunk.

A LabelChunk is the unit that gets embedded and indexed. Each chunk carries
enough metadata (drug names, section) to cite back to the source label
section in later RAG days, and to let eval_retrieval.py check whether a
retrieved chunk is actually the section a question is asking about.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LabelSection(str, Enum):
    indications_and_usage = "indications_and_usage"
    dosage_and_administration = "dosage_and_administration"
    contraindications = "contraindications"
    warnings_and_cautions = "warnings_and_cautions"
    boxed_warning = "boxed_warning"
    drug_interactions = "drug_interactions"
    adverse_reactions = "adverse_reactions"
    overdosage = "overdosage"
    use_in_specific_populations = "use_in_specific_populations"
    pregnancy = "pregnancy"
    pediatric_use = "pediatric_use"
    geriatric_use = "geriatric_use"
    description = "description"


class LabelChunk(BaseModel):
    chunk_id: str = Field(description="stable id, f'{set_id}:{section}:{part}'")
    set_id: str = Field(description="openFDA set_id, stable identifier for one drug label")
    brand_name: str
    generic_name: str
    section: LabelSection
    part: int = Field(default=0, description="0-indexed split within an oversized section")
    text: str = Field(min_length=1)
