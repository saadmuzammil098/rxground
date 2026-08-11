import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import LabelChunk, LabelSection


def test_valid_chunk():
    chunk = LabelChunk(
        chunk_id="abc:dosage_and_administration:0",
        set_id="abc",
        brand_name="Lipitor",
        generic_name="ATORVASTATIN CALCIUM",
        section=LabelSection.dosage_and_administration,
        text="Take one tablet daily.",
    )
    assert chunk.section == LabelSection.dosage_and_administration


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        LabelChunk(
            chunk_id="abc:description:0",
            set_id="abc",
            brand_name="Lipitor",
            generic_name="ATORVASTATIN CALCIUM",
            section=LabelSection.description,
            text="",
        )


def test_invalid_section_rejected():
    with pytest.raises(ValidationError):
        LabelChunk(
            chunk_id="abc:not_a_section:0",
            set_id="abc",
            brand_name="Lipitor",
            generic_name="ATORVASTATIN CALCIUM",
            section="not_a_real_section",
            text="some text",
        )
