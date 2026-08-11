import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking import MAX_CHUNK_CHARS, naive_fixed_size_chunks, section_aware_chunks
from schema import LabelSection

# Small synthetic fixture, not the real (DVC-tracked, not in git) data/raw
# directory, so these tests run offline in a fresh checkout with no
# `dvc pull` needed.
FIXTURE_LABELS = [
    {
        "set_id": "fixture-1",
        "openfda": {"brand_name": ["FixtureDrug"], "generic_name": ["FIXTURINE"]},
        "indications_and_usage": ["Indicated for treatment of the fixture condition."],
        "dosage_and_administration": ["Take one tablet twice daily. " * 60],
        "contraindications": ["Do not use if allergic to fixturine."],
    },
    {
        "set_id": "fixture-2",
        "openfda": {"brand_name": ["OtherFixture"], "generic_name": ["OTHERFIXTURINE"]},
        "warnings_and_cautions": ["May cause drowsiness in rare cases."],
    },
]


def test_section_aware_chunks_stay_within_one_section():
    chunks = section_aware_chunks(FIXTURE_LABELS)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk.section, LabelSection)
        assert len(chunk.text) <= MAX_CHUNK_CHARS
        assert chunk.text.strip() != ""


def test_section_aware_splits_long_sections_without_crossing_boundaries():
    chunks = section_aware_chunks(FIXTURE_LABELS)
    dosage_chunks = [c for c in chunks if c.section == LabelSection.dosage_and_administration]
    assert len(dosage_chunks) > 1  # the long dosage text must have been split
    contraindication_chunks = [c for c in chunks if c.section == LabelSection.contraindications]
    assert all("fixturine" in c.text.lower() or "allergic" in c.text.lower() for c in contraindication_chunks)


def test_section_aware_chunk_ids_are_unique():
    chunks = section_aware_chunks(FIXTURE_LABELS)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_naive_chunks_cover_the_same_labels():
    aware = section_aware_chunks(FIXTURE_LABELS)
    naive = naive_fixed_size_chunks(FIXTURE_LABELS)
    aware_set_ids = {c.set_id for c in aware}
    naive_set_ids = {c.set_id for c in naive}
    assert aware_set_ids == naive_set_ids


def test_naive_chunk_can_straddle_a_section_boundary():
    """fixture-1 has indications_and_usage (short) immediately followed by
    a long dosage_and_administration section. A 1200-character naive
    window starting right at the end of indications will pull in text from
    both, this is the concrete failure mode section-aware chunking avoids.
    """
    naive = naive_fixed_size_chunks(FIXTURE_LABELS)
    fixture1_chunks = [c for c in naive if c.set_id == "fixture-1"]
    assert len(fixture1_chunks) > 0
    combined = " ".join(c.text.lower() for c in fixture1_chunks)
    assert "fixture condition" in combined  # from indications_and_usage
    assert "twice daily" in combined  # from dosage_and_administration
    straddling = [c for c in fixture1_chunks if "fixture condition" in c.text.lower() and "twice daily" in c.text.lower()]
    assert len(straddling) >= 1, "expected at least one naive chunk to blend two sections together"
