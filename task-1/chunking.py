"""Two chunking strategies over the same raw openFDA labels, so
eval_retrieval.py can measure the real difference between them instead of
just asserting one is better.

section_aware_chunks(): one chunk per label section (dosing, warnings,
interactions, and so on stay separate), the strategy the roadmap asks for.
Long sections are split further, but never across a section boundary, a
dosing sentence never ends up sharing a chunk with a contraindication.

naive_fixed_size_chunks(): the label's full text (every section
concatenated with no boundary awareness) sliced into fixed-size
character windows. This is the baseline every "why does chunking
strategy matter" comparison needs, splits land wherever the character
count runs out, mid-section, or mid-sentence.
"""

from __future__ import annotations

import json
from pathlib import Path

from schema import LabelChunk, LabelSection

DATA_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

MAX_CHUNK_CHARS = 1200
NAIVE_CHUNK_CHARS = 1200
NAIVE_CHUNK_OVERLAP = 150


def _clean_text(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    return str(value).strip()


def _load_raw_labels() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(DATA_RAW_DIR.glob("*.json"))]


def _resolve_labels(raw_labels: list[dict] | None) -> list[dict]:
    """raw_labels lets tests inject small synthetic fixtures instead of
    reading the real DVC-tracked data/raw directory, which is not present
    in a fresh git checkout before `dvc pull` runs.
    """
    return raw_labels if raw_labels is not None else _load_raw_labels()


def _names(label: dict) -> tuple[str, str]:
    openfda = label.get("openfda", {})
    brand = openfda.get("brand_name", ["unknown brand"])[0]
    generic = openfda.get("generic_name", ["unknown generic"])[0]
    return brand, generic


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        start = end
    return parts


def section_aware_chunks(raw_labels: list[dict] | None = None) -> list[LabelChunk]:
    chunks: list[LabelChunk] = []
    for label in _resolve_labels(raw_labels):
        set_id = label.get("set_id", "unknown")
        brand, generic = _names(label)
        for section in LabelSection:
            raw_value = label.get(section.value)
            if not raw_value:
                continue
            text = _clean_text(raw_value)
            if not text:
                continue
            for part_index, part_text in enumerate(_split_long_text(text, MAX_CHUNK_CHARS)):
                chunks.append(
                    LabelChunk(
                        chunk_id=f"{set_id}:{section.value}:{part_index}",
                        set_id=set_id,
                        brand_name=brand,
                        generic_name=generic,
                        section=section,
                        part=part_index,
                        text=part_text,
                    )
                )
    return chunks


def naive_fixed_size_chunks(raw_labels: list[dict] | None = None) -> list[LabelChunk]:
    """Concatenates every section into one blob per label, in schema field
    order, then slices it into fixed-size windows with no regard for where
    one section ends and the next begins. section is set to whichever
    LabelSection the window happens to start inside, this is itself part of
    the problem naive chunking has: a window can straddle two sections, so
    "which section is this chunk" stops being a well-defined question.
    """
    chunks: list[LabelChunk] = []
    for label in _resolve_labels(raw_labels):
        set_id = label.get("set_id", "unknown")
        brand, generic = _names(label)

        full_text = ""
        section_boundaries: list[tuple[int, LabelSection]] = []
        for section in LabelSection:
            raw_value = label.get(section.value)
            if not raw_value:
                continue
            text = _clean_text(raw_value)
            if not text:
                continue
            section_boundaries.append((len(full_text), section))
            full_text += text + " "

        if not full_text.strip():
            continue

        start = 0
        part_index = 0
        while start < len(full_text):
            end = min(start + NAIVE_CHUNK_CHARS, len(full_text))
            window_text = full_text[start:end].strip()
            if window_text:
                owning_section = section_boundaries[0][1]
                for boundary_offset, section in section_boundaries:
                    if boundary_offset <= start:
                        owning_section = section
                    else:
                        break
                chunks.append(
                    LabelChunk(
                        chunk_id=f"{set_id}:naive:{part_index}",
                        set_id=set_id,
                        brand_name=brand,
                        generic_name=generic,
                        section=owning_section,
                        part=part_index,
                        text=window_text,
                    )
                )
                part_index += 1
            start += NAIVE_CHUNK_CHARS - NAIVE_CHUNK_OVERLAP
    return chunks


if __name__ == "__main__":
    aware = section_aware_chunks()
    naive = naive_fixed_size_chunks()
    print(f"section-aware chunks: {len(aware)}")
    print(f"naive fixed-size chunks: {len(naive)}")
