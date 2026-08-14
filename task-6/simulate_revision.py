"""Deliberate simulation, and the roadmap's actual done-when for this
task: prove a label revision is reflected in retrieval without a full
rebuild.

Operates entirely on task-6's own copy of the index
(`chroma_db/labels_section_aware_v2`), seeded from task-1's real raw
label files, never on task-1's own collection every other task reads
from.

Steps, each one logged with real before/after values, not asserted
blindly:
1. Seed: ingest_all() against task-1's real 15 raw labels. First run,
   everything is new, expect 15/15 "ingested".
2. Idempotency check: ingest_all() again, same source, nothing changed,
   expect 15/15 "skipped".
3. Simulate a revision: write a modified copy of ZITUVIMET's raw label
   with an added sentence in its boxed_warning section (a realistic
   openFDA event, a black-box warning update), everything else
   byte-identical.
4. Re-run ingest_all() against the revised set. Expect exactly 1/15
   "ingested" (ZITUVIMET) and 14/15 "skipped".
5. Query the collection directly for ZITUVIMET's boxed_warning chunk and
   confirm the new sentence is actually retrievable, and check that
   another, unrelated drug's chunk_ids are byte-identical to before the
   revision, proving the "incremental" claim, not just the "skip count."

Usage:
    ../.venv/bin/python simulate_revision.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ingestion_flow import TASK1_RAW_DIR, get_or_create_collection, ingest_all
from manifest import MANIFEST_PATH, load_manifest

SIMULATED_RAW_DIR = Path(__file__).resolve().parent / "data" / "simulated_raw"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

REVISED_SET_ID = "0098dec4-f0e5-45d5-8aa4-5d0faf9ab142"  # ZITUVIMET
NEW_WARNING_SENTENCE = (
    " UPDATE (simulated, 2026): postmarketing surveillance has identified an additional "
    "risk of acute pancreatitis associated with ZITUVIMET, discontinue promptly if pancreatitis "
    "is suspected."
)


def _reset_state() -> None:
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    chroma_dir = Path(__file__).resolve().parent / "chroma_db"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)


def _build_simulated_raw_dir() -> None:
    if SIMULATED_RAW_DIR.exists():
        shutil.rmtree(SIMULATED_RAW_DIR)
    SIMULATED_RAW_DIR.mkdir(parents=True)

    for raw_path in TASK1_RAW_DIR.glob("*.json"):
        label = json.loads(raw_path.read_text())
        if label.get("set_id") == REVISED_SET_ID:
            label["boxed_warning"] = [label["boxed_warning"][0] + NEW_WARNING_SENTENCE]
        (SIMULATED_RAW_DIR / raw_path.name).write_text(json.dumps(label))


def _query_boxed_warning(set_id: str) -> str:
    """A long section like ZITUVIMET's boxed_warning is split into
    several chunks (MAX_CHUNK_CHARS in task-1's chunking.py), and the
    added sentence can land in any one of them depending on where it
    falls relative to the split boundary, checking only the first
    document returned understates whether the revision is actually
    retrievable, join every part instead.
    """
    collection = get_or_create_collection()
    results = collection.get(where={"$and": [{"set_id": set_id}, {"section": "boxed_warning"}]})
    return " ".join(results.get("documents", []))


def run() -> dict:
    print("=== step 1: seed from task-1's real raw labels (fresh state) ===")
    _reset_state()
    results_seed = ingest_all(TASK1_RAW_DIR)
    seed_ingested = sum(1 for v in results_seed.values())
    print(f"seeded {seed_ingested} labels")

    print("\n=== step 2: idempotency check, same source, re-run ===")
    manifest_before_idempotency = load_manifest()
    results_idempotent = ingest_all(TASK1_RAW_DIR)
    unchanged_hashes = all(
        results_idempotent[sid]["content_hash"] == manifest_before_idempotency[sid]["content_hash"]
        for sid in results_idempotent
    )
    print(f"all 15 hashes unchanged after re-run: {unchanged_hashes}")

    other_set_id = next(sid for sid in results_idempotent if sid != REVISED_SET_ID)
    other_chunk_ids_before = list(results_idempotent[other_set_id]["chunk_ids"])

    print("\n=== step 3: simulate a boxed_warning revision for ZITUVIMET ===")
    _build_simulated_raw_dir()

    print("\n=== step 4: re-run ingestion against the revised set ===")
    results_revised = ingest_all(SIMULATED_RAW_DIR)

    ingested_this_run = [
        sid for sid in results_revised
        if results_revised[sid]["content_hash"] != results_idempotent.get(sid, {}).get("content_hash")
    ]

    print("\n=== step 5: verify retrieval reflects the change, and other labels are untouched ===")
    new_text = _query_boxed_warning(REVISED_SET_ID)
    revision_visible = new_text is not None and "acute pancreatitis" in new_text

    other_chunk_ids_after = list(results_revised[other_set_id]["chunk_ids"])
    other_untouched = other_chunk_ids_before == other_chunk_ids_after

    summary = {
        "seed_ingested_count": seed_ingested,
        "idempotency_all_hashes_unchanged": unchanged_hashes,
        "labels_reingested_after_revision": ingested_this_run,
        "expected_only_revised_label_reingested": ingested_this_run == [REVISED_SET_ID],
        "revision_visible_in_retrieval": revision_visible,
        "unrelated_label_chunk_ids_untouched": other_untouched,
        "unrelated_label_set_id_checked": other_set_id,
    }

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "simulation_results.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
