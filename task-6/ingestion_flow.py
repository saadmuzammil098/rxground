"""Ingestion, orchestrated as a Prefect flow, kept separate from serving.

Serving (task-2's, task-3's, and task-4's retrieval and generation) reads
whatever Chroma collection is currently there and never touches raw
label JSON or re-embeds anything. Ingestion is the only thing that
writes to that collection, and it runs as its own flow, on its own
schedule, independent of any query being served. This is the actual
point of separating the two: an in-progress label revision never blocks
or slows down a live pharmacist query, and a query never accidentally
triggers a re-embed.

Incremental, not a full rebuild: `manifest.py` tracks a content hash per
label. `ingest_one()` only rechunks and re-embeds a label whose hash
changed since the last run, and only deletes/replaces that label's own
chunks in Chroma, every other label's vectors are untouched. `ingest_all()`
is the manual backfill path, same logic, looped over every label, safe
to run at any time since unchanged labels are skipped automatically.

This operates on its own Chroma collection (`task-6/chroma_db/`), seeded
from task-1's real chunks, not task-1's own collection, so the label
revision simulated in `simulate_revision.py` never touches the shared
index every other task in this repo reads from.
"""

from __future__ import annotations

import json
from pathlib import Path

import paths  # noqa: F401

import chromadb
from chunking import section_aware_chunks
from index import EMBEDDING_MODEL_NAME, get_embedder
from prefect import flow, task

from manifest import build_entry, has_changed, load_manifest, save_manifest

TASK1_RAW_DIR = Path(__file__).resolve().parent.parent / "task-1" / "data" / "raw"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "labels_section_aware_v2"

_client = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_or_create_collection():
    client = get_client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME)


@task(retries=3, retry_delay_seconds=2)
def load_raw_label(path: Path) -> dict:
    return json.loads(path.read_text())


@task(retries=3, retry_delay_seconds=2)
def embed_chunks(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


@task
def upsert_label_chunks(set_id: str, chunks, embeddings: list[list[float]]) -> list[str]:
    collection = get_or_create_collection()
    # Delete this label's existing chunks first, an incremental
    # re-index replaces exactly one label's vectors, not the whole
    # collection, and not a stale mix of old-and-new chunks for the
    # same label.
    try:
        collection.delete(where={"set_id": set_id})
    except Exception:
        pass
    if not chunks:
        return []
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "set_id": c.set_id,
                "brand_name": c.brand_name,
                "generic_name": c.generic_name,
                "section": c.section.value,
                "part": c.part,
            }
            for c in chunks
        ],
    )
    return [c.chunk_id for c in chunks]


@flow(name="ingest-one-label")
def ingest_one(raw_path: Path, manifest: dict[str, dict]) -> tuple[str, dict | None]:
    """Returns (set_id, entry), entry is None when the label was
    unchanged and skipped. Deliberately does NOT mutate `manifest` in
    place and rely on the caller seeing that mutation, Prefect
    validates/copies flow parameters, so a dict mutated inside this
    subflow does not propagate back to ingest_all()'s own copy, an
    earlier version of this function did exactly that and silently
    produced an empty manifest, this return-value approach sidesteps it.
    """
    raw_label = load_raw_label(raw_path)
    set_id = raw_label.get("set_id", "unknown")

    if not has_changed(manifest, set_id, raw_label):
        return set_id, None

    chunks = section_aware_chunks([raw_label])
    embeddings = embed_chunks([c.text for c in chunks])
    chunk_ids = upsert_label_chunks(set_id, chunks, embeddings)
    entry = build_entry(raw_label, chunk_ids)
    return set_id, entry


@flow(name="ingest-all-labels", log_prints=True)
def ingest_all(raw_dir: Path = TASK1_RAW_DIR) -> dict[str, dict]:
    """The manual backfill path: safe to run at any time, unchanged
    labels are skipped via the manifest, so this never re-embeds
    anything that doesn't need it.
    """
    manifest = load_manifest()
    for raw_path in sorted(raw_dir.glob("*.json")):
        set_id, entry = ingest_one(raw_path, manifest)
        if entry is None:
            print(f"skipped {set_id} (unchanged)")
        else:
            manifest[set_id] = entry
            print(f"ingested {set_id} ({len(entry['chunk_ids'])} chunks)")
    save_manifest(manifest)
    return manifest


if __name__ == "__main__":
    ingest_all()
