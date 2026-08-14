"""The ingestion manifest: one JSON file tracking, per drug label
(set_id), a content hash and the chunk_ids currently indexed for it.

This is what makes incremental re-indexing possible at all. Without it,
"has this label changed since last ingestion" has no answer, and "which
chunks belong to this label, so I can delete the stale ones before
inserting the revised ones" has no answer either, ingestion would have
no choice but to rebuild the whole index every run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent / "state" / "ingestion_manifest.json"


def content_hash(raw_label: dict) -> str:
    canonical = json.dumps(raw_label, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(manifest: dict[str, dict]) -> None:
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def build_entry(raw_label: dict, chunk_ids: list[str]) -> dict:
    """A standalone manifest entry, returned rather than written directly
    into a caller's manifest dict. Ingestion runs `ingest_one()` as its
    own Prefect subflow, and Prefect validates/copies flow parameters, so
    a dict mutated inside a subflow does not propagate back to the
    caller's own copy, an in-place record_ingestion(manifest, ...) call
    across that boundary silently no-ops. Returning the entry and letting
    the caller assign `manifest[set_id] = entry` itself, in its own
    scope, sidesteps that entirely.
    """
    return {
        "content_hash": content_hash(raw_label),
        "chunk_ids": chunk_ids,
        "last_ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def record_ingestion(manifest: dict[str, dict], set_id: str, raw_label: dict, chunk_ids: list[str]) -> None:
    """Same-process convenience wrapper around build_entry(), safe to use
    only when manifest is mutated and read back in the same Python scope,
    not across a Prefect flow/subflow boundary, see build_entry()'s
    docstring and ingestion_flow.ingest_all() for why that distinction
    matters here.
    """
    manifest[set_id] = build_entry(raw_label, chunk_ids)


def has_changed(manifest: dict[str, dict], set_id: str, raw_label: dict) -> bool:
    entry = manifest.get(set_id)
    if entry is None:
        return True
    return entry["content_hash"] != content_hash(raw_label)
