from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manifest import content_hash, has_changed, record_ingestion  # noqa: E402


def test_identical_content_has_same_hash():
    label = {"set_id": "abc", "boxed_warning": ["warning text"]}
    assert content_hash(label) == content_hash(dict(label))


def test_different_content_has_different_hash():
    label_a = {"set_id": "abc", "boxed_warning": ["warning text"]}
    label_b = {"set_id": "abc", "boxed_warning": ["a different warning"]}
    assert content_hash(label_a) != content_hash(label_b)


def test_unknown_set_id_has_changed():
    manifest: dict = {}
    label = {"set_id": "new-label", "boxed_warning": ["x"]}
    assert has_changed(manifest, "new-label", label) is True


def test_recorded_then_unchanged_content_is_not_changed():
    manifest: dict = {}
    label = {"set_id": "abc", "boxed_warning": ["warning text"]}
    record_ingestion(manifest, "abc", label, ["abc:boxed_warning:0"])
    assert has_changed(manifest, "abc", label) is False


def test_recorded_then_revised_content_has_changed():
    manifest: dict = {}
    label = {"set_id": "abc", "boxed_warning": ["warning text"]}
    record_ingestion(manifest, "abc", label, ["abc:boxed_warning:0"])
    revised = {"set_id": "abc", "boxed_warning": ["warning text, plus a new sentence"]}
    assert has_changed(manifest, "abc", revised) is True


def test_record_ingestion_stores_chunk_ids_and_hash():
    manifest: dict = {}
    label = {"set_id": "abc", "boxed_warning": ["warning text"]}
    record_ingestion(manifest, "abc", label, ["abc:boxed_warning:0", "abc:boxed_warning:1"])
    assert manifest["abc"]["chunk_ids"] == ["abc:boxed_warning:0", "abc:boxed_warning:1"]
    assert manifest["abc"]["content_hash"] == content_hash(label)
    assert "last_ingested_at" in manifest["abc"]
