from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_answers import _build_user_prompt  # noqa: E402

# data/golden_set.json is DVC-tracked (see data/.gitignore), not present in
# a plain git checkout without `dvc pull`, exactly like task-1's raw labels.
# CI never runs `dvc pull` (the DVC remote is a local-only Floci emulator,
# unreachable from a GitHub Actions runner), so these tests validate the
# structural checks against a small inline fixture, the same reasoning
# task-1's README gives for its own tests, and only run the same checks
# against the real file when it happens to be present locally.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_set.json"

FIXTURE_GOLDEN_SET = [
    {
        "question": "What are the contraindications for Drug A?",
        "expected_set_id": "set-a",
        "expected_section": "contraindications",
        "reference_answer": "Drug A is contraindicated in condition X.",
    },
    {
        "question": "What is the boxed warning for Drug B?",
        "expected_set_id": "set-b",
        "expected_section": "boxed_warning",
        "reference_answer": "Drug B carries a boxed warning for condition Y.",
    },
]


def _assert_valid_golden_set(golden_set: list[dict]) -> None:
    assert len(golden_set) >= 2
    for item in golden_set:
        assert item["question"]
        assert item["expected_set_id"]
        assert item["expected_section"]
        assert item["reference_answer"]
    questions = [item["question"] for item in golden_set]
    assert len(questions) == len(set(questions))


def test_prompt_includes_question_and_contexts():
    prompt = _build_user_prompt("What is the dose of X?", ["excerpt one", "excerpt two"])
    assert "What is the dose of X?" in prompt
    assert "excerpt one" in prompt
    assert "excerpt two" in prompt


def test_prompt_with_no_contexts_says_so():
    prompt = _build_user_prompt("What is the dose of X?", [])
    assert "no label excerpts retrieved" in prompt


def test_fixture_golden_set_is_structurally_valid():
    _assert_valid_golden_set(FIXTURE_GOLDEN_SET)


@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/golden_set.json requires `dvc pull`, not available in CI")
def test_real_golden_set_is_structurally_valid():
    golden_set = json.loads(DATA_PATH.read_text())
    assert len(golden_set) >= 10
    _assert_valid_golden_set(golden_set)
