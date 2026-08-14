from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_answers import _build_user_prompt  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_set.json"


def test_prompt_includes_question_and_contexts():
    prompt = _build_user_prompt("What is the dose of X?", ["excerpt one", "excerpt two"])
    assert "What is the dose of X?" in prompt
    assert "excerpt one" in prompt
    assert "excerpt two" in prompt


def test_prompt_with_no_contexts_says_so():
    prompt = _build_user_prompt("What is the dose of X?", [])
    assert "no label excerpts retrieved" in prompt


def test_golden_set_has_required_fields():
    golden_set = json.loads(DATA_PATH.read_text())
    assert len(golden_set) >= 10
    for item in golden_set:
        assert item["question"]
        assert item["expected_set_id"]
        assert item["expected_section"]
        assert item["reference_answer"]


def test_golden_set_questions_are_unique():
    golden_set = json.loads(DATA_PATH.read_text())
    questions = [item["question"] for item in golden_set]
    assert len(questions) == len(set(questions))
