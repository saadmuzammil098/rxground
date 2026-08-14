from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_guardrail import check_scope  # noqa: E402


def test_reference_question_is_in_scope():
    in_scope, category = check_scope("What are the contraindications for Lipitor?")
    assert in_scope is True
    assert category is None


def test_prescriptive_question_is_flagged():
    in_scope, category = check_scope("What should I take for my headache?")
    assert in_scope is False
    assert category == "prescriptive"


def test_diagnostic_question_is_flagged():
    in_scope, category = check_scope("Do I have a heart attack right now?")
    assert in_scope is False
    assert category == "diagnostic"


def test_case_insensitive():
    in_scope, _ = check_scope("SHOULD I TAKE ibuprofen for this?")
    assert in_scope is False


def test_third_person_reference_question_is_not_flagged():
    # a pharmacist reviewing a patient's chart, not asking for personal advice
    in_scope, category = check_scope("What interactions does warfarin have with aspirin?")
    assert in_scope is True
    assert category is None
