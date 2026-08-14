"""Scope guardrail: keeps RxGround at reference lookup, never diagnostic
or prescriptive advice, the one piece of the roadmap's anti-hallucination
task that task-3's citation-and-refusal work did not cover. task-3
answers "what does the label say", this answers a different question
first, "is this even a question RxGround should be answering at all,"
before retrieval or generation ever runs.

A cheap keyword/pattern heuristic, not an LLM call, on purpose, the same
reasoning as task-3's groundedness check: the failure being guarded
against, the system giving a real person medical advice about their own
body, is too high-stakes to depend on a second model call correctly
noticing it. A pattern match is slower to write well but cannot be
talked out of firing.

Two categories are flagged:
- Diagnostic: the question asks RxGround to diagnose the asker
  ("do I have", "am I having", "what's wrong with me").
- Prescriptive: the question asks RxGround to decide what the asker
  personally should take or do ("what should I take", "should I stop
  taking", "can I take X and Y together", framed about "I"/"me"/"my").

Honest limitation, documented rather than hidden: this is a heuristic
over surface phrasing, not language understanding. A rephrased question
with the same intent but none of these patterns will slip through, and a
legitimate reference question that happens to use first-person phrasing
("what interactions should I be aware of when reviewing this
patient's chart") could be flagged when it should not be. See the
README for real examples of both.
"""

from __future__ import annotations

import re

_DIAGNOSTIC_PATTERNS = [
    r"\bdo i have\b",
    r"\bam i having\b",
    r"\bwhat'?s wrong with me\b",
    r"\bwhat is wrong with me\b",
    r"\bdiagnose me\b",
    r"\bdo you think i have\b",
]

_PRESCRIPTIVE_PATTERNS = [
    r"\bshould i take\b",
    r"\bshould i stop taking\b",
    r"\bshould i start taking\b",
    r"\bcan i take\b",
    r"\bwhat should i take\b",
    r"\bwhich (drug|medication|medicine) should i\b",
    r"\bhow much should i take\b",
    r"\bis it safe for me to\b",
    r"\bcan you prescribe\b",
    r"\bwhat dose should i give myself\b",
]

_ALL_PATTERNS = [(re.compile(p), "diagnostic") for p in _DIAGNOSTIC_PATTERNS] + [
    (re.compile(p), "prescriptive") for p in _PRESCRIPTIVE_PATTERNS
]

OUT_OF_SCOPE_REFUSAL = (
    "RxGround is a reference lookup tool over FDA drug labeling, it does not diagnose "
    "a person's condition or decide what a specific person should take. Please consult a "
    "pharmacist or physician for personal medical advice. If you want, rephrase this as a "
    "reference question about a specific drug's label content instead."
)


def check_scope(question: str) -> tuple[bool, str | None]:
    """Returns (in_scope, category). category is None when in scope,
    otherwise "diagnostic" or "prescriptive", whichever matched first.
    """
    question_lower = question.lower()
    for pattern, category in _ALL_PATTERNS:
        if pattern.search(question_lower):
            return False, category
    return True, None
