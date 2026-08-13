"""Second-pass groundedness check: a lexical overlap heuristic between a
claim's text and the chunk text it cites, rather than an LLM-as-judge call.
Chosen over a second LLM call because the failure mode being checked for,
a claim's text saying something its cited chunk does not actually contain,
is a fact a second LLM could just as easily rubber-stamp as the first one
hallucinated. A cheap, deterministic word-overlap score cannot be talked
into agreeing with a fabricated claim.

score(claim) = fraction of the claim's significant words (lowercased,
stopwords and punctuation stripped) that also appear in the concatenated
text of every chunk that claim cites. 1.0 means every content word in the
claim is traceable to the cited text, 0.0 means none of it is.
"""

from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "has", "have", "if", "in", "is", "it", "its", "may",
    "not", "of", "on", "or", "should", "than", "that", "the", "this",
    "to", "was", "were", "will", "with", "you", "your",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _significant_words(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def claim_groundedness(claim_text: str, cited_texts: list[str]) -> float:
    claim_words = _significant_words(claim_text)
    if not claim_words:
        return 0.0
    source_words = _significant_words(" ".join(cited_texts))
    overlap = claim_words & source_words
    return len(overlap) / len(claim_words)


def answer_groundedness(claim_texts_and_sources: list[tuple[str, list[str]]]) -> float:
    """Mean per-claim groundedness across an answer's claims. Callers pass
    an empty list for a refused answer, in which case there is nothing to
    ground and this returns 1.0 by convention (a refusal cannot hallucinate).
    """
    if not claim_texts_and_sources:
        return 1.0
    scores = [claim_groundedness(text, sources) for text, sources in claim_texts_and_sources]
    return sum(scores) / len(scores)
