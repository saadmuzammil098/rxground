"""Query expansion for brand/generic drug name variants.

A pharmacist might type "Lipitor" or "atorvastatin" for the same
question, and dense embeddings do not reliably treat those as
interchangeable (see task-4's README for a real measured case). This
expands a query with the counterpart name(s) whenever a known brand or
generic name is found in it, so both the dense and BM25 legs of hybrid
search see both forms.
"""

from __future__ import annotations

import re

from drug_classes import BRAND_LOWER_TO_BRAND, GENERIC_NAME, GENERIC_TO_BRAND

_WORD_RE = re.compile(r"[a-z0-9]+")


def expand_query(query: str) -> str:
    query_lower = query.lower()
    query_words = set(_WORD_RE.findall(query_lower))

    additions: list[str] = []

    for brand_first_word, brand in BRAND_LOWER_TO_BRAND.items():
        if brand_first_word in query_words:
            for generic in GENERIC_NAME[brand]:
                if generic not in query_lower:
                    additions.append(generic)

    for generic_word, brand in GENERIC_TO_BRAND.items():
        if generic_word in query_words and brand.lower() not in query_lower:
            additions.append(brand)

    if not additions:
        return query

    # De-duplicate while preserving order, a query can match more than one
    # brand/generic pair (e.g. mentions two drugs).
    seen: set[str] = set()
    unique_additions = [a for a in additions if not (a.lower() in seen or seen.add(a.lower()))]
    return f"{query} ({' '.join(unique_additions)})"
