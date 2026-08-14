"""Cross-encoder re-ranking on top of hybrid_retrieve's fused candidates.

RRF fusion combines two rankings cheaply but only ever looks at each
chunk in isolation against the query through two coarse signals (does it
embed nearby, does it share keywords). A cross-encoder scores the query
and the chunk text together, in one forward pass, so it can pick up on
finer-grained relevance a bi-encoder or BM25 alone would miss.
cross-encoder/ms-marco-MiniLM-L-6-v2 is a small, free, well-established
Hugging Face reranker, trained specifically for this "rerank a retrieved
candidate list" job.
"""

from __future__ import annotations

from hybrid_retrieve import HybridResult

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates: list[HybridResult], top_k: int = 5) -> list[HybridResult]:
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
