"""Hybrid search: BM25 (exact keyword match) plus task-1's dense vector
search, combined with reciprocal rank fusion (RRF), then optionally
filtered by drug class.

Why hybrid at all: dense embeddings compare *meaning*, and every drug
label's dosage_and_administration section reads in almost the same
shape, "recommended starting dose is X mg once daily", regardless of
which drug it is. That similarity in phrasing can outweigh the one word
that actually matters, the drug's own name, so a purely semantic search
can retrieve the wrong drug's dosing section for a query that names an
exact drug. BM25 scores exact term overlap, it does not blur "Norvasc"
into "Lipitor" the way a dense embedding can, see task-4's README for a
real measured case of exactly this.

Reciprocal rank fusion (RRF) combines the two ranked lists without
needing their raw scores to be on comparable scales, which they are not,
cosine similarity and BM25 score live in different ranges entirely. RRF
only uses each list's *rank*, chunk_score = sum(1 / (RRF_K + rank)) across
every list the chunk appears in, RRF_K=60 is the standard default from
the original RRF paper, big enough that no single top-1 finish dominates
the fused score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import paths  # noqa: F401
from chunking import section_aware_chunks
from drug_classes import DRUG_CLASS
from index import SECTION_AWARE_COLLECTION, get_client, get_embedder
from rank_bm25 import BM25Okapi
from schema import LabelChunk

RRF_K = 60
_WORD_RE = re.compile(r"[a-z0-9]+")

_chunks_cache: list[LabelChunk] | None = None
_bm25_cache: BM25Okapi | None = None
_chunk_by_id_cache: dict[str, LabelChunk] | None = None


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _get_chunks() -> list[LabelChunk]:
    global _chunks_cache, _chunk_by_id_cache
    if _chunks_cache is None:
        _chunks_cache = section_aware_chunks()
        _chunk_by_id_cache = {c.chunk_id: c for c in _chunks_cache}
    return _chunks_cache


def _get_bm25() -> BM25Okapi:
    global _bm25_cache
    if _bm25_cache is None:
        chunks = _get_chunks()
        _bm25_cache = BM25Okapi([_tokenize(c.text) for c in chunks])
    return _bm25_cache


def _chunk_by_id(chunk_id: str) -> LabelChunk:
    _get_chunks()
    return _chunk_by_id_cache[chunk_id]


@dataclass
class HybridResult:
    chunk_id: str
    set_id: str
    brand_name: str
    generic_name: str
    section: str
    text: str
    dense_rank: int | None
    bm25_rank: int | None
    rrf_score: float
    drug_class: str = field(default="")

    def __post_init__(self) -> None:
        self.drug_class = DRUG_CLASS.get(self.brand_name, "unclassified")


def _dense_ranked_ids(query: str, top_k: int) -> list[str]:
    client = get_client()
    collection = client.get_collection(SECTION_AWARE_COLLECTION)
    embedder = get_embedder()
    embedding = embedder.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["ids"][0]


def _bm25_ranked_ids(query: str, top_k: int) -> list[str]:
    chunks = _get_chunks()
    bm25 = _get_bm25()
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i].chunk_id for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(
    dense_ids: list[str], bm25_ids: list[str], k: int = RRF_K
) -> dict[str, float]:
    """Pure RRF math, no Chroma or BM25 dependency, so it can be unit
    tested without loading the embedding model. chunk_score =
    sum(1 / (k + rank)) across every ranked list the id appears in, ids
    that appear in neither list are absent from the result entirely.
    """
    dense_rank = {cid: rank for rank, cid in enumerate(dense_ids, start=1)}
    bm25_rank = {cid: rank for rank, cid in enumerate(bm25_ids, start=1)}
    scores: dict[str, float] = {}
    for chunk_id in set(dense_ids) | set(bm25_ids):
        score = 0.0
        if chunk_id in dense_rank:
            score += 1 / (k + dense_rank[chunk_id])
        if chunk_id in bm25_rank:
            score += 1 / (k + bm25_rank[chunk_id])
        scores[chunk_id] = score
    return scores


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_pool: int = 20,
    drug_class: str | None = None,
) -> list[HybridResult]:
    dense_ids = _dense_ranked_ids(query, candidate_pool)
    bm25_ids = _bm25_ranked_ids(query, candidate_pool)

    fused = reciprocal_rank_fusion(dense_ids, bm25_ids)
    dense_rank = {cid: rank for rank, cid in enumerate(dense_ids, start=1)}
    bm25_rank = {cid: rank for rank, cid in enumerate(bm25_ids, start=1)}

    scored: list[HybridResult] = []
    for chunk_id, rrf_score in fused.items():
        chunk = _chunk_by_id(chunk_id)
        if drug_class is not None and DRUG_CLASS.get(chunk.brand_name) != drug_class:
            continue
        scored.append(
            HybridResult(
                chunk_id=chunk_id,
                set_id=chunk.set_id,
                brand_name=chunk.brand_name,
                generic_name=chunk.generic_name,
                section=chunk.section.value,
                text=chunk.text,
                dense_rank=dense_rank.get(chunk_id),
                bm25_rank=bm25_rank.get(chunk_id),
                rrf_score=rrf_score,
            )
        )

    scored.sort(key=lambda r: r.rrf_score, reverse=True)
    return scored[:top_k]
