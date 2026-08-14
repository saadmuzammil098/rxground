"""Public entry point for task-4: scope guardrail, then query expansion,
then hybrid (BM25 + dense) search, then cross-encoder reranking.

Order matters. The scope check runs first and short-circuits everything
else, there is no reason to spend an embedding call, a BM25 lookup, or a
reranker pass on a question RxGround should not be answering at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hybrid_retrieve import HybridResult, hybrid_search
from query_expansion import expand_query
from rerank import rerank
from scope_guardrail import OUT_OF_SCOPE_REFUSAL, check_scope


@dataclass
class AdvancedRetrievalResult:
    query: str
    expanded_query: str
    in_scope: bool
    scope_category: str | None
    refusal: str | None = None
    chunks: list[HybridResult] = field(default_factory=list)


def retrieve(
    query: str,
    top_k: int = 5,
    candidate_pool: int = 20,
    drug_class: str | None = None,
    use_reranker: bool = True,
) -> AdvancedRetrievalResult:
    in_scope, category = check_scope(query)
    if not in_scope:
        return AdvancedRetrievalResult(
            query=query,
            expanded_query=query,
            in_scope=False,
            scope_category=category,
            refusal=OUT_OF_SCOPE_REFUSAL,
        )

    expanded = expand_query(query)
    candidates = hybrid_search(expanded, top_k=candidate_pool, candidate_pool=candidate_pool, drug_class=drug_class)

    final_chunks = rerank(expanded, candidates, top_k=top_k) if use_reranker else candidates[:top_k]

    return AdvancedRetrievalResult(
        query=query,
        expanded_query=expanded,
        in_scope=True,
        scope_category=None,
        chunks=final_chunks,
    )
