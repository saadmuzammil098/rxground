"""Retrieval step of the baseline RAG pipeline. Reuses task-1's already
built section-aware Chroma index and embedding model directly, does not
rebuild or duplicate either.

Chroma's default distance metric is squared L2, not cosine, but every
embedding here is normalized (unit length), so the two are a fixed
one-to-one function of each other: for unit vectors,
l2_squared = 2 * (1 - cosine_similarity). similarity() below converts back
to cosine similarity so retrieved chunks have one consistent, comparable
score, on a 0 to 1 scale where 1 is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import paths  # noqa: F401
from index import SECTION_AWARE_COLLECTION, get_client, get_embedder


@dataclass
class RetrievedChunk:
    text: str
    brand_name: str
    generic_name: str
    section: str
    set_id: str
    similarity: float


def _l2_squared_to_cosine_similarity(l2_squared: float) -> float:
    return 1 - (l2_squared / 2)


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    client = get_client()
    collection = client.get_collection(SECTION_AWARE_COLLECTION)
    embedder = get_embedder()

    query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        RetrievedChunk(
            text=doc,
            brand_name=meta["brand_name"],
            generic_name=meta["generic_name"],
            section=meta["section"],
            set_id=meta["set_id"],
            similarity=_l2_squared_to_cosine_similarity(dist),
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
