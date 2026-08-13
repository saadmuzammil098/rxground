"""Same retrieval as task-2's retrieve.py, against the same task-1
section-aware Chroma index, extended to also return each chunk's
chunk_id. Task-2's RetrievedChunk did not carry chunk_id because task-2's
citation format was (Brand Name, section_name) only, task-3's citation
schema requires (set_id, section, chunk_id), so chunk_id has to come back
from Chroma's query response too.
"""

from __future__ import annotations

from dataclasses import dataclass

import paths  # noqa: F401
from index import SECTION_AWARE_COLLECTION, get_client, get_embedder


@dataclass
class RetrievedChunk:
    chunk_id: str
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

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=doc,
            brand_name=meta["brand_name"],
            generic_name=meta["generic_name"],
            section=meta["section"],
            set_id=meta["set_id"],
            similarity=_l2_squared_to_cosine_similarity(dist),
        )
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances)
    ]
