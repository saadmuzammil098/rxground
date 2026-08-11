"""Embeds both chunk sets (section-aware and naive fixed-size) with
BAAI/bge-base-en-v1.5 and indexes each into its own persistent Chroma
collection, so eval_retrieval.py can query both and compare real retrieval
quality rather than assume one is better.

Usage:
    ../.venv/bin/python index.py
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from chunking import naive_fixed_size_chunks, section_aware_chunks
from schema import LabelChunk

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

SECTION_AWARE_COLLECTION = "labels_section_aware"
NAIVE_COLLECTION = "labels_naive_fixed_size"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def _index_chunks(client: chromadb.ClientAPI, collection_name: str, chunks: list[LabelChunk]) -> None:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    embedder = get_embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "set_id": chunk.set_id,
                "brand_name": chunk.brand_name,
                "generic_name": chunk.generic_name,
                "section": chunk.section.value,
                "part": chunk.part,
            }
            for chunk in chunks
        ],
    )
    print(f"indexed {len(chunks)} chunks into '{collection_name}'")


def build_indices() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _index_chunks(client, SECTION_AWARE_COLLECTION, section_aware_chunks())
    _index_chunks(client, NAIVE_COLLECTION, naive_fixed_size_chunks())


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


if __name__ == "__main__":
    build_indices()
