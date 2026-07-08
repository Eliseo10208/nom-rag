"""ChromaDB persistent vector store over the NOM chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from nom_rag.config import settings
from nom_rag.corpus.parse import Chunk
from nom_rag.index.embed import Embedder

COLLECTION = "noms"


@dataclass
class RetrievedChunk:
    chunk_id: str
    nom_id: str
    short_id: str
    numeral: str
    section_title: str
    text: str
    score: float  # cosine similarity, higher is better


def _client(chroma_dir: Path | None = None) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(chroma_dir or settings.chroma_dir))


def build_index(
    chunks: list[Chunk], embedder: Embedder, chroma_dir: Path | None = None
) -> int:
    client = _client(chroma_dir)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    embeddings = embedder.embed_passages([c.text for c in chunks])
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "nom_id": c.nom_id,
                "short_id": c.short_id,
                "numeral": c.numeral,
                "section_title": c.section_title,
            }
            for c in chunks
        ],
    )
    return collection.count()


class Retriever:
    def __init__(self, embedder: Embedder, chroma_dir: Path | None = None):
        self.embedder = embedder
        self.collection = _client(chroma_dir).get_collection(COLLECTION)

    def query(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or settings.top_k
        [embedding] = self.embedder.embed_queries([question])
        result = self.collection.query(query_embeddings=[embedding], n_results=k)
        retrieved = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    nom_id=metadata["nom_id"],
                    short_id=metadata["short_id"],
                    numeral=metadata["numeral"],
                    section_title=metadata["section_title"],
                    text=document,
                    score=round(1.0 - distance, 4),
                )
            )
        return retrieved
