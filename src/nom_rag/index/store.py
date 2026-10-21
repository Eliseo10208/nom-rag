"""ChromaDB persistent vector store over the NOM chunks."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb

from nom_rag.config import settings
from nom_rag.corpus.parse import Chunk
from nom_rag.index.embed import Embedder

COLLECTION = "noms"

# Bibliography chunks are citation lists, not norm content; they attract
# keyword-heavy queries (e.g. "violencia") and displace real clauses.
# Measured impact of excluding them: evals/runs/ baseline-with-bib vs baseline.
EXCLUDED_SECTION_PREFIXES = ("bibliograf",)

# Dense retrieval finds the topic; a short obligation can still rank just
# outside top-k. Re-rank a small dense candidate pool with a bounded lexical
# bonus, then return only k chunks to keep the generation context unchanged.
LEXICAL_WEIGHT = 0.03
_STOPWORDS = frozenset(
    "a al con como cual cuando cuanto de del debe deben el en es esa esta este "
    "hay la las lo los para por que se ser si son su sus un una y nom stps "
    "trabajo seguridad salud".split()
)


def _terms(text: str) -> list[str]:
    folded = "".join(
        char
        for char in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(char)
    )
    return [
        term
        for term in re.findall(r"[a-z]+", folded)
        if len(term) > 2 and term not in _STOPWORDS
    ]


def _indexable(chunk: Chunk) -> bool:
    return not chunk.section_title.lower().startswith(EXCLUDED_SECTION_PREFIXES)


@dataclass
class RetrievedChunk:
    chunk_id: str
    nom_id: str
    short_id: str
    numeral: str
    section_title: str
    text: str
    score: float  # cosine similarity plus a small lexical bonus, higher is better


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
    chunks = [c for c in chunks if _indexable(c)]
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
        indexed = self.collection.get(include=["documents"])
        self._term_counts = {
            chunk_id: Counter(_terms(document or ""))
            for chunk_id, document in zip(indexed["ids"], indexed["documents"] or [])
        }
        self._document_lengths = {
            chunk_id: sum(counts.values())
            for chunk_id, counts in self._term_counts.items()
        }
        self._document_count = len(self._term_counts)
        self._document_frequency = Counter(
            term for counts in self._term_counts.values() for term in counts
        )
        self._average_length = (
            sum(self._document_lengths.values()) / self._document_count
            if self._document_count
            else 1.0
        )

    def _lexical_score(self, query_terms: set[str], chunk_id: str) -> float:
        counts = self._term_counts.get(chunk_id, Counter())
        length = self._document_lengths.get(chunk_id, 0)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            document_frequency = self._document_frequency[term]
            idf = math.log(
                1 + (self._document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            score += idf * frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * length / self._average_length)
            )
        return score

    def query(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or settings.top_k
        if not self._document_count:
            return []
        [embedding] = self.embedder.embed_queries([question])
        candidate_k = min(max(2 * k, 10), self._document_count)
        result = self.collection.query(query_embeddings=[embedding], n_results=candidate_k)
        query_terms = set(_terms(question))
        lexical_scores = [
            self._lexical_score(query_terms, chunk_id)
            for chunk_id in result["ids"][0]
        ]
        max_lexical = max(lexical_scores, default=0.0)
        scored = []
        for chunk_id, document, metadata, distance, lexical in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0],
            result["distances"][0], lexical_scores,
        ):
            relevance = 1.0 - distance
            if max_lexical:
                relevance += LEXICAL_WEIGHT * lexical / max_lexical
            scored.append(
                (relevance, RetrievedChunk(
                    chunk_id=chunk_id,
                    nom_id=metadata["nom_id"],
                    short_id=metadata["short_id"],
                    numeral=metadata["numeral"],
                    section_title=metadata["section_title"],
                    text=document,
                    score=round(relevance, 4),
                ))
            )
        return [
            chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:k]
        ]
