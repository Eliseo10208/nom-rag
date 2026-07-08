"""Local embeddings with multilingual-e5. E5 models require task prefixes:
"query: " for questions and "passage: " for indexed text — retrieval quality
drops measurably without them."""

from __future__ import annotations


class Embedder:
    def __init__(self, model_name: str):
        # lazy import: keeps CLI commands that don't touch the index fast
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            [f"query: {t}" for t in texts], normalize_embeddings=True
        ).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        ).tolist()
