"""Local multilingual embeddings; no paid embedding API is used."""
from __future__ import annotations
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 8,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()
