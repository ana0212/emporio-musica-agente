"""Persistent ChromaDB vector store using cosine distance."""
from pathlib import Path
import chromadb


class PolicyVectorStore:
    def __init__(
        self,
        persist_dir: str | Path = "data/chroma",
        collection_name: str = "store_policies_multilingual_experiment",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate collection to prevent stale chunks after re-ingestion."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must match.")
        if not chunks:
            raise ValueError("No chunks were generated; refusing to create an empty index.")

        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks],
            embeddings=embeddings,
        )

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(1, int(top_k)), count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
            }
            for i in range(len(result["ids"][0]))
        ]

    def count(self) -> int:
        return self.collection.count()
