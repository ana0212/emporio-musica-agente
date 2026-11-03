"""End-to-end policy ingestion and retrieval using local multilingual embeddings."""
from pathlib import Path
from .document_loader import load_pdf_pages
from .chunker import chunk_pages
from .embedding import LocalEmbedder, DEFAULT_MODEL
from .vector_store import PolicyVectorStore


DEFAULT_COLLECTION = "store_policies_multilingual_experiment"
DEFAULT_CHUNK_SIZE = 900
DEFAULT_OVERLAP = 150


def ingest_policy_pdf(
    pdf_path: str | Path,
    persist_dir: str | Path = "data/chroma",
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    model_name: str = DEFAULT_MODEL,
    reset_collection: bool = True,
) -> int:
    pages = load_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No chunks were generated from the PDF.")

    embedder = LocalEmbedder(model_name=model_name)
    embeddings = embedder.embed_documents([chunk["text"] for chunk in chunks])

    store = PolicyVectorStore(persist_dir, collection_name)
    if reset_collection:
        store.reset()
    store.upsert_chunks(chunks, embeddings)
    return store.count()


def retrieve_policy(
    query: str,
    persist_dir: str | Path = "data/chroma",
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 4,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    embedder = LocalEmbedder(model_name=model_name)
    query_embedding = embedder.embed_query(query)
    store = PolicyVectorStore(persist_dir, collection_name)
    return store.search(query_embedding, top_k=top_k)
