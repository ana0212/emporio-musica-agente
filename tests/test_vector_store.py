from src.rag.vector_store import PolicyVectorStore


def test_upsert_and_search(tmp_path):
    store = PolicyVectorStore(tmp_path, "test_policy_collection")
    chunks = [
        {
            "id": "p1-c1",
            "text": "A loja vende instrumentos musicais.",
            "metadata": {"source": "policy.pdf", "page": 1},
        },
        {
            "id": "p2-c1",
            "text": "O atendimento funciona de segunda a sexta.",
            "metadata": {"source": "policy.pdf", "page": 2},
        },
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store.upsert_chunks(chunks, embeddings)

    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "p1-c1"
    assert results[0]["metadata"]["page"] == 1


def test_reset_removes_stale_chunks(tmp_path):
    store = PolicyVectorStore(tmp_path, "test_reset_collection")
    store.upsert_chunks(
        [{"id": "old", "text": "old chunk", "metadata": {"page": 1}}],
        [[1.0, 0.0]],
    )
    store.reset()
    assert store.count() == 0
