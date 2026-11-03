from src.rag.chunker import chunk_pages, normalize_text


def test_normalize_pdf_whitespace():
    assert normalize_text("Empório\n \nda\n\nMúsica") == "Empório da Música"


def test_chunker_preserves_page_metadata_and_word_boundaries():
    pages = [{"text": "palavra " * 250, "page": 3, "source": "policy.pdf"}]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunk["metadata"]["page"] == 3 for chunk in chunks)
    assert all(chunk["metadata"]["source"] == "policy.pdf" for chunk in chunks)
    assert all(not chunk["text"].startswith(" ") for chunk in chunks)
    assert all(not chunk["text"].endswith(" ") for chunk in chunks)
    assert all(chunk["text"].split()[0] == "palavra" for chunk in chunks)


def test_chunker_rejects_invalid_parameters():
    pages = [{"text": "texto", "page": 1, "source": "p.pdf"}]
    try:
        chunk_pages(pages, chunk_size=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
