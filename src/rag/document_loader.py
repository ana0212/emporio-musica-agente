"""Extract PDF text while preserving page-level source metadata."""
from pathlib import Path
from pypdf import PdfReader


def load_pdf_pages(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "text": text,
                "page": page_number,
                "source": path.name,
            })

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")
    return pages
