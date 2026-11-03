"""Whitespace normalization and word-boundary-aware PDF chunking."""
import re


def normalize_text(text: str) -> str:
    """Collapse line breaks and repeated whitespace introduced by PDF extraction."""
    return re.sub(r"\s+", " ", text).strip()


def _move_to_word_boundary(text: str, position: int, direction: str) -> int:
    """Move a character position to a nearby whitespace boundary."""
    if position <= 0 or position >= len(text):
        return position

    if direction == "forward":
        whitespace = text.find(" ", position)
        return whitespace if whitespace != -1 else len(text)

    whitespace = text.rfind(" ", 0, position)
    return whitespace + 1 if whitespace != -1 else position


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[dict]:
    """Split each PDF page independently, retaining source and page metadata.

    The target chunk size is approximate: boundaries are moved to whitespace
    to avoid cutting words. Overlap is also aligned to word boundaries.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > overlap >= 0.")

    chunks = []
    for page in pages:
        text = normalize_text(page["text"])
        if not text:
            continue

        start = 0
        page_chunk_index = 0
        while start < len(text):
            target_end = min(start + chunk_size, len(text))
            end = target_end

            if target_end < len(text):
                # Prefer sentence/paragraph punctuation, then whitespace.
                lower_bound = start + int(chunk_size * 0.60)
                candidates = [
                    text.rfind(". ", lower_bound, target_end),
                    text.rfind("? ", lower_bound, target_end),
                    text.rfind("! ", lower_bound, target_end),
                    text.rfind("; ", lower_bound, target_end),
                    text.rfind(" ", lower_bound, target_end),
                ]
                boundary = max(candidates)
                if boundary >= lower_bound:
                    end = boundary + (1 if text[boundary] == " " else 2)
                else:
                    end = _move_to_word_boundary(text, target_end, "backward")

            chunk_text = text[start:end].strip()
            if chunk_text:
                page_chunk_index += 1
                chunks.append({
                    "id": f'{page["source"]}-p{page["page"]}-c{page_chunk_index}',
                    "text": chunk_text,
                    "metadata": {
                        "source": str(page["source"]),
                        "page": int(page["page"]),
                    },
                })

            if end >= len(text):
                break

            next_start = max(start + 1, end - overlap)
            next_start = _move_to_word_boundary(text, next_start, "forward")
            if next_start <= start:
                next_start = end
            start = next_start

    return chunks
