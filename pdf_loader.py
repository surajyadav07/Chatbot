from pathlib import Path
from pypdf import PdfReader

# Config for chunking
CHUNK_SIZE = 400      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks

def load_manual(pdf_path: str):
    """
    Read the Enertainer manual PDF and return a list of chunks.
    Each chunk is a dict: { "page": page_number, "text": text_chunk }
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Manual not found at {pdf_path}")

    reader = PdfReader(str(pdf_path))
    chunks = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").replace("\n", " ").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            chunks.append({"page": i + 1, "text": chunk})
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP

    return chunks
