"""
Utility functions for extracting text from uploaded documents.

Supported formats:
- PDF
- TXT
- CSV
"""

from pathlib import Path
import pandas as pd
import pdfplumber

"""Extract text from a file using PyMuPDF."""
def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return _read_txt(file_path)
    if suffix == ".csv":
        return _read_csv(file_path)
    if suffix == ".pdf":
        return _read_pdf(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _read_txt(file_path: Path) -> str:
    # Try utf-8 first, fall back to latin-1 for odd encodings.
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1")


def _read_csv(file_path: Path) -> str:
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False, on_bad_lines="skip")
    # Flatten the dataframe into a text blob: header + row-by-row "col: val" pairs.
    lines = [", ".join(df.columns.astype(str))]
    for _, row in df.iterrows():
        lines.append(" | ".join(f"{col}: {val}" for col, val in row.items()))
    return "\n".join(lines)


def _read_pdf(file_path: Path) -> str:
    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)