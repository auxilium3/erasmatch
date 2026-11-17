"""Loading catalogs from files and text from PDFs."""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from erasmatch.domain import Catalog


def load_catalog(path: Path) -> Catalog:
    """Load a catalog from a .yaml/.yml or .json file (same structure for both)."""
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    elif suffix == ".json":
        data = json.loads(raw)
    else:
        raise ValueError(f"Unsupported catalog format: {path.name} (use .yaml or .json)")
    return Catalog.model_validate(data)


def pdf_to_text(data: bytes) -> str:
    """Extract text from a PDF. Scanned PDFs without a text layer return an empty string."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def file_to_text(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        return pdf_to_text(data)
    return data.decode("utf-8", errors="replace")
