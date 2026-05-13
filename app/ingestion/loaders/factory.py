from __future__ import annotations

from typing import Any, Dict

from app.ingestion.loaders.base import BaseLoader
from app.ingestion.loaders.providers.composite import CompositeLoader
from app.ingestion.loaders.providers.docling import DoclingLoader
from app.ingestion.loaders.providers.docx import DocxLoader
from app.ingestion.loaders.providers.markdown import MarkdownLoader
from app.ingestion.loaders.providers.pymupdf import PyMuPDFLoader
from app.ingestion.loaders.providers.unstructured import UnstructuredLoader

_DISPATCH: Dict[str, type] = {
    "pymupdf": PyMuPDFLoader,
    "docling": DoclingLoader,
    "unstructured": UnstructuredLoader,
    "docx": DocxLoader,
    "markdown": MarkdownLoader,
    "composite": CompositeLoader,
}


def get_loader(provider: str, options: Dict[str, Any]) -> BaseLoader:
    cls = _DISPATCH.get(provider)
    if cls is None:
        raise ValueError(f"Unknown loader provider: {provider!r}")
    return cls(options)
