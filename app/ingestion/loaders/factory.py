from __future__ import annotations

from typing import Any, Dict

from app.ingestion.loaders.base import BaseLoader


def get_loader(provider: str, options: Dict[str, Any]) -> BaseLoader:
    p = (provider or "composite").strip().lower()
    if p == "pymupdf":
        from app.ingestion.loaders.providers.pymupdf import PyMuPDFLoader
        return PyMuPDFLoader(options)
    elif p == "docling":
        from app.ingestion.loaders.providers.docling import DoclingLoader
        return DoclingLoader(options)
    elif p == "unstructured":
        from app.ingestion.loaders.providers.unstructured import UnstructuredLoader
        return UnstructuredLoader(options)
    elif p == "docx":
        from app.ingestion.loaders.providers.docx import DocxLoader
        return DocxLoader(options)
    elif p == "markdown":
        from app.ingestion.loaders.providers.markdown import MarkdownLoader
        return MarkdownLoader(options)
    elif p == "composite":
        from app.ingestion.loaders.providers.composite import CompositeLoader
        return CompositeLoader(options)
    else:
        raise ValueError(f"Unknown loader provider: {provider!r}")
