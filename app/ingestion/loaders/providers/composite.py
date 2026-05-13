from __future__ import annotations

import os
from typing import Any, Dict, List

from app.ingestion.loaders.base import BaseLoader, Element
from app.ingestion.loaders.providers.docling import DoclingLoader
from app.ingestion.loaders.providers.docx import DocxLoader
from app.ingestion.loaders.providers.markdown import MarkdownLoader
from app.ingestion.loaders.providers.pymupdf import PyMuPDFLoader
from app.ingestion.loaders.providers.unstructured import UnstructuredLoader

_SUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".markdown", ".txt"}
)


class CompositeLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self._pdf_loader = DoclingLoader(options)
        self._pdf_fallback = PyMuPDFLoader(options)
        self._docx_loader = DocxLoader(options)
        self._docx_fallback = UnstructuredLoader(options)
        self._html_loader = UnstructuredLoader(options)
        self._md_loader = MarkdownLoader(options)

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext in _SUPPORTED_EXTENSIONS

    def load(self, path: str) -> List[Element]:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            return self._pdf_loader.load(path)

        if ext in (".docx", ".doc"):
            try:
                return self._docx_loader.load(path)
            except ImportError:
                return self._docx_fallback.load(path)

        if ext in (".html", ".htm"):
            return self._html_loader.load(path)

        if ext in (".md", ".markdown"):
            return self._md_loader.load(path)

        if ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            paragraphs = raw.split("\n\n")
            return [
                Element(type="paragraph", text=para.strip(), page=1)
                for para in paragraphs
                if para.strip()
            ]

        raise ValueError(f"Unsupported file extension: {ext!r}")
