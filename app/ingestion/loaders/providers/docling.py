from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Dict, List

from app.ingestion.loaders.base import BaseLoader, Element

LOADER_TIMEOUT = int(os.getenv("LOADER_TIMEOUT", "60"))


class DoclingLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext in (".pdf",)

    def load(self, path: str) -> List[Element]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            from app.ingestion.loaders.providers.pymupdf import PyMuPDFLoader
            return PyMuPDFLoader(self.options).load(path)

        converter = DocumentConverter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(converter.convert, path)
            try:
                result = future.result(timeout=LOADER_TIMEOUT)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Docling conversion timed out after {LOADER_TIMEOUT}s"
                )

        elements: List[Element] = []
        for item, level in result.document.iterate_items():
            from docling.datamodel.document import TextItem, TableItem, SectionHeaderItem
            if isinstance(item, SectionHeaderItem):
                heading_type = f"heading_l{min(item.level, 6)}"
                elements.append(Element(type=heading_type, text=item.text, page=None))
            elif isinstance(item, TableItem):
                elements.append(Element(type="table", text=item.export_to_markdown(), page=None))
            elif isinstance(item, TextItem):
                elements.append(Element(type="paragraph", text=item.text, page=None))
        return elements
