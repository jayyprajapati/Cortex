from __future__ import annotations

from typing import Any, Dict, List

from app.ingestion.loaders.base import BaseLoader, Element


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
        result = converter.convert(path)
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
