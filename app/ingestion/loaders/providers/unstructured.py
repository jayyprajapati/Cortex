from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.ingestion.loaders.base import BaseLoader, Element

_UNSTRUCTURED_AVAILABLE = True
try:
    from unstructured.partition.auto import partition as _partition
except ImportError:
    _UNSTRUCTURED_AVAILABLE = False


_TYPE_MAP = {
    "Title": "heading_l1",
    "NarrativeText": "paragraph",
    "Text": "paragraph",
    "ListItem": "list_item",
    "Table": "table",
    "CodeSnippet": "code_block",
}


class UnstructuredLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext in (".docx", ".html", ".htm", ".txt")

    def load(self, path: str) -> List[Element]:
        if not _UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "unstructured is not installed. Run: pip install unstructured[local-inference]"
            )
        raw_elements = _partition(filename=path)
        elements: List[Element] = []
        for el in raw_elements:
            type_name = type(el).__name__
            elem_type = _TYPE_MAP.get(type_name, "paragraph")
            page: Optional[int] = None
            try:
                page = el.metadata.page_number
            except AttributeError:
                pass
            text = str(el)
            if text.strip():
                elements.append(Element(type=elem_type, text=text, page=page))
        return elements
