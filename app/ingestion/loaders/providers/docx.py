from __future__ import annotations

import re
from typing import Any, Dict, List

from app.ingestion.loaders.base import BaseLoader, Element

_DOCX_AVAILABLE = True
try:
    import docx as _docx_lib
except ImportError:
    _DOCX_AVAILABLE = False


class DocxLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext == ".docx"

    def load(self, path: str) -> List[Element]:
        if not _DOCX_AVAILABLE:
            raise ImportError(
                "python-docx is not installed. Run: pip install python-docx"
            )
        doc = _docx_lib.Document(path)
        elements: List[Element] = []
        for para in doc.paragraphs:
            text = para.text
            if not text.strip():
                continue
            style_name = para.style.name if para.style else ""
            if style_name.startswith("Heading"):
                match = re.search(r"\d+", style_name)
                level = int(match.group()) if match else 1
                level = min(max(level, 1), 6)
                elem_type = f"heading_l{level}"
            elif "List" in style_name:
                elem_type = "list_item"
            else:
                elem_type = "paragraph"
            elements.append(Element(type=elem_type, text=text, page=None))
        return elements
