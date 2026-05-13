from __future__ import annotations

import re
from typing import Any, Dict, List

from app.ingestion.loaders.base import BaseLoader, Element

_HEADING_RE = re.compile(r'^Heading (\d+)$')


class DocxLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext == ".docx"

    def load(self, path: str) -> List[Element]:
        try:
            import docx as _docx
        except ImportError:
            raise ImportError("python-docx is not installed. Run: pip install python-docx")
        doc = _docx.Document(path)
        elements: List[Element] = []
        for para in doc.paragraphs:
            text = para.text
            if not text.strip():
                continue
            style_name = para.style.name if para.style else ""
            m = _HEADING_RE.match(style_name)
            if m:
                level = int(m.group(1))
                elem_type = f"heading_l{min(level, 6)}"
            elif "List" in style_name:
                elem_type = "list_item"
            else:
                elem_type = "paragraph"
            elements.append(Element(type=elem_type, text=text, page=None))
        return elements
