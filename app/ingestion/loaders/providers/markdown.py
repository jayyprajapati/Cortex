from __future__ import annotations

from typing import Any, Dict, List

from app.ingestion.loaders.base import BaseLoader, Element

_MDI_AVAILABLE = True
try:
    from markdown_it import MarkdownIt as _MarkdownIt
except ImportError:
    _MDI_AVAILABLE = False

_HEADING_TAG_MAP = {
    "h1": "heading_l1",
    "h2": "heading_l2",
    "h3": "heading_l3",
    "h4": "heading_l4",
    "h5": "heading_l5",
    "h6": "heading_l6",
}


class MarkdownLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext in (".md", ".markdown")

    def load(self, path: str) -> List[Element]:
        if not _MDI_AVAILABLE:
            raise ImportError(
                "markdown-it-py is not installed. Run: pip install markdown-it-py"
            )
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        md = _MarkdownIt()
        tokens = md.parse(content)
        elements: List[Element] = []
        current_heading_type: str | None = None
        in_list_item = False
        in_fence = False

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "fence":
                if token.content.strip():
                    elements.append(Element(type="code_block", text=token.content, page=1))
                i += 1
                continue

            if token.type == "heading_open":
                tag = token.tag.lower()
                current_heading_type = _HEADING_TAG_MAP.get(tag, "heading_l1")
                i += 1
                continue

            if token.type == "heading_close":
                current_heading_type = None
                i += 1
                continue

            if token.type in ("bullet_list_open", "ordered_list_open"):
                in_list_item = True
                i += 1
                continue

            if token.type in ("bullet_list_close", "ordered_list_close"):
                in_list_item = False
                i += 1
                continue

            if token.type == "inline":
                text = token.content
                if not text.strip():
                    i += 1
                    continue
                if current_heading_type:
                    elem_type = current_heading_type
                elif in_list_item:
                    elem_type = "list_item"
                else:
                    elem_type = "paragraph"
                elements.append(Element(type=elem_type, text=text, page=1))
                i += 1
                continue

            i += 1

        return elements
