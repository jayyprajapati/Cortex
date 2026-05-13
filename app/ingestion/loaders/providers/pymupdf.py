from __future__ import annotations

from typing import Any, Dict, List

import fitz

from app.ingestion.loaders.base import BaseLoader, Element


class PyMuPDFLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext in (".pdf", ".PDF")

    def load(self, path: str) -> List[Element]:
        doc = fitz.open(path)
        elements: List[Element] = []

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:
                    for line in block["lines"]:
                        spans = line["spans"]
                        if not spans:
                            continue
                        text = "".join(span["text"] for span in spans)
                        if not text.strip():
                            continue
                        size = spans[0]["size"]
                        flags = spans[0]["flags"]
                        is_bold = bool(flags & 16)
                        is_large = size > 13
                        is_short = len(text) <= 80
                        no_terminal_punct = not text.rstrip().endswith((".", ",", ";", ":", "!", "?"))
                        if (is_large or is_bold) and is_short and no_terminal_punct:
                            if size > 16:
                                elem_type = "heading_l1"
                            elif size > 14:
                                elem_type = "heading_l2"
                            else:
                                elem_type = "heading_l3"
                        else:
                            elem_type = "paragraph"
                        elements.append(Element(
                            type=elem_type,
                            text=text,
                            page=page_num + 1,
                            bbox=tuple(block["bbox"]),
                        ))
                elif block["type"] == 1:
                    elements.append(Element(
                        type="table",
                        text="[table]",
                        page=page_num + 1,
                    ))

        doc.close()
        return elements
