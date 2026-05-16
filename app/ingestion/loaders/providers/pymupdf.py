from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from app.ingestion.loaders.base import BaseLoader, Element


class PyMuPDFLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext in (".pdf", ".PDF")

    def load(self, path: str) -> List[Element]:
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF is not installed. Run: pip install PyMuPDF")
        doc = fitz.open(path)
        total_pages = len(doc)
        strip_hf = bool(self.options.get("strip_headers_footers", False))

        # ---------------------------------------------------------------------------
        # P2.5: First pass — collect page dimensions and candidate header/footer text
        # for documents that opt into stripping.
        # ---------------------------------------------------------------------------
        hf_texts: set[str] = set()
        if strip_hf and total_pages > 0:
            # text -> set of page indices on which it appears
            text_pages: dict[str, set[int]] = defaultdict(set)
            for page_num, page in enumerate(doc):
                page_h = page.rect.height
                if page_h <= 0:
                    continue
                margin_top = page_h * 0.05
                margin_bottom = page_h * 0.95
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] != 0:
                        continue
                    y0 = block["bbox"][1]
                    y1 = block["bbox"][3]
                    # Block is in the top or bottom 5% of the page
                    if y1 <= margin_top or y0 >= margin_bottom:
                        for line in block["lines"]:
                            text = "".join(
                                span["text"] for span in line["spans"]
                            ).strip()
                            if text:
                                text_pages[text].add(page_num)

            # Any text appearing on ≥80% of pages is a header/footer
            threshold = max(1, total_pages * 0.80)
            hf_texts = {t for t, pages in text_pages.items() if len(pages) >= threshold}

        # ---------------------------------------------------------------------------
        # Second pass — build elements
        # ---------------------------------------------------------------------------
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

                        # P2.5: skip header/footer lines
                        if strip_hf and text.strip() in hf_texts:
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
                    # P2.1: fitz type 1 is an image block, not a table.
                    # Emit as image element with empty text; chunkers skip empty-text elements.
                    elements.append(Element(
                        type="image",
                        text="",
                        page=page_num + 1,
                        bbox=tuple(block["bbox"]),
                    ))

        # P2.7: Extract AcroForm fields if option is enabled
        if self.options.get("extract_form_fields", False):
            form_elements = self._extract_form_fields(doc)
            elements.extend(form_elements)

        doc.close()
        return elements

    def _extract_form_fields(self, doc) -> List[Element]:
        """Extract AcroForm field values from a PDF."""
        elements: List[Element] = []
        try:
            for page in doc:
                for widget in page.widgets() or []:
                    label = (widget.field_name or "").strip()
                    value = str(widget.field_value or "").strip()
                    if label and value:
                        elements.append(Element(
                            type="form_field",
                            text=f"{label}: {value}",
                            page=page.number + 1,
                        ))
        except Exception:
            pass  # AcroForm not present or extraction failed
        return elements
