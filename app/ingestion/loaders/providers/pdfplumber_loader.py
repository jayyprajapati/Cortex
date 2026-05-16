"""pdfplumber-based PDF loader: extracts tables as GFM markdown + text paragraphs."""
from __future__ import annotations

from typing import Any, Dict, List

from app.ingestion.loaders.base import BaseLoader, Element


def _table_to_gfm(table) -> str:
    """Convert a pdfplumber table (list of rows, each row is list of cells) to GFM markdown."""
    if not table or not table[0]:
        return ""
    rows = []
    for i, row in enumerate(table):
        cells = [str(c or "").replace("|", "\\|").strip() for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows)


class PdfPlumberLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext in (".pdf", ".PDF")

    def load(self, path: str) -> List[Element]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")

        elements: List[Element] = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables first
                tables = page.extract_tables()
                for table in tables:
                    md = _table_to_gfm(table)
                    if md.strip():
                        elements.append(Element(
                            type="table",
                            text=md,
                            page=page_num,
                        ))

                # Extract text (may overlap with table regions, but is best-effort)
                text = page.extract_text() or ""
                if text.strip():
                    for para in text.split("\n\n"):
                        para = para.strip()
                        if para:
                            elements.append(Element(
                                type="paragraph",
                                text=para,
                                page=page_num,
                            ))

        return elements
