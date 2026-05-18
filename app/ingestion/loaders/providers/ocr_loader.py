"""OCR-based PDF loader for scanned documents."""
from __future__ import annotations
import os
from typing import Any, Dict, List
from app.ingestion.loaders.base import BaseLoader, Element

TEXT_THRESHOLD = int(os.getenv("OCR_TEXT_THRESHOLD", "50"))  # chars per page


class OCRLoader(BaseLoader):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self.language = options.get("ocr_language", "eng")

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext in (".pdf", ".PDF")

    def load(self, path: str) -> List[Element]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required for OCR loader")

        doc = fitz.open(path)
        total_text = sum(len(page.get_text()) for page in doc)
        page_count = len(doc)

        if page_count == 0:
            doc.close()
            return []

        avg_chars_per_page = total_text / page_count
        if avg_chars_per_page >= TEXT_THRESHOLD:
            # Not scanned — don't use OCR
            doc.close()
            return []

        # Scanned PDF — render each page to image and OCR
        from app.ingestion.ocr.tesseract import TesseractOCRProvider
        provider = TesseractOCRProvider()
        elements = []

        for page_num, page in enumerate(doc, 1):
            mat = fitz.Matrix(2, 2)  # 2x scale for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            page_elements = provider.ocr_page_bytes(img_bytes, page_num, self.language)
            elements.extend(page_elements)

        doc.close()
        return elements
