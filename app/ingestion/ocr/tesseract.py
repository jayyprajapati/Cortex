"""pytesseract OCR provider."""
from __future__ import annotations
import io
from typing import List
from app.ingestion.loaders.base import Element
from app.ingestion.ocr.base import BaseOCRProvider


class TesseractOCRProvider(BaseOCRProvider):
    def ocr_page_bytes(self, image_bytes: bytes, page_num: int, language: str = "eng") -> List[Element]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("pytesseract and Pillow required: pip install pytesseract Pillow")

        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang=language)
        elements = []
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                elements.append(Element(type="paragraph", text=para, page=page_num))
        return elements
