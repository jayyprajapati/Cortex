"""OCR provider abstraction."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from app.ingestion.loaders.base import Element


class BaseOCRProvider(ABC):
    @abstractmethod
    def ocr_page_bytes(self, image_bytes: bytes, page_num: int, language: str = "eng") -> List[Element]:
        """Run OCR on a rendered page image and return paragraph elements."""
