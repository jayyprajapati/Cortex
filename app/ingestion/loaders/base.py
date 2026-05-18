from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Element:
    type: str
    text: str
    page: Optional[int] = None
    bbox: Optional[tuple] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_heading: Optional[str] = None


class BaseLoader(ABC):
    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return False

    @abstractmethod
    def load(self, path: str) -> List[Element]:
        ...
