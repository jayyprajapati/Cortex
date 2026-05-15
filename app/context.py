from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from app.registry.models import ApplicationConfig
    from app.ingestion.loaders.base import BaseLoader
    from app.chunking.base import BaseChunker
    from app.embeddings.base import BaseEmbedder
    from app.vectorstore.base import BaseVectorStore
    from app.reranker.base import BaseReranker


GroundingMode = Literal["strict", "truthful", "off"]


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: Optional[float] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class EffectiveGenerationConfig:
    """Merged generation config: app-level defaults + optional task overrides."""

    system_prompt: str
    response_type: str  # "markdown" | "json"
    schema: Optional[Dict[str, Any]]
    temperature: float
    strict: bool
    max_retries: int
    grounding_mode: GroundingMode = "off"
    max_context_tokens: int = 4000
    voice_footer: Optional[str] = None


@dataclass
class ResolvedComponents:
    loader: "BaseLoader"
    chunker: "BaseChunker"
    embedder: "BaseEmbedder"
    reranker: Optional["BaseReranker"]
    vector_store: "BaseVectorStore"


@dataclass
class ExecutionContext:
    """
    Carries fully resolved configuration through every pipeline stage.
    Built once per request by app.registry.service.build_execution_context().
    Never constructed directly by pipeline code.
    """

    app_name: str
    user_id: str
    registry: "ApplicationConfig"
    llm_config: LLMConfig
    effective_generation: EffectiveGenerationConfig
    components: Optional[ResolvedComponents] = None   # populated by build_execution_context
    doc_ids: Optional[List[str]] = None
    task: Optional[str] = None
    prompt_override: Optional[str] = None
    request_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def collection(self) -> str:
        return self.registry.collection

    def get_override(self, key: str, default: Any = None) -> Any:
        return self.request_overrides.get(key, default)
