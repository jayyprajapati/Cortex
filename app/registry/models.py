from __future__ import annotations

import re
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


class LoaderConfig(BaseModel):
    provider: Literal["docling", "unstructured", "pymupdf", "composite"] = "composite"
    provider_options: Dict[str, Any] = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    provider: str = "internal"
    strategy: str = "layout_aware_semantic"
    max_tokens: int = 512
    min_tokens: int = 128
    keep_tables_atomic: bool = True
    keep_code_atomic: bool = True
    provider_options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("max_tokens")
    @classmethod
    def _max_tokens(cls, v: int) -> int:
        if not 64 <= v <= 4096:
            raise ValueError("max_tokens must be between 64 and 4096")
        return v

    @field_validator("min_tokens")
    @classmethod
    def _min_tokens(cls, v: int) -> int:
        if not 10 <= v <= 512:
            raise ValueError("min_tokens must be between 10 and 512")
        return v

    @model_validator(mode="after")
    def _check_sizes(self) -> "ChunkingConfig":
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be less than max_tokens")
        return self


class EmbeddingConfig(BaseModel):
    provider: Literal["fastembed", "sentence_transformers", "openai", "cohere"] = "sentence_transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: Optional[str] = "prithivida/Splade_PP_en_v1"
    batch_size: int = 32
    normalize: bool = True
    dimension: Optional[int] = None
    provider_options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("embedding model name cannot be empty")
        return v

    @field_validator("batch_size")
    @classmethod
    def _batch_size(cls, v: int) -> int:
        if not 1 <= v <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        return v

    @field_validator("dimension")
    @classmethod
    def _dimension(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not 1 <= v <= 65536:
            raise ValueError("dimension must be between 1 and 65536")
        return v


class VectorStoreConfig(BaseModel):
    provider: Literal["qdrant", "pinecone", "weaviate", "chroma"] = "qdrant"
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    provider_options: Dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    top_k: int = 10
    fusion: Literal["rrf", "alpha"] = "rrf"
    alpha: float = 0.5
    query_rewrite: bool = True
    hyde: bool = False
    expand_neighbors: bool = True
    neighbor_budget_tokens: int = 400
    confidence_min_score: float = 0.25
    score_threshold: float = 0.0
    metadata_filter: Optional[Dict[str, Any]] = None

    @field_validator("top_k")
    @classmethod
    def _top_k(cls, v: int) -> int:
        if not 1 <= v <= 100:
            raise ValueError("top_k must be between 1 and 100")
        return v

    @field_validator("alpha")
    @classmethod
    def _alpha(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v


class RerankingConfig(BaseModel):
    enabled: bool = True
    provider: Literal["fastembed", "sentence_transformers", "cohere", "jina"] = "sentence_transformers"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 6
    candidate_cap: int = 30
    diversity: float = 0.0
    provider_options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("top_k")
    @classmethod
    def _top_k(cls, v: int) -> int:
        if not 1 <= v <= 100:
            raise ValueError("top_k must be between 1 and 100")
        return v

    @field_validator("candidate_cap")
    @classmethod
    def _candidate_cap(cls, v: int) -> int:
        if not 1 <= v <= 500:
            raise ValueError("candidate_cap must be between 1 and 500")
        return v

    @field_validator("diversity")
    @classmethod
    def _diversity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("diversity must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def _top_k_lte_cap(self) -> "RerankingConfig":
        if self.enabled and self.top_k > self.candidate_cap:
            raise ValueError("top_k must be <= candidate_cap when reranking is enabled")
        return self


class GenerationConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    response_type: Literal["markdown", "json", "text"] = "markdown"
    output_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    temperature: float = 0.1
    strict: bool = False
    max_retries: int = 2
    grounding_mode: Literal["strict", "truthful", "off"] = "off"
    max_context_tokens: int = 4000

    @field_validator("temperature")
    @classmethod
    def _temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_retries")
    @classmethod
    def _max_retries(cls, v: int) -> int:
        if not 0 <= v <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        return v


class ConversationConfig(BaseModel):
    clarification_policy: Literal["aggressive", "balanced", "never"] = "balanced"
    max_history_turns: int = 10
    summary_threshold: int = 12
    use_query_analyzer: bool = True
    analyzer_model_override: Optional[str] = None


class ApplicationDefaults(BaseModel):
    system_prompt: str

    @field_validator("system_prompt")
    @classmethod
    def _system_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("system_prompt cannot be empty")
        return v


class TaskOverride(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    system_prompt: Optional[str] = None
    response_type: Optional[Literal["markdown", "json", "text"]] = None
    output_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    temperature: Optional[float] = None
    strict: Optional[bool] = None
    max_retries: Optional[int] = None
    grounding_mode: Optional[Literal["strict", "truthful", "off"]] = None
    max_context_tokens: Optional[int] = None
    voice_footer: Optional[str] = None

    @field_validator("temperature")
    @classmethod
    def _temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    app_name: str
    collection: str
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    defaults: ApplicationDefaults
    tasks: Dict[str, TaskOverride] = Field(default_factory=dict)
    default_task: Optional[str] = None

    @field_validator("app_name")
    @classmethod
    def _app_name(cls, v: str) -> str:
        v = v.strip()
        if not _NAME_RE.match(v):
            raise ValueError(
                "app_name must start with a lowercase letter, contain only "
                "[a-z0-9_-], and be at most 63 characters"
            )
        return v

    @field_validator("collection")
    @classmethod
    def _collection(cls, v: str) -> str:
        v = v.strip()
        if not _NAME_RE.match(v):
            raise ValueError(
                "collection must start with a lowercase letter, contain only "
                "[a-z0-9_-], and be at most 63 characters"
            )
        return v

    @model_validator(mode="after")
    def _default_task_exists(self) -> "ApplicationConfig":
        if self.default_task is not None and self.default_task not in self.tasks:
            raise ValueError(
                f"default_task '{self.default_task}' must exist in tasks"
            )
        if self.tasks and self.default_task is None:
            raise ValueError("default_task is required when tasks are defined")
        return self


# Backward compat alias — remove in a future cleanup
IngestionConfig = ChunkingConfig
