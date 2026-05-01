from __future__ import annotations

import re
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


class IngestionConfig(BaseModel):
    strategy: Literal["semantic_doc", "resume_structured", "markdown_aware"] = "semantic_doc"
    max_tokens: int = 512
    min_tokens: int = 50
    overlap_tokens: int = 64
    semantic_split: bool = True

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

    @field_validator("overlap_tokens")
    @classmethod
    def _overlap_tokens(cls, v: int) -> int:
        if not 0 <= v <= 256:
            raise ValueError("overlap_tokens must be between 0 and 256")
        return v

    @model_validator(mode="after")
    def _check_sizes(self) -> IngestionConfig:
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be less than max_tokens")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
        return self


class EmbeddingConfig(BaseModel):
    model: str
    batch_size: int = 32
    normalize: bool = True
    dimension: Optional[int] = None

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


class RetrievalConfig(BaseModel):
    top_k: int = 10
    hybrid: bool = True
    alpha: float = 0.7
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
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    candidate_cap: int = 20

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

    @model_validator(mode="after")
    def _top_k_lte_cap(self) -> RerankingConfig:
        if self.enabled and self.top_k > self.candidate_cap:
            raise ValueError("top_k must be <= candidate_cap when reranking is enabled")
        return self


class GenerationConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    response_type: Literal["markdown", "json"] = "markdown"
    output_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    temperature: float = 0.1
    strict: bool = False
    max_retries: int = 2

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

    @model_validator(mode="after")
    def _strict_needs_schema(self) -> GenerationConfig:
        # Only enforce at runtime (generate_pipeline) — task overrides may supply the schema.
        # So no static rejection here; strict is validated contextually.
        return self


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
    """Per-task overrides applied on top of app-level generation config."""

    model_config = ConfigDict(populate_by_name=True)

    system_prompt: Optional[str] = None
    response_type: Optional[Literal["markdown", "json"]] = None
    output_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    temperature: Optional[float] = None
    strict: Optional[bool] = None
    max_retries: Optional[int] = None

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
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    generation: GenerationConfig
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
    def _default_task_exists(self) -> ApplicationConfig:
        if self.default_task is not None and self.default_task not in self.tasks:
            raise ValueError(
                f"default_task '{self.default_task}' must exist in tasks"
            )
        if self.tasks and self.default_task is None:
            raise ValueError("default_task is required when tasks are defined")
        return self
