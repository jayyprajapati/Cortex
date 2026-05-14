from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from cortex.schemas.resumelab import LLMOverride


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    app_name: str
    user_id: str
    query: str
    thread_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    task: Optional[str] = None
    llm: Optional[LLMOverride] = None
    prompt_override: Optional[str] = None


class Citation(BaseModel):
    index: int
    section: Optional[str] = None
    page: Optional[Any] = None
    text: Optional[str] = None


class ChatMessage(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: Optional[List[Citation]] = None
    grounded: Optional[bool] = None
    created_at: int


class ChatResponse(BaseModel):
    thread_id: str
    answer: Any
    grounded: bool
    citations: List[Citation] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ThreadSummary(BaseModel):
    id: str
    app_name: str
    user_id: str
    doc_ids: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    message_count: int = 0
    created_at: int
    updated_at: int


class ThreadListResponse(BaseModel):
    threads: List[ThreadSummary]


class ThreadDetailResponse(BaseModel):
    id: str
    app_name: str
    user_id: str
    doc_ids: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    summary: Optional[str] = None
    created_at: int
    updated_at: int
    messages: List[ChatMessage] = Field(default_factory=list)


class ThreadPatchRequest(BaseModel):
    title: Optional[str] = None
