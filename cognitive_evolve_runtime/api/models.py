from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = ""
    name: str | None = None



class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = None
    stream: bool = False
    max_tokens: int | None = None
    metadata: dict[str, Any] | None = None

    model_config = {"extra": "allow"}



__all__ = ['ChatMessage', 'ChatCompletionRequest']
