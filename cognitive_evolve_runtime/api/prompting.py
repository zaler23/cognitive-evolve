from __future__ import annotations

from typing import Any

from .models import ChatMessage


def _message_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif "text" in item:
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part).strip()



def build_one_shot_prompt(messages: list[ChatMessage]) -> str:
    """Convert chat history into one seed prompt without reopening interaction."""
    if not messages:
        return ""
    user_messages = [msg for msg in messages if msg.role == "user"]
    latest_user = _message_text(user_messages[-1].content) if user_messages else _message_text(messages[-1].content)
    prior: list[str] = []
    for msg in messages[:-1]:
        text = _message_text(msg.content)
        if not text:
            continue
        if msg.role == "system":
            prior.append(f"[system context]\n{text}")
        elif msg.role in {"user", "assistant"}:
            prior.append(f"[{msg.role} context]\n{text}")
    if prior:
        return (
            "Treat this frontend chat history as one CognitiveEvolve one-shot request. "
            "Do not ask for mid-turn clarification; infer, evolve, verify, and return the final answer.\n\n"
            + "\n\n".join(prior)
            + "\n\n[current user request]\n"
            + latest_user
        ).strip()
    return latest_user.strip()


__all__ = ['build_one_shot_prompt']
