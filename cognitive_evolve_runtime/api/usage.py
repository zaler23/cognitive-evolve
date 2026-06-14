from __future__ import annotations


def _count_tokens(text: str, *, model: str | None = None) -> int:
    try:
        from litellm import token_counter

        counted = token_counter(model=model or "gpt-4o-mini", text=text)
        return max(1, int(counted))
    except Exception:
        # Safe local approximation when LiteLLM's token counter is unavailable.
        return max(1, len(text) // 4)



def _usage(prompt: str, answer: str, *, model: str | None = None) -> dict[str, int]:
    # OpenAI-compatible clients expect a usage object. Upstream authoritative
    # usage is still recorded in evaluations/llm-runtime-report.json; this
    # response-local surface now uses LiteLLM's token counter when available.
    prompt_tokens = _count_tokens(prompt, model=model)
    completion_tokens = _count_tokens(answer, model=model)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


__all__ = ['_count_tokens', '_usage']
