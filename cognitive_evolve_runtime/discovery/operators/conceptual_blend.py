from .base import PromptOperator


def build_operator() -> PromptOperator:
    return PromptOperator("conceptual_blend", "archive_elite_blend", "Blend two remote archive elites while preserving falsifiable mechanism boundaries.")
