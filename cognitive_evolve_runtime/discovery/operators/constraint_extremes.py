from .base import PromptOperator


def build_operator() -> PromptOperator:
    return PromptOperator("constraint_extremes", "limit_case", "Push one constraint to an extreme limit and explore the resulting behavior cell.")
