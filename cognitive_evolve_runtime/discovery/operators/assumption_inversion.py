from .base import PromptOperator


def build_operator() -> PromptOperator:
    return PromptOperator("assumption_inversion", "invert_core_assumption", "Invert a central assumption and derive the smallest viable alternative mechanism.")
