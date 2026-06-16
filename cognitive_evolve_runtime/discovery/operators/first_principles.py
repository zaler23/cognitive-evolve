from .base import PromptOperator


def build_operator() -> PromptOperator:
    return PromptOperator("first_principles", "primitive_constraints", "Rebuild the candidate from primitive constraints instead of inherited patterns.")
