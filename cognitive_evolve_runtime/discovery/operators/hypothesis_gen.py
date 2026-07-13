from .base import PromptOperator


def build_operator() -> PromptOperator:
    return PromptOperator("hypothesis_gen", "testable_hypothesis", "Generate a new hypothesis plus the falsification test that would kill it.")
