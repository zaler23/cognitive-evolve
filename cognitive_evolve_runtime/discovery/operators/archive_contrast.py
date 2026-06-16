from .base import PromptOperator


def build_operator() -> PromptOperator:
    return PromptOperator("archive_contrast", "empty_cell_contrast", "Contrast the parent with empty or failed archive cells and target the least explored descriptor.")
