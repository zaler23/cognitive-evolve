"""Divergent operator registry."""
from __future__ import annotations

from typing import Any

from .archive_contrast import build_operator as _archive_contrast
from .assumption_inversion import build_operator as _assumption_inversion
from .conceptual_blend import build_operator as _conceptual_blend
from .constraint_extremes import build_operator as _constraint_extremes
from .first_principles import build_operator as _first_principles
from .hypothesis_gen import build_operator as _hypothesis_gen
from .lens_transplant import build_operator as _lens_transplant


def operator_registry() -> dict[str, Any]:
    operators = [_lens_transplant(), _conceptual_blend(), _assumption_inversion(), _first_principles(), _constraint_extremes(), _hypothesis_gen(), _archive_contrast()]
    return {op.operator_id: op for op in operators}


__all__ = ["operator_registry"]
