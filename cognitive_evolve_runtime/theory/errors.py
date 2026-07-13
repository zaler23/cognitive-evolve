"""Theory-layer local errors.

These errors are caught inside the advisory theory layer.  They must never
propagate into Nexus control flow.
"""
from __future__ import annotations


class TheoryProducerError(RuntimeError):
    """A theory producer failed to emit valid advisory features."""


class TheoryTimeout(TimeoutError):
    """A theory producer exceeded its configured advisory budget."""


class TheoryCancelled(RuntimeError):
    """The advisory theory layer was cancelled by its caller."""


__all__ = ["TheoryCancelled", "TheoryProducerError", "TheoryTimeout"]
