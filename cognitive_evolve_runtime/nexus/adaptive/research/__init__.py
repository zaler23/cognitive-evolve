"""Adaptive-internal research extension registry.

This package is deliberately nested under ``nexus.adaptive`` to avoid a second
runtime/control plane.  Extensions emit ResearchSignal objects; the existing
AdaptiveRuntimeController remains the only entry point.
"""
from .manager import ResearchExtensionRegistry
from .protocol import NoOpResearchExtension, ResearchContext, ResearchExtension
from .registry import ResearchConfig
from .signal import ResearchSignal, merge_research_signals
from .state import RESEARCH_STATE_VERSION, ResearchRegistryState

__all__ = [
    "NoOpResearchExtension",
    "RESEARCH_STATE_VERSION",
    "ResearchConfig",
    "ResearchContext",
    "ResearchExtension",
    "ResearchExtensionRegistry",
    "ResearchRegistryState",
    "ResearchSignal",
    "merge_research_signals",
]
