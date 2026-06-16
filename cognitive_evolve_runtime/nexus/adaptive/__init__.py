"""Adaptive evidence-layer public boundary."""
from .config import AdaptiveConfig, SpatialAdaptiveConfig
from .controller import AdaptiveRuntimeController
from .elite_gate import apply_final_certificate_to_closure, apply_research_final_gate_directives, build_final_certificate
from .state import ADAPTIVE_STATE_VERSION, AdaptiveRuntimeState
from .telemetry import adaptive_event

__all__ = [
    "ADAPTIVE_STATE_VERSION",
    "AdaptiveConfig",
    "AdaptiveRuntimeController",
    "AdaptiveRuntimeState",
    "SpatialAdaptiveConfig",
    "apply_final_certificate_to_closure",
    "apply_research_final_gate_directives",
    "adaptive_event",
    "build_final_certificate",
]
