#!/usr/bin/env python3
"""Path settings for the CognitiveEvolve source project and standalone runtime."""
from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COGEV = ROOT / ".cogev"
_RESOURCES = files("cognitive_evolve_runtime").joinpath("resources")
TEMPLATES = _RESOURCES.joinpath("templates")
SPECS = _RESOURCES.joinpath("specs")
STANDALONE_RUNTIME_ROOT = Path(os.environ.get("COGEV_RUNTIME_ROOT", Path.home() / ".cognitive-evolve")).expanduser()
# Runtime-wide alias for the standalone CognitiveEvolve state root.
LOCAL_RUNTIME_ROOT = STANDALONE_RUNTIME_ROOT
TASKS = Path(os.environ.get("COGEV_TASKS_ROOT", LOCAL_RUNTIME_ROOT / ".cogev" / "tasks")).expanduser()
CAPABILITY_REGISTRY = SPECS / "native-capabilities.json"
EXTENSION_PORTS_REGISTRY = SPECS / "extension-ports.json"
NATIVE_RUNTIME_SPEC = SPECS / "native-runtime.json"
NATIVE_EVAL_SUITE = SPECS / "native-eval-suite.json"
