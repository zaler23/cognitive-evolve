#!/usr/bin/env python3
"""Path settings for the CognitiveEvolve source project and standalone runtime."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COGEV = ROOT / ".cogev"
TEMPLATES = COGEV / "templates"
SPECS = COGEV / "specs"
STANDALONE_RUNTIME_ROOT = Path(os.environ.get("COGEV_RUNTIME_ROOT", Path.home() / ".cognitive-evolve")).expanduser()
# Current variable name used throughout the runtime package.  It now
# means the standalone CognitiveEvolve runtime root, not a host-specific mirror.
LOCAL_RUNTIME_ROOT = STANDALONE_RUNTIME_ROOT
TASKS = Path(os.environ.get("COGEV_TASKS_ROOT", LOCAL_RUNTIME_ROOT / ".cogev" / "tasks")).expanduser()
CAPABILITY_REGISTRY = SPECS / "native-capabilities.json"
EXTENSION_PORTS_REGISTRY = SPECS / "extension-ports.json"
NATIVE_RUNTIME_SPEC = SPECS / "native-runtime.json"
NATIVE_EVAL_SUITE = SPECS / "native-eval-suite.json"

