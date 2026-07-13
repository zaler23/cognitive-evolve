from __future__ import annotations

import ast
from pathlib import Path

from cognitive_evolve_runtime.fabric.config import FabricRuntimeConfig

ROOT = Path(__file__).resolve().parents[1]
V3_FILES = [
    ROOT / "cognitive_evolve_runtime" / "fabric",
    ROOT / "cognitive_evolve_runtime" / "nexus" / "pool_preprocessing",
]
ALLOWED_ENV_FILES = {ROOT / "cognitive_evolve_runtime" / "fabric" / "config.py"}


def test_fabric_config_loads_connected_policy_metadata() -> None:
    policy = type("Policy", (), {"metadata": {"fabric_runtime": {"pool": {"cluster_similarity_threshold": 0.7}}}})()
    cfg = FabricRuntimeConfig.from_runtime_context(policy=policy)
    assert cfg.pool.cluster_similarity_threshold == 0.7
    assert cfg.config_hash
    assert "fabric_config_loaded_from_policy_metadata" in cfg.diagnostics


def test_fabric_env_reads_are_centralized_to_config_module() -> None:
    offenders: list[str] = []
    for root in V3_FILES:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("COGEV_FABRIC_"):
                    if path not in ALLOWED_ENV_FILES:
                        offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
