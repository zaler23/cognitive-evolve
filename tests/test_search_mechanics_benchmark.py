from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_search_mechanics_benchmark_emits_deterministic_invariants(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "bench" / "run_search_mechanics.py"
    spec = importlib.util.spec_from_file_location("search_mechanics", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.run(tmp_path / "bench")
    persisted = json.loads((tmp_path / "bench" / "search-mechanics.v1.json").read_text(encoding="utf-8"))

    assert persisted == report
    assert report["fanout"] == {"call_count": 3, "stable_order": True, "max_concurrent": 3}
    assert report["replay"]["physical_provider_calls"] == 1
    assert report["persistence"]["callback_returned_before_write"] is True
    assert report["islands"]["all_islands_covered"] is True
    assert report["selection_novelty_stop"]["selection_tiers"] == [2, 1, 0]
    assert report["selection_novelty_stop"]["near_novelty"] < report["selection_novelty_stop"]["far_novelty"]
    assert report["selection_novelty_stop"]["adaptive_stop_after_interventions"] is True
