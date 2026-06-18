from __future__ import annotations

import json

from cognitive_evolve_runtime.nexus.model_adapter_core import StructuredModelAdapterCore
from cognitive_evolve_runtime.nexus.prompt_profiles import apply_prompt_profile
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view


def test_verification_regime_omits_strength_shortcuts() -> None:
    payload = {
        "candidates": [{"id": "c", "artifact": "x"}],
        "contract": {"normalized_goal": "g"},
        "_prompt_context_controls": {
            "verification_regime": [
                {"id": "obl", "must_pass": True, "strength_contribution": 4, "replayable": True, "exogeneity_probe": {"content": "p"}}
            ]
        },
    }
    view = build_prompt_view("nexus_critique_candidates", payload, max_chars=5000)
    text = json.dumps(view.payload, ensure_ascii=False, sort_keys=True)
    assert "verification_regime" in view.payload
    assert "strength_contribution" not in text
    assert "measured_strength" not in text


def test_offspring_prompt_profile_prunes_full_population() -> None:
    payload = {"population": [{"id": str(i), "artifact": "x" * 1000} for i in range(30)], "plans": [{"parent_ids": ["p"]}], "parents": [{"id": "p"}], "contract": {"normalized_goal": "g"}}
    profiled, meta = apply_prompt_profile("nexus_generate_offspring", payload)
    assert meta["profile_applied"] is True
    assert "population" not in profiled
    assert meta["payload_chars_after_profile"] < 12000


def test_prompt_audit_writes_runtime_jsonl(tmp_path) -> None:
    audit_path = tmp_path / "prompt-audit.jsonl"

    def caller(_request_type, _payload, _schema):
        return {"ok": True}

    adapter = StructuredModelAdapterCore(caller=caller, metadata={"prompt_audit_path": str(audit_path)})
    result = adapter._call("unit_test", {"value": 1}, {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    assert result["ok"] is True
    line = json.loads(audit_path.read_text(encoding="utf-8").strip())
    assert line["request_type"] == "unit_test"
    assert line["forbidden_strength_shortcuts_present"] is False
