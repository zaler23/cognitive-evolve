from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.nexus.runtime import NexusRuntime


def test_serialize_state_emits_fresh_event_signature_and_roundtrips_extra(tmp_path: Path) -> None:
    runtime = NexusRuntime(output_dir=tmp_path)

    first = runtime.serialize_state(extra={"round": 1, "candidate": "C1"})
    second = runtime.serialize_state(extra={"round": 1, "candidate": "C1"})

    assert first["status"] == "success"
    assert first["runtime_path"] == "nexus"
    assert first["signature"] == first["serialization_signature"]
    assert second["signature"] == second["serialization_signature"]
    assert first["signature"].startswith("NEXUS-RUNTIME-STATE-SIG-v2-")
    assert second["signature"].startswith("NEXUS-RUNTIME-STATE-SIG-v2-")
    assert first["signature"] != second["signature"]
    assert first["signature_semantics"] == "fresh_serialization_event_not_deterministic_state_hash"
    assert first["state"]["extra"] == {"round": 1, "candidate": "C1"}
    assert first["state"]["output_dir"] == str(tmp_path)


def test_serialize_state_is_not_a_deterministic_content_hash() -> None:
    runtime = NexusRuntime()

    signatures = {runtime.serialize_state()["signature"] for _ in range(3)}

    assert len(signatures) == 3
