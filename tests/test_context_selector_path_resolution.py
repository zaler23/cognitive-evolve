from __future__ import annotations

from cognitive_evolve_runtime.contracts.objective_contract import NexusProjectObjectiveContract
from cognitive_evolve_runtime.inputs.context_selector import ContextRequest, ContextSelector
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot


def test_context_selector_resolves_hallucinated_basename_to_real_source_and_imports(tmp_path) -> None:
    source = tmp_path / "cognitive_evolve_runtime" / "nexus" / "model_adapter.py"
    source.parent.mkdir(parents=True)
    source.write_text("class StructuredModelAdapter: pass\n", encoding="utf-8")
    test_file = tmp_path / "tests" / "test_nexus_model_adapter_schema_repair.py"
    test_file.parent.mkdir()
    test_file.write_text(
        "from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter\n\n"
        "def test_adapter():\n"
        "    assert StructuredModelAdapter\n",
        encoding="utf-8",
    )
    snapshot = ProjectSnapshot.from_path(tmp_path)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="repair nexus model adapter schema")

    packet = ContextSelector().build_context_packet(
        contract=NexusProjectObjectiveContract(original_user_goal="repair", normalized_goal="repair"),
        snapshot=snapshot,
        world=world,
        request=ContextRequest(
            need_files=["nexus_model_adapter_schema_repair.py"],
            need_tests=["test_nexus_model_adapter_schema_repair.py"],
        ),
    )

    assert "cognitive_evolve_runtime/nexus/model_adapter.py" in packet.coverage["selected_files"]
    assert "tests/test_nexus_model_adapter_schema_repair.py" in packet.coverage["selected_files"]
    assert "cognitive_evolve_runtime/nexus/model_adapter.py" in packet.raw_file_slices


def test_context_selector_keeps_imported_sources_for_each_requested_test(tmp_path) -> None:
    first_source = tmp_path / "cognitive_evolve_runtime" / "evidence" / "ledger.py"
    first_source.parent.mkdir(parents=True)
    first_source.write_text("class EvidenceLedger: pass\n", encoding="utf-8")
    second_source = tmp_path / "cognitive_evolve_runtime" / "api" / "models.py"
    second_source.parent.mkdir(parents=True)
    second_source.write_text("class ChatCompletionRequest: pass\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_active_evidence_obligation_ledger.py").write_text(
        "from cognitive_evolve_runtime.evidence.ledger import EvidenceLedger\n",
        encoding="utf-8",
    )
    (tests_dir / "test_architecture_boundaries.py").write_text(
        "from cognitive_evolve_runtime.api.models import ChatCompletionRequest\n",
        encoding="utf-8",
    )
    snapshot = ProjectSnapshot.from_path(tmp_path)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="active evidence obligation ledger architecture")

    packet = ContextSelector().build_context_packet(
        contract=NexusProjectObjectiveContract(original_user_goal="repair", normalized_goal="repair"),
        snapshot=snapshot,
        world=world,
        request=ContextRequest(
            need_tests=[
                "tests/test_active_evidence_obligation_ledger.py",
                "tests/test_architecture_boundaries.py",
            ],
        ),
    )

    assert "cognitive_evolve_runtime/evidence/ledger.py" in packet.raw_file_slices
    assert "cognitive_evolve_runtime/api/models.py" in packet.raw_file_slices
