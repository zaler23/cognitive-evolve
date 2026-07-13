from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusProjectObjectiveContract
from cognitive_evolve_runtime.inputs.context_selector import ContextRequest, ContextSelector
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.nexus.context_protocol import ContextOrchestrator


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


def test_fallback_context_prioritizes_action_sources_and_filters_missing_parent_paths(tmp_path) -> None:
    action_paths = [
        "cognitive_evolve_runtime/nexus/loop/seeding.py",
        "cognitive_evolve_runtime/nexus/search_kernel/harvesting.py",
        "cognitive_evolve_runtime/llm/fanout.py",
        "cognitive_evolve_runtime/nexus/loop/offspring.py",
        "cognitive_evolve_runtime/nexus/reproduction.py",
        "cognitive_evolve_runtime/archives/failure.py",
    ]
    parent_paths = [
        "cognitive_evolve_runtime/nexus/prompt_view.py",
        "tests/test_missing_one.py",
        "cognitive_evolve_runtime/nexus/model_adapter_core.py",
        "tests/test_missing_two.py",
        "tests/test_missing_three.py",
        "cognitive_evolve_runtime/nexus/loop/round.py",
    ]
    for index, relative in enumerate([*action_paths, *[path for path in parent_paths if not path.startswith("tests/")]]):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"SOURCE_{index} = True\n", encoding="utf-8")
    snapshot = ProjectSnapshot.from_path(tmp_path)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="improve evolution efficiency")
    contract = NexusProjectObjectiveContract(original_user_goal="improve", normalized_goal="improve")
    parent = ProjectCandidateGenome(id="parent", touched_files=parent_paths)
    instruction = "; then inspect ".join(path.rsplit("/", 1)[-1] for path in action_paths)

    first = ContextOrchestrator().build_for_parents(
        contract=contract,
        snapshot=snapshot,
        world=world,
        parents=[parent],
        archives=ArchiveManager(),
        mutation_instruction=instruction,
    )
    replay = ContextOrchestrator().build_for_parents(
        contract=contract,
        snapshot=snapshot,
        world=world,
        parents=[parent],
        archives=ArchiveManager(),
        mutation_instruction=instruction,
    )
    focused = ContextOrchestrator().build_for_parents(
        contract=contract,
        snapshot=snapshot,
        world=world,
        parents=[parent],
        archives=ArchiveManager(),
        mutation_instruction="inspect seeding.py then harvesting.py",
    )

    assert first.requests[0].need_files == action_paths
    assert list(first.packets[0].raw_file_slices) == action_paths
    assert first.to_source_context() == replay.to_source_context()
    assert focused.requests[0].need_files == [
        *action_paths[:2],
        "cognitive_evolve_runtime/nexus/prompt_view.py",
        "cognitive_evolve_runtime/nexus/model_adapter_core.py",
        "cognitive_evolve_runtime/nexus/loop/round.py",
    ]
    assert focused.to_source_context() != first.to_source_context()


def test_fallback_context_filters_missing_paths_before_parent_file_limit(tmp_path) -> None:
    existing = [f"pkg/source_{index}.py" for index in range(6)]
    for relative in existing:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"PATH = {relative!r}\n", encoding="utf-8")
    proposed = ["tests/test_missing_one.py", "tests/test_missing_two.py", "tests/test_missing_three.py"]
    snapshot = ProjectSnapshot.from_path(tmp_path)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="repair sources")
    parent = ProjectCandidateGenome(id="parent", touched_files=[existing[0], *proposed, *existing[1:]])

    result = ContextOrchestrator().build_for_parents(
        contract=NexusProjectObjectiveContract(original_user_goal="repair", normalized_goal="repair"),
        snapshot=snapshot,
        world=world,
        parents=[parent],
        archives=ArchiveManager(),
    )

    assert result.requests[0].need_files == existing[:5]
    assert list(result.packets[0].raw_file_slices) == existing[:5]
