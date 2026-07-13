from __future__ import annotations

import json
import sys
from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationPlan
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.evaluators import EvaluatorSpec, ExternalEvaluatorRunner
from cognitive_evolve_runtime.evaluators.progressive import _repair_hints
from cognitive_evolve_runtime.evaluators.runner import _parse_evaluator_output
from cognitive_evolve_runtime.fabric.config import PreprocessConfig
from cognitive_evolve_runtime.inputs.context_selector import ContextPacket, ContextRequest, ContextSelector
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket, TextWorldModel
from cognitive_evolve_runtime.llm.env import llm_prompt_char_limit
from cognitive_evolve_runtime.llm.request_policy import LLMRequestPolicy
from cognitive_evolve_runtime.llm.transport import max_tokens_for_request
from cognitive_evolve_runtime.nexus.context_protocol import ContextProtocolResult
from cognitive_evolve_runtime.nexus.final_projection import _render_artifact
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring, _policy_for_generation_batch
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.nextgen import family_signature, transition_signature
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.prompt_view import _select_candidates_for_prompt, build_prompt_view, candidate_prompt_view, contract_prompt_view, history_prompt_view, policy_prompt_view, prompt_char_budget, prompt_char_budget_details, world_prompt_view
from cognitive_evolve_runtime.nexus.runtime import _build_text_world_model
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_semantic_signature
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import plan_signature
from cognitive_evolve_runtime.nexus.search_space import build_search_space_map, classify_candidate
from cognitive_evolve_runtime.nexus.semantics import assess
from cognitive_evolve_runtime.nexus.synthesis import synthesize_result
from cognitive_evolve_runtime.theory.representations import CandidateRepresentation
from cognitive_evolve_runtime.verification.reformulation import reformulate_for_verification


def test_external_evaluator_preserves_and_parses_long_structured_output(tmp_path: Path) -> None:
    evaluator = tmp_path / "long_evaluator.py"
    evaluator.write_text(
        "import json, sys\n"
        "json.load(open(sys.argv[1]))\n"
        "print(json.dumps({'passed': True, 'metrics': {'score': 0.73, 'raw_U': 35}, "
        "'diagnostics': [], 'details': {'counterexamples': ['nested-detail']}, "
        "'missing_units': [{'index': i, 'why': 'X' * 80} for i in range(80)]}))\n"
        "print('evaluator-warning-after-json', file=sys.stderr)\n",
        encoding="utf-8",
    )
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": f"{sys.executable} long_evaluator.py {{candidate_path}}",
            "cwd": str(tmp_path),
            "timeout_seconds": 5,
        }
    )

    result = ExternalEvaluatorRunner().evaluate_candidate(CandidateGenome(id="C-long"), spec=spec)

    assert result.status == "passed"
    assert result.passed is True
    assert result.metrics["raw_U"] == 35
    assert len(result.details["missing_units"]) == 80
    assert result.details["counterexamples"] == ["nested-detail"]
    assert "details" not in result.details


def test_evaluator_parser_never_promotes_nested_verdicts() -> None:
    root_failure = {"passed": False, "metrics": {"correctness": True, "score": 0.1}, "diagnostics": []}
    root_success = {"passed": True, "metrics": {"score": 0.9}, "details": {"passed": False}}

    parsed_failure = _parse_evaluator_output(json.dumps(root_failure) + "\ntrailing stderr")
    parsed_success = _parse_evaluator_output("log prefix\n" + json.dumps(root_success))

    assert parsed_failure == root_failure
    assert parsed_success == root_success


def test_full_text_signatures_distinguish_tail_only_changes() -> None:
    shared = "same-prefix " * 800
    left = CandidateGenome(
        id="left",
        artifact=shared + "LEFT-TAIL",
        artifact_type="text",
        concise_claim="same claim",
        core_mechanism="same mechanism",
        parent_ids=["P"],
    )
    right = CandidateGenome(
        id="right",
        artifact=shared + "RIGHT-TAIL",
        artifact_type="text",
        concise_claim="same claim",
        core_mechanism="same mechanism",
        parent_ids=["P"],
    )
    left_plan = MutationPlan(operator="Deepen", parent_ids=["P"], instruction=shared + "LEFT", rarity_seed=shared + "L")
    right_plan = MutationPlan(operator="Deepen", parent_ids=["P"], instruction=shared + "RIGHT", rarity_seed=shared + "R")
    search_map = build_search_space_map(
        {
            "task_type": "test",
            "real_objective": "test",
            "candidate_families": [{"id": "known", "description": "known"}],
        },
        requested_candidate_count=1,
    )

    assert candidate_semantic_signature(left) != candidate_semantic_signature(right)
    assert plan_signature(left_plan) != plan_signature(right_plan)
    assert family_signature(left) != family_signature(right)
    assert transition_signature(left) != transition_signature(right)
    assert classify_candidate({"artifact": shared + "LEFT", "search_space": {"family_id": "unknown"}}, search_map)["family_id"] != classify_candidate(
        {"artifact": shared + "RIGHT", "search_space": {"family_id": "unknown"}}, search_map
    )["family_id"]


def test_inheritable_gene_keeps_full_mechanism_and_lessons() -> None:
    tail = "GENE-TAIL-MUST-SURVIVE"
    candidate = CandidateGenome(
        id="gene",
        core_mechanism=("mechanism " * 300) + tail,
        edge_knowledge_seeds=["edge-a", "edge-b", "edge-c"],
        failure_lessons=["lesson-a", "lesson-b", "lesson-c"],
    )

    gene = candidate.extract_inheritable_gene_summary()

    assert tail in gene
    assert "edge-c" in gene
    assert "lesson-c" in gene


def test_authoritative_problem_and_final_artifact_paths_are_lossless() -> None:
    tail = "TAIL-CONSTRAINT-MUST-SURVIVE"
    problem = ("long objective " * 300) + tail
    assessment = assess(problem)
    packet = TextInputPacket(raw_text=problem, extracted_claims=[])
    reformulation = reformulate_for_verification(problem)[0]
    artifact = {"matrix": ["0" * 100 for _ in range(180)], "tail": tail}
    rendered = _render_artifact(artifact)
    rendered_json = rendered.removeprefix("```json\n").removesuffix("\n```")

    assert assessment.surface_request == problem
    assert assessment.real_objective.lower().endswith(tail.lower())
    assert TextWorldModel.from_packet(packet).goal_summary == problem
    assert reformulation.reformulated_prompt.endswith(problem)
    assert json.loads(rendered_json) == artifact


def test_offspring_shaped_world_response_falls_back_to_full_packet_world() -> None:
    problem = "Construct X. Must preserve constraint Y."
    packet = TextInputPacket.from_text(problem)
    model = StructuredModelAdapter(caller=lambda *_: {"matrix": ["0"]})

    world = _build_text_world_model(packet, model=model)

    assert world.goal_summary
    assert world.input_packet_id == packet.packet_id
    assert "Must preserve constraint Y." in world.constraint_summary


def test_compatible_world_adapter_cannot_bypass_nonempty_goal_boundary() -> None:
    class CompatibleAdapter:
        def build_text_world_model(self, *, packet):
            return {"matrix": ["0"]}

    packet = TextInputPacket.from_text("Construct X. Must preserve constraint Y.")

    world = _build_text_world_model(packet, model=CompatibleAdapter())

    assert world.goal_summary == packet.raw_text
    assert "Must preserve constraint Y." in world.constraint_summary


def test_compatible_world_object_is_rebound_to_text_packet_identity() -> None:
    class CompatibleAdapter:
        def build_text_world_model(self, *, packet):
            return TextWorldModel(kind="project", input_packet_id="wrong", goal_summary="valid goal")

    packet = TextInputPacket.from_text("Construct X.")

    world = _build_text_world_model(packet, model=CompatibleAdapter())

    assert world.kind == "text"
    assert world.input_packet_id == packet.packet_id


def test_malformed_compatible_world_mapping_falls_back_instead_of_crashing() -> None:
    class CompatibleAdapter:
        def build_text_world_model(self, *, packet):
            return {"goal_summary": "x", "evidence_boundaries": "not-a-mapping"}

    packet = TextInputPacket.from_text("Construct X.")

    world = _build_text_world_model(packet, model=CompatibleAdapter())

    assert world.goal_summary == packet.raw_text


def test_exact_project_parent_preserves_complete_patch_and_execution_feedback() -> None:
    long_text = "BEGIN-" + ("P" * 5_000) + "-END"
    candidate = ProjectCandidateGenome(
        id="project-parent",
        artifact={},
        patch_set=[PatchOperation(path="module.py", operation="replace", old_text=long_text, new_text=long_text[::-1])],
        patch_application_result={"status": "failed", "diagnostics": [long_text]},
        commands_run=[{"command": "pytest", "raw_output_ref": long_text}],
    )

    view = candidate_prompt_view(candidate, detail="exact")

    assert view["patch_set"] == [operation.to_dict() for operation in candidate.patch_set]
    assert view["patch_application_result"] == candidate.patch_application_result
    assert view["commands_run"] == candidate.commands_run


def test_offspring_prompt_preserves_all_exact_parents_and_long_plans() -> None:
    sentinel = "PLAN-MIDDLE-SENTINEL"
    parents = [
        CandidateGenome(
            id=f"P{index}",
            parent_ids=[f"A{offset}" for offset in range(6)],
            artifact={"value": index},
            concise_claim=f"parent {index}",
            core_mechanism=f"mechanism {index}",
        )
        for index in range(20)
    ]
    instructions = [("L" * 2_500) + sentinel + str(index) + ("R" * 2_500) for index in range(20)]
    plans = [MutationPlan(operator="Deepen", parent_ids=[parents[index].id], instruction=instructions[index]) for index in range(20)]

    view = build_prompt_view(
        "nexus_generate_offspring",
        {
            "parents": parents,
            "plans": plans,
            "world": {"kind": "text", "goal_summary": "test"},
            "contract": NexusObjectiveContract(original_user_goal="test", normalized_goal="test"),
            "policy": EvolutionPolicy(),
        },
        max_chars=500_000,
    )

    assert [item["id"] for item in view.payload["parents"]] == [parent.id for parent in parents]
    assert all(item["parent_ids"] == parent.parent_ids for item, parent in zip(view.payload["parents"], parents))
    assert [item["instruction"] for item in view.payload["plans"]] == instructions


def test_prompt_candidate_selection_reserves_rare_and_dormant_buckets() -> None:
    def candidate(candidate_id: str, fate: str, score: float, *, rare: bool = False) -> CandidateGenome:
        return CandidateGenome(
            id=candidate_id,
            artifact={"id": candidate_id},
            concise_claim=candidate_id,
            core_mechanism=candidate_id,
            current_fate=fate,
            edge_knowledge_seeds=["rare"] if rare else [],
            multihead_scores={"answer_likelihood": score},
        )

    candidates = [candidate(f"E{i}", "Elite", 1.0 - i / 100) for i in range(4)]
    candidates += [candidate(f"A{i}", "Active", 0.9 - i / 100) for i in range(8)]
    candidates += [candidate(f"I{i}", "Incubating", 0.7 - i / 100) for i in range(4)]
    candidates += [candidate(f"R{i}", "Active", 0.2 - i / 100, rare=True) for i in range(3)]
    candidates += [candidate(f"D{i}", "Dormant", 0.1 - i / 100) for i in range(4)]

    selected = _select_candidates_for_prompt(candidates, limit=16)
    selected_ids = {item.id for item in selected}

    assert len(selected) == 16
    assert any(candidate_id.startswith("R") for candidate_id in selected_ids)
    assert any(candidate_id.startswith("D") for candidate_id in selected_ids)


def test_critique_prompt_can_read_middle_only_candidate_differences() -> None:
    shared_head = "H" * 1_000
    shared_tail = "T" * 1_000
    candidates = [
        CandidateGenome(id="left", artifact=shared_head + "LEFT-CAUSAL-ROUTE" + shared_tail, concise_claim="left", core_mechanism="left"),
        CandidateGenome(id="right", artifact=shared_head + "RIGHT-CAUSAL-ROUTE" + shared_tail, concise_claim="right", core_mechanism="right"),
    ]

    view = build_prompt_view(
        "nexus_critique_candidates",
        {"candidates": candidates, "contract": NexusObjectiveContract(original_user_goal="test", normalized_goal="test")},
        max_chars=250_000,
    )

    artifacts = {item["id"]: item["artifact"] for item in view.payload["candidates"]}
    assert "LEFT-CAUSAL-ROUTE" in artifacts["left"]
    assert "RIGHT-CAUSAL-ROUTE" in artifacts["right"]


def test_project_world_manifest_is_not_locally_top_n_truncated() -> None:
    file_roles = {f"src/module_{index}.py": "implementation" for index in range(120)}
    view = world_prompt_view({"kind": "project", "file_manifest": list(file_roles), "file_roles": file_roles})

    assert len(view["file_manifest"]) == 120
    assert len(view["file_roles"]) == 120


def test_exact_parent_prompt_preserves_causal_fields_and_evaluator_details() -> None:
    long_text = "BEGIN-" + ("Z" * 5_000) + "-END"
    candidate = CandidateGenome(
        id="C-exact",
        parent_ids=["P"],
        artifact={"answer": long_text},
        concise_claim=long_text,
        core_mechanism=long_text,
        assumptions=[long_text],
        missing_parts=[long_text],
        uncertainty_notes=[long_text],
        edge_knowledge_seeds=[long_text],
    )
    candidate.metadata["evaluator"] = {
        "status": "passed",
        "passed": True,
        "metrics": {"raw_U": 35},
        "diagnostics": [long_text],
        "details": {"missing_units": [{"description": long_text}]},
    }

    exact = candidate_prompt_view(candidate, detail="exact")
    summary = candidate_prompt_view(candidate, detail="summary")

    assert exact["concise_claim"] == long_text
    assert exact["core_mechanism"] == long_text
    assert exact["assumptions"] == [long_text]
    assert exact["external_evaluator"]["diagnostics"] == [long_text]
    assert exact["external_evaluator"]["details"]["missing_units"][0]["description"] == long_text
    assert len(summary["core_mechanism"]) > 500


def test_default_model_caps_no_longer_undercut_model_capacity(monkeypatch) -> None:
    for name in (
        "COGEV_LLM_MAX_PROMPT_CHARS",
        "COGEV_NEXUS_PROMPT_MAX_CHARS",
        "COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS",
        "COGEV_LLM_MAX_TOKENS",
        "COGEV_LLM_LIGHT_MAX_TOKENS",
        "COGEV_LLM_LONG_MAX_TOKENS",
        "COGEV_LLM_RETRY_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert llm_prompt_char_limit() == 0
    assert prompt_char_budget(long_context=False) == 250_000
    assert prompt_char_budget(long_context=True) == 500_000
    assert prompt_char_budget_details(long_context=True) == {
        "requested": 500_000,
        "transport": 0,
        "effective": 500_000,
        "clamped": False,
    }
    assert max_tokens_for_request("ordinary") == 16_384
    assert max_tokens_for_request("long", LLMRequestPolicy(long_context=True)) == 65_536

    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "100000")
    assert prompt_char_budget_details(long_context=True)["effective"] == 100_000
    assert prompt_char_budget_details(long_context=True)["clamped"] is True


def test_long_task_classification_prompt_is_not_replaced_by_a_hash_summary() -> None:
    tail = "CLASSIFICATION-TAIL-CONSTRAINT"
    prompt = ("unknown problem " * 2_000) + tail
    captured: list[dict] = []

    def caller(_request_type: str, payload: dict, _schema: dict) -> dict:
        captured.append(payload)
        return {"level": "L2_structured", "profile": "balanced", "search": True, "checkmodel": True, "artifacts": True, "reason": "test"}

    StructuredModelAdapter(caller=caller).classify_task(prompt=prompt)

    assert captured[0]["prompt"] == prompt


def test_self_evolve_entrypoints_do_not_reintroduce_legacy_model_caps() -> None:
    root = Path(__file__).resolve().parents[1]
    python_runner = (root / "scripts" / "run-core-self-evolve-openai.py").read_text(encoding="utf-8")
    shell_runner = (root / "scripts" / "run-core-self-evolve-openai.sh").read_text(encoding="utf-8")

    assert 'setdefault("COGEV_LLM_MAX_TOKENS"' not in python_runner
    assert "COGEV_MAX_PROMPT_CHARS" not in python_runner
    assert "COGEV_LLM_MAX_TOKENS" not in shell_runner


def test_project_source_context_has_one_wide_semantic_slice(tmp_path: Path) -> None:
    sentinel = "SOURCE-TAIL-SENTINEL"
    source = tmp_path / "module.py"
    source.write_text(("x = 1\n" * 8_000) + sentinel, encoding="utf-8")
    snapshot = ProjectSnapshot.from_path(tmp_path)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="inspect module tail")
    packet = ContextSelector(max_file_chars=64_000).build_context_packet(
        contract={},
        snapshot=snapshot,
        world=world,
        request=ContextRequest(need_files=["module.py"]),
    )
    source_context = ContextProtocolResult(packets=[packet]).to_source_context()

    assert sentinel in packet.raw_file_slices["module.py"]
    assert sentinel in source_context["slices"][0]["text"]
    assert source_context["budget_policy"] == "all_selected_files_full_selected_slices"


def test_wide_source_slice_preserves_the_complete_file_by_default(tmp_path: Path) -> None:
    source = tmp_path / "large.py"
    source_text = (
        ("A" * 20_000)
        + "QUARTER-SENTINEL"
        + ("B" * 30_000)
        + "MIDDLE-SENTINEL"
        + ("C" * 30_000)
        + "THREE-QUARTER-SENTINEL"
        + ("D" * 20_000)
    )
    source.write_text(source_text, encoding="utf-8")
    snapshot = ProjectSnapshot.from_path(tmp_path)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="inspect all source regions")

    packet = ContextSelector().build_context_packet(
        contract={},
        snapshot=snapshot,
        world=world,
        request=ContextRequest(need_files=["large.py"]),
    )

    assert packet.raw_file_slices["large.py"] == source_text
    assert packet.coverage["source_text_policy"] == "full_files"
    assert packet.coverage["truncated_files"] == {}


def test_context_selector_consumes_requested_symbols(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "target.py"
    target.parent.mkdir()
    target.write_text("class RequestedSymbol:\n    pass\n", encoding="utf-8")
    (target.parent / "other.py").write_text("class OtherSymbol:\n    pass\n", encoding="utf-8")
    snapshot = ProjectSnapshot.from_path(tmp_path)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="inspect a requested symbol")

    packet = ContextSelector().build_context_packet(
        contract={},
        snapshot=snapshot,
        world=world,
        request=ContextRequest(need_symbols=["pkg.target.RequestedSymbol", "MissingSymbol"]),
    )

    assert packet.coverage["symbol_selected_files"] == ["pkg/target.py"]
    assert packet.coverage["unresolved_symbols"] == ["MissingSymbol"]
    assert packet.raw_file_slices["pkg/target.py"] == target.read_text(encoding="utf-8")


def test_source_context_and_prompt_view_preserve_all_explicit_slices() -> None:
    paths = [f"pkg/module_{index}.py" for index in range(14)]
    packet = ContextPacket(
        objective_contract={},
        project_summary="test",
        raw_file_slices={path: f"SOURCE-{index}" for index, path in enumerate(paths)},
        source_hashes={path: f"hash-{index}" for index, path in enumerate(paths)},
    )

    source_context = ContextProtocolResult(packets=[packet]).to_source_context()
    view = build_prompt_view("nexus_generate_offspring", {"source_context": source_context}, max_chars=100_000)
    over_budget_view = build_prompt_view("nexus_generate_offspring", {"source_context": source_context}, max_chars=1_000)

    assert source_context["selected_files"] == paths
    assert source_context["context_limits"]["omitted_files"] == []
    assert view.payload["source_context"]["selected_files"] == paths
    assert [item["path"] for item in view.payload["source_context"]["slices"]] == paths
    assert [item["path"] for item in over_budget_view.payload["source_context"]["slices"]] == paths
    assert over_budget_view.metadata["protected_over_budget"] is True


def test_explicit_source_context_limits_are_observable() -> None:
    packet = ContextPacket(
        objective_contract={},
        project_summary="test",
        raw_file_slices={"a.py": "A" * 20, "b.py": "B" * 20, "c.py": "C" * 20},
    )

    source_context = ContextProtocolResult(packets=[packet]).to_source_context(max_files=2, max_chars=8)

    assert source_context["budget_policy"] == "top_2_files_capped_8chars_from_context_packets"
    assert source_context["context_limits"] == {
        "max_files": 2,
        "max_chars_per_file": 8,
        "available_file_count": 3,
        "selected_file_count": 2,
        "omitted_files": ["c.py"],
        "truncated_files": ["a.py", "b.py"],
    }
    assert source_context["slices"][0]["truncated"] is True
    assert source_context["slices"][0]["original_chars"] == 20
    assert source_context["slices"][0]["selected_chars"] == 8


class _DuplicateThenRefillModel:
    def __init__(self, *, first_batch_full: bool = False) -> None:
        self.calls = 0
        self.first_batch_full = first_batch_full
        self.policies: list[EvolutionPolicy] = []

    def generate_offspring(self, *, plans, parents, world, contract, policy):
        self.calls += 1
        self.policies.append(policy)
        if self.first_batch_full:
            return [
                CandidateGenome(id="new-a", parent_ids=[parents[0].id], artifact={"value": "new-a"}, concise_claim="new-a", core_mechanism="new-a"),
                CandidateGenome(id="new-b", parent_ids=[parents[0].id], artifact={"value": "new-b"}, concise_claim="new-b", core_mechanism="new-b"),
            ]
        if self.calls == 1:
            return [
                CandidateGenome(id="clone", parent_ids=[parents[0].id], artifact={"value": "prior"}, concise_claim="prior", core_mechanism="prior"),
                CandidateGenome(id="new-a", parent_ids=[parents[0].id], artifact={"value": "new-a"}, concise_claim="new-a", core_mechanism="new-a"),
            ]
        return [CandidateGenome(id="new-b", parent_ids=[parents[0].id], artifact={"value": "new-b"}, concise_claim="new-b", core_mechanism="new-b")]


class _PolicyAwareStatelessRefillModel:
    def __init__(self) -> None:
        self.calls = 0
        self.policies: list[EvolutionPolicy] = []

    def generate_offspring(self, *, plans, parents, world, contract, policy):
        self.calls += 1
        self.policies.append(policy)
        rejected = policy.metadata.get("rejected_offspring_feedback") or []
        accepted = policy.metadata.get("accepted_offspring_candidates") or []
        if rejected and accepted:
            return [CandidateGenome(id="new-b", parent_ids=[parents[0].id], artifact={"value": "new-b"}, concise_claim="new-b", core_mechanism="new-b")]
        return [
            CandidateGenome(id="clone", parent_ids=[parents[0].id], artifact={"value": "prior"}, concise_claim="prior", core_mechanism="prior"),
            CandidateGenome(id="new-a", parent_ids=[parents[0].id], artifact={"value": "new-a"}, concise_claim="new-a", core_mechanism="new-a"),
        ]


class _AlwaysCloneRefillModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate_offspring(self, *, plans, parents, world, contract, policy):
        self.calls += 1
        return [CandidateGenome(id=f"clone-{self.calls}", parent_ids=[parents[0].id], artifact={"value": "prior"}, concise_claim="prior", core_mechanism="prior")]


def _offspring_from(model: _DuplicateThenRefillModel) -> list[CandidateGenome]:
    parent = CandidateGenome(id="P", artifact={"value": "parent"}, concise_claim="parent", core_mechanism="parent")
    prior = CandidateGenome(id="OLD", artifact={"value": "prior"}, concise_claim="prior", core_mechanism="prior")
    plan = MutationPlan(operator="Deepen", parent_ids=[parent.id], instruction="create a distinct child")
    return _generate_offspring(
        model=model,
        mutation_engine=MutationEngine(),
        parents=[parent],
        plans=[plan],
        world={},
        contract=NexusObjectiveContract(original_user_goal="test", normalized_goal="test"),
        policy=EvolutionPolicy(),
        candidate_pool=[parent, prior],
        target_size=2,
    )


def test_exact_population_duplicate_does_not_trigger_refill() -> None:
    model = _DuplicateThenRefillModel()

    offspring = _offspring_from(model)

    assert model.calls == 1
    assert [candidate.artifact["value"] for candidate in offspring] == ["new-a"]


def test_stateless_duplicate_does_not_trigger_policy_feedback_refill() -> None:
    model = _PolicyAwareStatelessRefillModel()

    offspring = _offspring_from(model)  # type: ignore[arg-type]

    assert model.calls == 1
    assert [candidate.artifact["value"] for candidate in offspring] == ["new-a"]


def test_duplicate_exhaustion_never_returns_a_rejected_clone() -> None:
    model = _AlwaysCloneRefillModel()
    outcome: dict = {}
    parent = CandidateGenome(id="P", artifact={"value": "parent"}, concise_claim="parent", core_mechanism="parent")
    prior = CandidateGenome(id="OLD", artifact={"value": "prior"}, concise_claim="prior", core_mechanism="prior")

    offspring = _generate_offspring(
        model=model,
        mutation_engine=MutationEngine(),
        parents=[parent],
        plans=[MutationPlan(operator="Deepen", parent_ids=[parent.id], instruction="distinct child")],
        world={},
        contract=NexusObjectiveContract(original_user_goal="test", normalized_goal="test"),
        policy=EvolutionPolicy(),
        candidate_pool=[parent, prior],
        target_size=1,
        harvest_outcome=outcome,
    )

    assert model.calls == 1
    assert offspring == []
    assert outcome["status"] == "duplicate_exhausted"


def test_full_unique_first_batch_does_not_refill() -> None:
    model = _DuplicateThenRefillModel(first_batch_full=True)

    offspring = _offspring_from(model)

    assert model.calls == 1
    assert len(offspring) == 2


def test_generation_batch_feedback_survives_policy_prompt_profile() -> None:
    policy = _policy_for_generation_batch(
        EvolutionPolicy(),
        batch_index=1,
        accepted_signatures=["accepted-signature"],
        rejected=[{"candidate_id": "dup", "reason": "duplicate_materialized_artifact", "signature": "duplicate-signature"}],
        kind="offspring",
    )

    view = policy_prompt_view(policy)

    assert view["accepted_offspring_signatures"] == ["accepted-signature"]
    assert view["rejected_offspring_feedback"][0]["signature"] == "duplicate-signature"


def test_pool_preprocess_defaults_are_not_micro_capped() -> None:
    config = PreprocessConfig()

    assert config.max_report_chars == 500_000
    assert config.prompt_candidate_artifact_chars == 64_000


def test_model_authority_views_preserve_long_semantic_fields() -> None:
    sentinel = "BEGIN-" + ("X" * 8_000) + "-END"
    contract = contract_prompt_view(
        {
            "normalized_goal": "goal",
            "input_constraints": [sentinel],
            "dynamic_artifact_contract": {"adapter_requirements": {"detail": sentinel}},
        }
    )
    policy = policy_prompt_view(
        {
            "mutation_operators": [{"instruction": sentinel}],
            "metadata": {"seed_instruction": sentinel, "theory": {"mechanism": sentinel}},
        }
    )
    world = world_prompt_view(
        {
            "kind": "text",
            "goal_summary": "goal",
            "constraint_summary": [sentinel],
            "edge_seed_pool": [sentinel],
        }
    )
    candidate = CandidateGenome(
        id="exact",
        artifact="answer",
        metadata={
            "causal_context": sentinel,
            "repair_seed": {"blockers": [sentinel], "disallowed_repeat_patterns": [sentinel]},
        },
    )
    candidate_view = candidate_prompt_view(candidate, detail="exact")

    assert contract["input_constraints"] == [sentinel]
    assert contract["dynamic_artifact_contract"]["adapter_requirements"]["detail"] == sentinel
    assert policy["mutation_operators"][0]["instruction"] == sentinel
    assert policy["seed_instruction"] == sentinel
    assert world["constraint_summary"] == [sentinel]
    assert world["edge_seed_pool"] == [sentinel]
    assert candidate_view["metadata"]["causal_context"] == sentinel
    assert candidate_view["repair_seed_contract"]["blockers"] == [sentinel]


def test_derived_model_context_does_not_drop_late_or_long_semantics() -> None:
    sentinel = "BEGIN-" + ("Z" * 8_000) + "-END"
    packet = TextInputPacket.from_text("\n".join([f"claim {index}." for index in range(25)] + [sentinel + "."]))
    history = history_prompt_view(
        [{"round": index, "diagnosis": {"notes": sentinel if index == 9 else str(index)}} for index in range(10)]
    )
    representation = CandidateRepresentation.from_candidate(
        CandidateGenome(id="long-representation", concise_claim=sentinel, novelty_descriptors=[sentinel])
    )
    hints = _repair_hints([f"diagnostic-{index}" for index in range(12)] + [sentinel], "malformed")

    assert packet.extracted_claims[-1] == sentinel + "."
    assert len(packet.extracted_claims) == 26
    assert history[-1]["diagnosis"]["notes"] == sentinel
    assert representation.concise_claim == sentinel
    assert representation.novelty_descriptors == (sentinel,)
    assert hints[-1] == "repair diagnostic: " + sentinel


def test_synthesis_prompt_overrun_falls_back_to_full_local_candidate(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS", "9000")
    artifact = "BEGIN-" + ("X" * 20_000) + "-END"
    candidate = CandidateGenome(id="long-final", artifact=artifact, concise_claim="long final", core_mechanism="long final")
    called = False

    def caller(_request_type: str, _payload: dict, _schema: dict) -> dict:
        nonlocal called
        called = True
        return {"status": "model_synthesized", "final_answer": artifact, "best_candidate_id": candidate.id}

    result = synthesize_result(
        population=CandidatePopulation([candidate]),
        archives=ArchiveManager(),
        model=StructuredModelAdapter(caller=caller),
    )

    assert called is False
    assert result.status == "final_synthesis_local_fallback"
    assert result.final_answer == artifact
