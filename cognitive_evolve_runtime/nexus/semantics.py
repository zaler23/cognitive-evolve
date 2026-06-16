"""Nexus-native semantic profiling, task intake, and capability hints.

This module absorbs the useful parts of the v1 routing/intake/capability path
without reintroducing a second runtime.  It does not decide domain-specific
answers; it gives Nexus bounded runtime hints: profile, round defaults, evidence
needs, and one-shot task artifacts.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.artifacts.store import _append_trace, _read, _write_json
from cognitive_evolve_runtime.nexus.protocols import NexusClassifierProtocol

LEVEL_ORDER = {
    "L0_direct": 0,
    "L1_clarified": 1,
    "L2_structured": 2,
    "L3_comparative": 3,
    "L4_evolutionary": 4,
    "L5_longitudinal": 5,
    "L6_governed": 6,
}

DEFAULT_CAPABILITIES = [
    "local_execution",
    "project_governance",
    "task_scoping",
    "tool_boundary",
    "durable_execution",
    "cognitive_search",
    "independent_review",
    "user_cognition",
    "observability",
]

ROUND_DEFAULTS: dict[str, int] = {}
ROUND_MINIMUMS: dict[str, int] = {}
MAX_NEXUS_ROUNDS = 0  # 0 means semantic intake does not cap Nexus rounds.

SIGNAL_KEYS = ("project", "architecture", "research", "math", "risk", "evolve")


@dataclass(frozen=True)
class NexusRoute:
    level: str
    profile: str
    search: bool
    checkmodel: bool
    artifacts: bool
    reason: str
    semantic: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NexusSemanticAssessment:
    prompt: str
    route: NexusRoute
    surface_request: str
    real_objective: str
    task_type: str
    complexity_assessment: dict[str, float]
    weak_signals: dict[str, bool] = field(default_factory=dict)
    semantic_control: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    capability_hints: list[str] = field(default_factory=list)
    evidence_needs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_request": self.surface_request,
            "real_objective": self.real_objective,
            "task_type": self.task_type,
            "route": self.route.to_dict(),
            "complexity_assessment": dict(self.complexity_assessment),
            "weak_signals": dict(self.weak_signals),
            "semantic_control": dict(self.semantic_control),
            "hypotheses": list(self.hypotheses),
            "capability_hints": list(self.capability_hints),
            "evidence_needs": list(self.evidence_needs),
        }


def classify(prompt: str, *, model: NexusClassifierProtocol | None = None) -> NexusRoute:
    model_route = _model_route(prompt, model)
    if model_route is not None:
        return model_route
    return _model_unavailable_route()


def _model_unavailable_route() -> NexusRoute:
    signals = _neutral_signal_map()
    return NexusRoute(
        level="L4_evolutionary",
        profile="deep",
        search=True,
        checkmodel=True,
        artifacts=True,
        reason="nexus_semantic_profile:model_unavailable_conservative_default",
        semantic={
            "task_type": "model_unavailable_unclassified",
            "weak_signals": signals,
            "router_source": "model_unavailable_conservative",
            "fallback_only": True,
            "model_route_available": False,
            "fallback_caveat": "No model-authored semantic profile was available; the runtime must not infer domain or objective class from hard-coded keywords.",
        },
    )


def _neutral_signal_map() -> dict[str, bool]:
    return {name: False for name in SIGNAL_KEYS}


def assess(prompt: str, *, model: NexusClassifierProtocol | None = None, context: dict[str, Any] | None = None) -> NexusSemanticAssessment:
    context = dict(context or {})
    route = context.get("route")
    if isinstance(route, NexusRoute):
        nexus_route = route
    elif isinstance(route, dict):
        nexus_route = NexusRoute(
            level=str(route.get("level") or "L2_structured"),
            profile=str(route.get("profile") or "balanced"),
            search=bool(route.get("search", True)),
            checkmodel=bool(route.get("checkmodel", True)),
            artifacts=bool(route.get("artifacts", True)),
            reason=str(route.get("reason") or "provided_nexus_route"),
            semantic=dict(route.get("semantic") or {}),
        )
    else:
        nexus_route = classify(prompt, model=model)
    model_semantic = dict(nexus_route.semantic or {})
    signals = _semantic_signals(prompt, model_semantic)
    task_type = str(model_semantic.get("task_type") or infer_task_type(prompt, signals))
    incomplete = task_type == "route_incomplete"
    complexity = _semantic_complexity(nexus_route, signals, model_semantic)
    evidence_needs = _semantic_list(model_semantic.get("evidence_needs")) or evidence_needs_for(prompt, task_type, signals)
    return NexusSemanticAssessment(
        prompt=prompt,
        route=nexus_route,
        surface_request=prompt[:1200],
        real_objective=str(model_semantic.get("real_objective") or extract_real_objective(prompt)),
        task_type=task_type,
        complexity_assessment=complexity,
        weak_signals=signals,
        semantic_control={
            "incomplete": incomplete,
            "source": "nexus_semantic_assessment",
            "one_shot_external_questions_allowed": False,
            "task_type_diagnostic": {
                "task_type": task_type,
                "source": str(model_semantic.get("router_source") or ("model_route" if model_semantic.get("model_route_available", True) else "model_unavailable_conservative")),
                "fallback_only": bool(model_semantic.get("fallback_only")),
            },
        },
        hypotheses=_semantic_hypotheses(model_semantic) or task_hypotheses(prompt, task_type, signals),
        capability_hints=_semantic_list(model_semantic.get("capability_hints")) or required_capabilities(prompt, nexus_route),
        evidence_needs=evidence_needs,
    )


def weak_signal_map(prompt: str) -> dict[str, bool]:
    del prompt
    return _neutral_signal_map()


def _semantic_signals(prompt: str, semantic: dict[str, Any]) -> dict[str, bool]:
    raw = semantic.get("weak_signals") or semantic.get("signals")
    fallback = _neutral_signal_map()
    if not isinstance(raw, dict):
        return fallback
    merged = dict(fallback)
    for key, value in raw.items():
        if key in merged:
            merged[key] = bool(value)
    return merged


def _semantic_complexity(route: NexusRoute, signals: dict[str, bool], semantic: dict[str, Any]) -> dict[str, float]:
    raw = semantic.get("complexity_assessment")
    if isinstance(raw, dict):
        out: dict[str, float] = {}
        for key, value in raw.items():
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            out[str(key)] = max(0.0, min(1.0, parsed))
        if out:
            return out
    active_signals = sum(1 for value in signals.values() if value)
    signal_count = max(1, len(signals))
    level_rank = LEVEL_ORDER.get(route.level, LEVEL_ORDER["L2_structured"])
    level_count = max(1, max(LEVEL_ORDER.values()))
    signal_ratio = active_signals / signal_count
    return {
        "semantic_complexity": (level_rank / level_count + signal_ratio) / 2,
        "tool_dependency": 1.0 if signals["project"] else signal_ratio,
        "external_evidence_dependency": 1.0 if signals["research"] else signal_ratio,
        "risk": 1.0 if signals["risk"] else signal_ratio,
    }


def _semantic_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _semantic_hypotheses(semantic: dict[str, Any]) -> list[dict[str, Any]]:
    raw = semantic.get("hypotheses")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def infer_task_type(prompt: str, signals: dict[str, bool] | None = None) -> str:
    del prompt
    signals = signals or _neutral_signal_map()
    if signals["math"]:
        return "proof_resolution"
    if signals["architecture"]:
        return "architecture_refactor_or_migration"
    if signals["project"]:
        return "technical_execution_or_codebase_task"
    if signals["research"]:
        return "research_or_evidence_dependent_plan"
    if signals["evolve"]:
        return "mechanism_discovery"
    if len(prompt.strip()) < 80:
        return "direct_answer_or_small_edit"
    return "structured_decision_or_design"


def infer_level(prompt: str, signals: dict[str, bool] | None = None) -> str:
    signals = signals or weak_signal_map(prompt)
    word_count = len(re.findall(r"\w+", prompt))
    if signals["risk"]:
        return "L6_governed"
    if signals["evolve"] or signals["architecture"] or signals["math"]:
        return "L4_evolutionary"
    if signals["project"] or signals["research"]:
        return "L3_comparative"
    if word_count > 80:
        return "L2_structured"
    return "L0_direct"


def infer_profile(task_type: str, level: str) -> str:
    if task_type in {"proof_resolution", "open_conjecture", "mechanism_discovery", "novel_algorithm_design"}:
        return "exhaustive"
    if level in {"L4_evolutionary", "L5_longitudinal", "L6_governed"}:
        return "deep"
    if level in {"L2_structured", "L3_comparative"}:
        return "balanced"
    return "fast"


def default_rounds_for_route(route: NexusRoute, prompt: str | None = None) -> int:
    del route, prompt
    return 0


def minimum_rounds_for_route(route: NexusRoute) -> int:
    del route
    return 0


def resolve_rounds(route: NexusRoute, explicit_rounds: int | None, prompt: str | None = None) -> int:
    del route, prompt
    if explicit_rounds is not None:
        return max(1, int(explicit_rounds))
    return 0


def required_capabilities(prompt: str, route: NexusRoute | None = None) -> list[str]:
    route = route or classify(prompt)
    signals = _semantic_signals(prompt, getattr(route, "semantic", {}) if isinstance(getattr(route, "semantic", {}), dict) else {})
    required = ["cognitive_search", "durable_execution", "user_cognition", "observability"]
    if signals["project"] or signals["architecture"]:
        required.extend(["local_execution", "tool_boundary", "project_governance"])
    if signals["research"]:
        required.extend(["task_scoping", "independent_review"])
    if route.checkmodel:
        required.append("independent_review")
    if route.level in {"L4_evolutionary", "L5_longitudinal", "L6_governed"}:
        required.extend(["task_scoping", "project_governance", "tool_boundary"])
    if signals["risk"]:
        required.extend(["project_governance", "tool_boundary"])
    return list(dict.fromkeys(required))


def select_capability_ids(prompt: str, *, model: NexusClassifierProtocol | None = None) -> list[str]:
    return required_capabilities(prompt, classify(prompt, model=model))


def evidence_needs_for(prompt: str, task_type: str, signals: dict[str, bool] | None = None) -> list[str]:
    signals = signals or _neutral_signal_map()
    needs = ["input_evidence"]
    if signals["project"]:
        needs.extend(["local_files", "local_tests", "patch_verification"])
    if signals["research"]:
        needs.append("provided_sources_or_local_docs")
    if task_type in {"proof_resolution", "open_conjecture"}:
        needs.extend(["formal_or_symbolic_checks_when_available", "counterexample_search"])
    return list(dict.fromkeys(needs))


def extract_real_objective(prompt: str) -> str:
    stripped = " ".join(str(prompt or "").split())
    return stripped[:1600] or "Nexus evolution task"


def task_hypotheses(prompt: str, task_type: str, signals: dict[str, bool]) -> list[dict[str, Any]]:
    hypotheses = [{"hypothesis": task_type, "confidence": 0.65, "source": "nexus_semantic_profile"}]
    for name, active in signals.items():
        if active:
            hypotheses.append({"hypothesis": f"signal:{name}", "confidence": 0.55, "source": "nexus_weak_signal"})
    return hypotheses


def build_routed_prompt(prompt: str, route: NexusRoute, task_dir: Path | None = None) -> str:
    task_ref = f"Task directory: {task_dir}" if task_dir else "No task directory bound."
    checkmodel = "Use independent review/tool feedback when available." if route.checkmodel else "Independent review not required for this shallow route."
    return f"""
[COGEV NEXUS TASK]
Route level: {route.level}
Profile: {route.profile}
Search requested: {str(route.search).lower()}
Artifacts required: {str(route.artifacts).lower()}
Default Nexus rounds: adaptive/model-budgeted
Reason: {route.reason}
{task_ref}

One-shot policies:
- Do not ask the user mid-run.
- Resolve ambiguity internally through ObjectiveContract, EvolutionPolicy, evidence selection, candidate ranking, and verification feedback.
- Preserve full state locally, but pass bounded prompt views to the model.
- Candidate seeds are search material, not final answers.
- {checkmodel}

User task:
{prompt}
""".strip()


def ensure_enhanced_task_contract(task_dir: Path, prompt: str, *, print_summary: bool = False, force: bool = False, model: NexusClassifierProtocol | None = None) -> dict[str, Any]:
    intake_dir = task_dir / "intake"
    contract_path = intake_dir / "enhanced-task-contract.json"
    if contract_path.exists() and not force:
        try:
            return json.loads(contract_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    intake_dir.mkdir(parents=True, exist_ok=True)
    assessment = assess(prompt, model=model)
    contract = {
        "intake_status": "completed",
        "runtime_architecture": "nexus",
        "source": "nexus_semantic_intake",
        "external_questions_allowed": False,
        "prompt": prompt,
        "semantic_assessment": assessment.to_dict(),
        "route": assessment.route.to_dict(),
        "objective": assessment.real_objective,
        "constraints": ["one_shot", "no_midturn_user_clarification", "bounded_prompt_view"],
        "evidence_needs": assessment.evidence_needs,
        "capability_hints": assessment.capability_hints,
    }
    _write_json(contract_path, contract)
    _write_prompt_intake(intake_dir / "user-input.md", prompt, force=force)
    _write_prompt_intake(intake_dir / "original-user-input.md", prompt, force=force)
    (intake_dir / "enhanced-task-contract.md").write_text(_contract_markdown(contract), encoding="utf-8")
    _write_json(intake_dir / "external-questioning-disabled.json", {"external_question_count": 0, "policy": "nexus_one_shot"})
    _write_json(intake_dir / "internal-resolution-ledger.json", {"items": assessment.hypotheses, "policy": "nexus_internal_resolution"})
    _write_current_task_artifacts(task_dir, contract)
    _append_trace(task_dir, "nexus_semantic_intake", {"status": "completed", "route": assessment.route.to_dict(), "task_type": assessment.task_type})
    if print_summary:
        print(f"Nexus intake: {assessment.route.level} / {assessment.task_type}")
    return contract



def _write_current_task_artifacts(task_dir: Path, contract: dict[str, Any]) -> None:
    objective = str(contract.get("objective") or contract.get("prompt") or "Nexus task")
    route = contract.get("route", {}) if isinstance(contract.get("route"), dict) else {}
    assessment = contract.get("semantic_assessment", {}) if isinstance(contract.get("semantic_assessment"), dict) else {}
    route_summary = f"# Nexus Route Summary\n\nTask type: `{assessment.get('task_type')}`.\nRoute: `{route.get('level')}` / `{route.get('profile')}`.\n"
    route_summary_path = task_dir / "intake" / "route-summary.md"
    route_summary_path.parent.mkdir(parents=True, exist_ok=True)
    route_summary_path.write_text(route_summary, encoding="utf-8")
    artifact_texts = {
        "problem-contract.md": f"# Problem Contract\n\n## Objective\n{objective}\n\n## Runtime\nNexus-only, one-shot, no external mid-run questions.\n",
        "research-brief.md": f"# Research Brief\n\nTask type: `{assessment.get('task_type')}`.\nRoute: `{route.get('level')}` / `{route.get('profile')}`.\n",
        "decision-record.md": "# Decision Record\n\n- Use Nexus CandidateGenome evolution as the single runtime path.\n- Preserve full local state while sending bounded prompt views to the model.\n",
        "validation-plan.md": "# Validation Plan\n\n- Verify Nexus artifacts, checkpoint/progress consistency, and candidate verification traces.\n",
        "feedback.md": "# Feedback\n\nNexus semantic intake completed; runtime feedback will be appended through structured events.\n",
        "working-memory.md": "# Working Memory\n\nNexus objective, policy, population, archives, and verification state are persisted under `nexus-runtime/`.\n",
        "candidates/candidate-001.md": "# Seed Candidate\n\nInitial candidates are search material only; final synthesis must use evolved/verified genome state.\n",
        "evaluations/checkmodel-report.md": "# CheckModel Report\n\nNexus verifier stack will attach structured feedback to candidate genomes.\n",
        "evaluations/review-report.md": "# Review Report\n\nNexus verification feedback is recorded in candidate genomes and runtime self-check artifacts.\n",
    }
    for rel, text in artifact_texts.items():
        path = task_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == "research-brief.md" and _is_user_authored_research_brief(path):
            continue
        path.write_text(text, encoding="utf-8")


def _write_prompt_intake(path: Path, prompt: str, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = str(prompt or "")
    if not path.exists():
        path.write_text(incoming, encoding="utf-8")
        return
    existing = path.read_text(encoding="utf-8")
    if not existing.strip():
        path.write_text(incoming, encoding="utf-8")
        return
    # Preserve a previously captured full task prompt if a later intake pass is
    # given a short route summary or scaffold.  The newer short text is still
    # available in enhanced-task-contract.json / route-summary.md.
    if len(existing.strip()) > max(500, len(incoming.strip()) * 2):
        return
    if force or len(incoming.strip()) >= len(existing.strip()):
        path.write_text(incoming, encoding="utf-8")


def _is_user_authored_research_brief(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return False
    generated_shape = text.startswith("# Research Brief") and "Task type:" in text[:240] and "Route:" in text[:240]
    return not (generated_shape and len(text) < 500)

def enhance_request(prompt: str, *, path: str | None = None, print_json: bool = False, model: NexusClassifierProtocol | None = None) -> int:
    task_dir = Path(path) if path else Path.cwd()
    contract = ensure_enhanced_task_contract(task_dir, prompt, print_summary=not print_json, force=True, model=model)
    if print_json:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


def route_prompt(prompt: str, *, model: NexusClassifierProtocol | None = None) -> NexusRoute:
    route = classify(prompt, model=model)
    print(f"level: {route.level}")
    print(f"profile: {route.profile}")
    print(f"search: {str(route.search).lower()}")
    print(f"checkmodel_required: {str(route.checkmodel).lower()}")
    print(f"artifacts_required: {str(route.artifacts).lower()}")
    print(f"reason: {route.reason}")
    return route


def _contract_markdown(contract: dict[str, Any]) -> str:
    assessment = contract.get("semantic_assessment", {}) if isinstance(contract.get("semantic_assessment"), dict) else {}
    route = contract.get("route", {}) if isinstance(contract.get("route"), dict) else {}
    needs = "\n".join(f"- {item}" for item in contract.get("evidence_needs", [])) or "- input_evidence"
    caps = "\n".join(f"- {item}" for item in contract.get("capability_hints", [])) or "- cognitive_search"
    return f"""# Nexus Enhanced Task Contract

- Runtime: `nexus`
- Intake status: `{contract.get('intake_status')}`
- Route: `{route.get('level')}` / `{route.get('profile')}`
- Task type: `{assessment.get('task_type')}`
- External questions allowed: `false`

## Objective

{contract.get('objective')}

## Evidence Needs

{needs}

## Capability Hints

{caps}

## Constraints

- One-shot execution.
- No mid-turn user clarification.
- Full state persists locally; provider prompts use bounded Nexus views.
"""


def _model_route(prompt: str, model: NexusClassifierProtocol | None) -> NexusRoute | None:
    if not isinstance(model, NexusClassifierProtocol):
        return None
    try:
        raw = model.classify_task(prompt=prompt)
    except Exception:
        # Classification is an optional model-authored hint.  A provider/fixture
        # that does not implement this newer hook must not prevent the canonical
        # Nexus runtime from using the model for contract, ranking, mutation, and
        # synthesis later in the run.
        return None
    if isinstance(raw, NexusRoute):
        return raw
    if not isinstance(raw, dict):
        return None
    level = str(raw.get("level") or "").strip()
    semantic_raw = raw.get("semantic") if isinstance(raw.get("semantic"), dict) else {}
    task_type = str(raw.get("task_type") or semantic_raw.get("task_type") or "")
    if level not in LEVEL_ORDER or not task_type:
        return NexusRoute(
            level=level if level in LEVEL_ORDER else "L2_structured",
            profile=str(raw.get("profile") or "balanced"),
            search=bool(raw.get("search", True)),
            checkmodel=bool(raw.get("checkmodel", True)),
            artifacts=bool(raw.get("artifacts", True)),
            reason="model_route_incomplete",
            semantic={"task_type": "route_incomplete", "raw": raw, "router_source": "nexus_bounded_profile", "model_route_available": True, "fallback_only": False},
        )
    semantic = dict(semantic_raw)
    semantic.setdefault("task_type", task_type)
    semantic.setdefault("router_source", "model")
    semantic.setdefault("model_route_available", True)
    semantic.setdefault("fallback_only", False)
    for key in (
        "weak_signals",
        "signals",
        "evidence_needs",
        "capability_hints",
        "real_objective",
        "complexity_assessment",
        "difficulty",
        "difficulty_level",
        "difficulty_score",
        "complexity_score",
        "suggested_profile",
        "difficulty_profile",
        "evolution_profile",
        "suggested_rounds",
        "recommended_rounds",
        "model_self_assessment",
        "self_assessed_capability",
        "model_capability",
        "model_capability_score",
        "model_capability_tier",
        "model_round_multiplier",
        "target_output_level",
        "desired_output_level",
        "effort_class",
        "task_effort_class",
        "complexity_dimensions",
        "round_complexity_dimensions",
        "expected_round_range",
        "round_range",
        "hypotheses",
        "search_space",
        "candidate_families",
        "outcome_policy",
    ):
        if key in raw and key not in semantic:
            semantic[key] = raw[key]
    return NexusRoute(
        level=level,
        profile=str(raw.get("profile") or infer_profile(task_type, level)),
        search=bool(raw.get("search", LEVEL_ORDER[level] >= 3)),
        checkmodel=bool(raw.get("checkmodel", LEVEL_ORDER[level] >= 2)),
        artifacts=bool(raw.get("artifacts", LEVEL_ORDER[level] >= 2)),
        reason=str(raw.get("reason") or "model_classified_nexus_route"),
        semantic=semantic,
    )


__all__ = [
    "DEFAULT_CAPABILITIES",
    "MAX_NEXUS_ROUNDS",
    "NexusRoute",
    "NexusSemanticAssessment",
    "assess",
    "build_routed_prompt",
    "classify",
    "default_rounds_for_route",
    "enhance_request",
    "ensure_enhanced_task_contract",
    "minimum_rounds_for_route",
    "required_capabilities",
    "resolve_rounds",
    "route_prompt",
    "select_capability_ids",
    "weak_signal_map",
]
