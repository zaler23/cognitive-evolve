"""Receipt-grounded, read-only contextual replay for reproduction slots."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from cognitive_evolve_runtime.candidates.crossover import jaccard_similarity
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.evaluators.evidence_authority import stable_artifact_hash
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.nexus.receipts import (
    BlendReceipt,
    InterventionReceipt,
    MoveReceipt,
    ReceiptValidationError,
    TransferReceipt,
)
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import productive_outcomes
from cognitive_evolve_runtime.nexus.search_kernel.descriptor_cells import descriptor_cell_key
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import (
    base_mechanism_family,
    candidate_descriptor_tokens,
    normalize_token,
)


@dataclass(frozen=True)
class MoveReplayKey:
    stagnation_type: str
    parent_mechanism_family: str
    descriptor_cell: str
    failed_challenge_category: str
    move_kind: str
    artifact_type: str

    @property
    def context_bucket(self) -> str:
        return "context-" + stable_hash(
            {
                "stagnation_type": self.stagnation_type,
                "parent_mechanism_family": self.parent_mechanism_family,
                "descriptor_cell": self.descriptor_cell,
                "failed_challenge_category": self.failed_challenge_category,
                "artifact_type": self.artifact_type,
            }
        )[:20]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "context_bucket": self.context_bucket}


@dataclass(frozen=True)
class EmitterKey:
    context_bucket: str
    move_kind: str
    donor_role: str
    sampling_profile: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class MoveReplayEntry:
    key: MoveReplayKey
    candidate_id: str
    parent_id: str
    source_round: int
    outcome: str
    reward: float
    risk: float
    cost: float
    reason_codes: tuple[str, ...]
    descriptor_tokens: tuple[str, ...]
    donor_role: str
    sampling_profile: str
    receipt_refs: tuple[str, ...]
    summary: str
    obligation_categories: tuple[str, ...] = ()

    @property
    def net_credit(self) -> float:
        return self.reward - self.risk - self.cost

    @property
    def emitter(self) -> EmitterKey:
        return EmitterKey(
            context_bucket=self.key.context_bucket,
            move_kind=self.key.move_kind,
            donor_role=self.donor_role,
            sampling_profile=self.sampling_profile,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "source_round": self.source_round,
            "outcome": self.outcome,
            "reward": round(self.reward, 6),
            "risk": round(self.risk, 6),
            "cost": round(self.cost, 6),
            "net_credit": round(self.net_credit, 6),
            "reason_codes": list(self.reason_codes),
            "descriptor_tokens": list(self.descriptor_tokens),
            "emitter": self.emitter.to_dict(),
            "receipt_refs": list(self.receipt_refs),
            "summary": self.summary,
            "obligation_categories": list(self.obligation_categories),
        }


@dataclass(frozen=True)
class MoveReplayView:
    entries: tuple[MoveReplayEntry, ...]
    before_round: int | None = None

    @property
    def view_id(self) -> str:
        return "move-replay-" + stable_hash(
            {
                "before_round": self.before_round,
                "entries": [entry.to_dict() for entry in self.entries],
            }
        )[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "move-replay-view/v1",
            "read_only_derivation": True,
            "view_id": self.view_id,
            "before_round": self.before_round,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class MoveReplayQuery:
    stagnation_type: str
    parent_mechanism_family: str
    descriptor_cell: str
    failed_challenge_category: str
    artifact_type: str
    descriptor_tokens: tuple[str, ...]
    target_obligation_ids: tuple[str, ...] = ()
    obligation_categories: tuple[str, ...] = ()

    @property
    def context_bucket(self) -> str:
        return MoveReplayKey(
            stagnation_type=self.stagnation_type,
            parent_mechanism_family=self.parent_mechanism_family,
            descriptor_cell=self.descriptor_cell,
            failed_challenge_category=self.failed_challenge_category,
            move_kind="",
            artifact_type=self.artifact_type,
        ).context_bucket

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "descriptor_tokens": list(self.descriptor_tokens),
            "target_obligation_ids": list(self.target_obligation_ids),
            "obligation_categories": list(self.obligation_categories),
            "context_bucket": self.context_bucket,
        }


@dataclass(frozen=True)
class EmitterStats:
    emitter: EmitterKey
    pulls: int
    reward_sum: float
    risk_sum: float
    cost_sum: float
    selection_score: float

    @property
    def mean_net_credit(self) -> float:
        return (self.reward_sum - self.risk_sum - self.cost_sum) / max(1, self.pulls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitter": self.emitter.to_dict(),
            "pulls": self.pulls,
            "reward_sum": round(self.reward_sum, 6),
            "risk_sum": round(self.risk_sum, 6),
            "cost_sum": round(self.cost_sum, 6),
            "mean_net_credit": round(self.mean_net_credit, 6),
            "selection_score": round(self.selection_score, 6),
        }


@dataclass(frozen=True)
class ReplaySelection:
    view_id: str
    query: MoveReplayQuery
    successful: tuple[MoveReplayEntry, ...]
    failed: tuple[MoveReplayEntry, ...]
    preferred_emitter: EmitterKey
    emitter_stats: tuple[EmitterStats, ...]
    selection_basis: dict[str, Any]

    @property
    def receipt_refs(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                ref
                for entry in (*self.successful, *self.failed)
                for ref in entry.receipt_refs
            )
        )

    def to_directive(self) -> dict[str, Any]:
        def item(entry: MoveReplayEntry) -> dict[str, Any]:
            return {
                "summary": entry.summary,
                "receipt_refs": list(entry.receipt_refs),
                "move_kind": entry.key.move_kind,
                "net_credit": round(entry.net_credit, 6),
            }

        return {
            "schema": "receipt-grounded-slot-replay/v1",
            "successful_moves": [item(entry) for entry in self.successful],
            "failed_counterexamples": [item(entry) for entry in self.failed],
            "preferred_emitter": self.preferred_emitter.to_dict(),
            "target_obligation_ids": list(self.query.target_obligation_ids),
            "receipt_refs": list(self.receipt_refs),
            "selection_basis": dict(self.selection_basis),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "query": self.query.to_dict(),
            "successful_candidate_ids": [entry.candidate_id for entry in self.successful],
            "failed_candidate_ids": [entry.candidate_id for entry in self.failed],
            "receipt_refs": list(self.receipt_refs),
            "preferred_emitter": self.preferred_emitter.to_dict(),
            "emitter_stats": [stats.to_dict() for stats in self.emitter_stats],
            "selection_basis": dict(self.selection_basis),
        }


def derive_move_replay_view(
    *,
    budget_history: Iterable[dict[str, Any]],
    candidates: Iterable[CandidateGenome],
    metric_directions: dict[str, str] | None = None,
    before_round: int | None = None,
) -> MoveReplayView:
    """Derive replay entries without persisting a second archive or reward state."""

    candidate_by_id = _candidate_map(candidates, before_round=before_round)
    history = [dict(item) for item in budget_history if isinstance(item, dict)]
    supports = _receipt_supports(
        history,
        candidate_by_id=candidate_by_id,
        before_round=before_round,
    )
    transfer_hashes = _credited_transfer_hashes(history, before_round=before_round)
    outcomes = productive_outcomes(
        candidate_by_id.values(),
        metric_directions=metric_directions,
        credited_transfer_artifact_hashes=transfer_hashes,
    )
    outcomes_by_id = {outcome.candidate_id: outcome for outcome in outcomes}
    entries: list[MoveReplayEntry] = []
    for candidate_id in sorted(supports):
        candidate = candidate_by_id.get(candidate_id)
        outcome = outcomes_by_id.get(candidate_id)
        support = supports[candidate_id]
        if candidate is None or outcome is None:
            continue
        classification = _outcome_class(outcome.reward, outcome.risk, outcome.reason_codes)
        if not classification:
            continue
        parent_id = str(support.get("parent_id") or (candidate.parent_ids[0] if candidate.parent_ids else ""))
        parent = candidate_by_id.get(parent_id, candidate)
        challenge_categories, obligation_categories, _obligation_ids = _challenge_context(parent)
        move_kind = _preferred_move_kind(support)
        key = MoveReplayKey(
            stagnation_type=str(support.get("stagnation_type") or "None"),
            parent_mechanism_family=base_mechanism_family(parent),
            descriptor_cell=descriptor_cell_key(parent),
            failed_challenge_category="+".join(challenge_categories) or "none",
            move_kind=move_kind,
            artifact_type=normalize_token(parent.artifact_type or "answer") or "answer",
        )
        cost = _grounded_cost(parent=candidate)
        reason_codes = tuple(str(item) for item in outcome.reason_codes)
        tokens = set(candidate_descriptor_tokens(parent))
        for value in (
            key.stagnation_type,
            key.parent_mechanism_family,
            key.descriptor_cell,
            key.failed_challenge_category,
            key.move_kind,
            key.artifact_type,
            *obligation_categories,
        ):
            token = normalize_token(value)
            if token:
                tokens.add(token)
                tokens.update(part for part in token.split("_") if part)
        entries.append(
            MoveReplayEntry(
                key=key,
                candidate_id=candidate.id,
                parent_id=parent_id,
                source_round=int(support["source_round"]),
                outcome=classification,
                reward=float(outcome.reward),
                risk=float(outcome.risk),
                cost=cost,
                reason_codes=reason_codes,
                descriptor_tokens=tuple(sorted(tokens)),
                donor_role=str(support.get("donor_role") or "none"),
                sampling_profile=str(support.get("sampling_profile") or "default"),
                receipt_refs=tuple(sorted(support["receipt_refs"])),
                summary=_entry_summary(
                    move_kind=move_kind,
                    outcome=classification,
                    reasons=reason_codes,
                    stagnation_type=key.stagnation_type,
                    challenge_category=key.failed_challenge_category,
                    net_credit=float(outcome.reward) - float(outcome.risk) - cost,
                ),
                obligation_categories=obligation_categories,
            )
        )
    entries.sort(key=lambda entry: (entry.source_round, entry.candidate_id, entry.key.move_kind))
    return MoveReplayView(entries=tuple(entries), before_round=before_round)


def replay_query_for_parent(
    parent: CandidateGenome,
    *,
    diagnosis: Any,
) -> MoveReplayQuery:
    diagnosis_data = diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else coerce_dict(diagnosis)
    challenge_categories, obligation_categories, obligation_ids = _challenge_context(parent)
    stagnation_type = str(diagnosis_data.get("stagnation_type") or "None")
    family = base_mechanism_family(parent)
    cell = descriptor_cell_key(parent)
    artifact_type = normalize_token(parent.artifact_type or "answer") or "answer"
    tokens = set(candidate_descriptor_tokens(parent))
    for value in (
        stagnation_type,
        family,
        cell,
        artifact_type,
        *challenge_categories,
        *obligation_categories,
    ):
        token = normalize_token(value)
        if token:
            tokens.add(token)
            tokens.update(part for part in token.split("_") if part)
    return MoveReplayQuery(
        stagnation_type=stagnation_type,
        parent_mechanism_family=family,
        descriptor_cell=cell,
        failed_challenge_category="+".join(challenge_categories) or "none",
        artifact_type=artifact_type,
        descriptor_tokens=tuple(sorted(tokens)),
        target_obligation_ids=obligation_ids,
        obligation_categories=obligation_categories,
    )


def select_contextual_replay(
    view: MoveReplayView,
    query: MoveReplayQuery,
    *,
    available_move_kinds: Iterable[str],
    donor_role: str,
    default_sampling_profile: str,
) -> ReplaySelection:
    """Select bounded examples and one contextual emitter with deterministic UCB."""

    scored = [
        (entry, _entry_similarity(entry, query))
        for entry in view.entries
    ]
    similar = [(entry, score) for entry, score in scored if score > 1.0]
    successes = tuple(
        entry
        for entry, _score in sorted(
            (item for item in similar if item[0].outcome == "success"),
            key=lambda item: (-item[1], -item[0].net_credit, -item[0].source_round, item[0].candidate_id),
        )[:2]
    )
    failures = tuple(
        entry
        for entry, _score in sorted(
            (item for item in similar if item[0].outcome == "failure"),
            key=lambda item: (-item[1], item[0].net_credit, -item[0].source_round, item[0].candidate_id),
        )[:1]
    )
    moves = list(
        dict.fromkeys(
            _canonical_move_kind(item)
            for item in available_move_kinds
            if _canonical_move_kind(item)
        )
    ) or ["deepen"]
    obligation_forced = bool(query.target_obligation_ids)
    if obligation_forced:
        moves = ["repair"]
    role = str(donor_role or "none")
    default_profile = str(default_sampling_profile or "default")
    contextual_entries = [
        entry
        for entry, score in similar
        if score > 1.0
        and entry.key.context_bucket == query.context_bucket
        and entry.key.move_kind in moves
        and entry.donor_role == role
    ]
    historical_profiles: dict[str, list[str]] = {move: [] for move in moves}
    for entry in contextual_entries:
        profiles = historical_profiles[entry.key.move_kind]
        if entry.sampling_profile not in profiles:
            profiles.append(entry.sampling_profile)
    emitters: list[EmitterKey] = []
    for move in moves:
        profiles = historical_profiles[move] or [default_profile]
        if default_profile not in profiles:
            profiles.append(default_profile)
        emitters.extend(
            EmitterKey(
                context_bucket=query.context_bucket,
                move_kind=move,
                donor_role=role,
                sampling_profile=profile,
            )
            for profile in profiles
        )
    total_pulls = len(contextual_entries)
    stats: list[EmitterStats] = []
    for emitter in emitters:
        matched = [
            entry
            for entry in contextual_entries
            if entry.key.move_kind == emitter.move_kind
            and entry.sampling_profile == emitter.sampling_profile
        ]
        pulls = len(matched)
        reward_sum = sum(entry.reward for entry in matched)
        risk_sum = sum(entry.risk for entry in matched)
        cost_sum = sum(entry.cost for entry in matched)
        mean = (reward_sum - risk_sum - cost_sum) / max(1, pulls)
        exploration = math.sqrt(2.0 * math.log(total_pulls + 2.0) / max(1, pulls))
        stats.append(
            EmitterStats(
                emitter=emitter,
                pulls=pulls,
                reward_sum=reward_sum,
                risk_sum=risk_sum,
                cost_sum=cost_sum,
                selection_score=mean + exploration,
            )
        )
    move_order = {move: index for index, move in enumerate(moves)}
    preferred_stats = min(
        stats,
        key=lambda item: (
            -item.selection_score,
            -item.mean_net_credit,
            move_order[item.emitter.move_kind],
            item.emitter.sampling_profile,
        ),
    )
    ordered_stats = tuple(
        sorted(
            stats,
            key=lambda item: (
                move_order[item.emitter.move_kind],
                item.emitter.sampling_profile,
            ),
        )
    )
    basis = {
        "similarity": "descriptor_token_jaccard",
        "similar_entry_count": len(similar),
        "selected_similarity_scores": {
            entry.candidate_id: round(score, 6)
            for entry, score in similar
            if entry in (*successes, *failures)
        },
        "max_successes": 2,
        "max_failed_counterexamples": 1,
        "credit": "productive_outcome_reward_minus_observed_risk_and_cost",
        "emitter_selection": "contextual_ucb_existing_slot_palette",
        "obligation_category_boost": obligation_forced,
    }
    return ReplaySelection(
        view_id=view.view_id,
        query=query,
        successful=successes,
        failed=failures,
        preferred_emitter=preferred_stats.emitter,
        emitter_stats=ordered_stats,
        selection_basis=basis,
    )


def _candidate_map(
    candidates: Iterable[CandidateGenome],
    *,
    before_round: int | None,
) -> dict[str, CandidateGenome]:
    out: dict[str, CandidateGenome] = {}
    for candidate in candidates:
        created_round = _candidate_round(candidate)
        if before_round is not None and created_round >= before_round:
            continue
        current = out.get(candidate.id)
        if current is None or _grounded_field_count(candidate) > _grounded_field_count(current):
            out[candidate.id] = candidate
    return out


def _candidate_round(candidate: CandidateGenome) -> int:
    metadata = coerce_dict(candidate.metadata)
    try:
        return int(metadata.get("created_in_round") or 0)
    except (TypeError, ValueError):
        return 0


def _grounded_field_count(candidate: CandidateGenome) -> int:
    metadata = coerce_dict(candidate.metadata)
    return sum(
        bool(item)
        for item in (
            metadata.get("evaluator"),
            metadata.get("evidence_state"),
            candidate.verification_result,
            getattr(candidate, "patch_application_result", None),
        )
    )


def _receipt_supports(
    history: list[dict[str, Any]],
    *,
    candidate_by_id: dict[str, CandidateGenome],
    before_round: int | None,
) -> dict[str, dict[str, Any]]:
    supports: dict[str, dict[str, Any]] = {}
    artifact_candidates: dict[str, list[str]] = {}
    for candidate in candidate_by_id.values():
        transfer = coerce_dict(coerce_dict(candidate.metadata).get("transfer_receipt"))
        artifact_hash = str(transfer.get("artifact_hash") or "")
        if artifact_hash and artifact_hash == stable_artifact_hash(candidate.artifact):
            artifact_candidates.setdefault(artifact_hash, []).append(candidate.id)

    def support(candidate_id: str, *, source_round: int) -> dict[str, Any] | None:
        if candidate_id not in candidate_by_id:
            return None
        current = supports.setdefault(
            candidate_id,
            {
                "source_round": source_round,
                "receipt_refs": set(),
                "move_kinds": [],
                "stagnation_type": "None",
                "donor_role": "none",
                "sampling_profile": "default",
                "parent_id": "",
            },
        )
        current["source_round"] = min(int(current["source_round"]), source_round)
        return current

    for record in history:
        plan = coerce_dict(record.get("generation_plan"))
        source_round = _history_round(record, plan)
        if before_round is not None and source_round >= before_round:
            continue
        profile_by_slot = {
            str(item.get("slot_id") or ""): str(item.get("sampling_profile_id") or "default")
            for item in plan.get("slot_sampling_profiles", [])
            if isinstance(item, dict) and item.get("slot_id")
        }
        diagnosis = coerce_dict(record.get("diagnosis"))
        default_stagnation = str(diagnosis.get("stagnation_type") or "None")
        for raw in plan.get("intervention_receipts", []):
            try:
                receipt = InterventionReceipt.from_dict(raw)
            except ReceiptValidationError:
                continue
            move_kind = _canonical_move_kind(receipt.target.get("action"))
            for candidate_id in receipt.produced_candidate_ids:
                current = support(candidate_id, source_round=source_round)
                if current is None:
                    continue
                current["receipt_refs"].add(receipt.receipt_id)
                current["move_kinds"].append((3, move_kind))
                current["stagnation_type"] = str(receipt.diagnosed_pressure.get("stagnation_type") or default_stagnation)
                slot_id = next(
                    (slot for slot in receipt.recipient_branch_slot_ids if slot in profile_by_slot),
                    "",
                )
                if slot_id:
                    current["sampling_profile"] = profile_by_slot[slot_id]
        for raw in plan.get("move_receipts", []):
            try:
                receipt = MoveReceipt.from_dict(raw)
            except ReceiptValidationError:
                continue
            current = support(receipt.candidate_id, source_round=source_round)
            if current is None:
                continue
            current["receipt_refs"].add(receipt.receipt_id)
            current["move_kinds"].append((0, _canonical_move_kind(receipt.move_kind)))
            current["parent_id"] = receipt.parent_id
            if current["stagnation_type"] == "None":
                current["stagnation_type"] = default_stagnation
            slot_id = str(candidate_by_id[receipt.candidate_id].metadata.get("branch_slot_id") or "")
            if slot_id in profile_by_slot:
                current["sampling_profile"] = profile_by_slot[slot_id]
        for raw in plan.get("blend_receipts", []):
            try:
                receipt = BlendReceipt.from_dict(raw)
            except ReceiptValidationError:
                continue
            current = support(receipt.candidate_id, source_round=source_round)
            if current is None:
                continue
            current["receipt_refs"].add(receipt.receipt_id)
            current["move_kinds"].append((2, "crossover"))
            current["parent_id"] = receipt.primary_parent_id
            current["donor_role"] = receipt.donor_role
            if current["stagnation_type"] == "None":
                current["stagnation_type"] = default_stagnation
            slot_id = str(candidate_by_id[receipt.candidate_id].metadata.get("branch_slot_id") or "")
            if slot_id in profile_by_slot:
                current["sampling_profile"] = profile_by_slot[slot_id]
        for raw in plan.get("transfer_receipts", []):
            try:
                receipt = TransferReceipt.from_dict(raw)
            except ReceiptValidationError:
                continue
            for candidate_id in artifact_candidates.get(receipt.artifact_hash, []):
                current = support(candidate_id, source_round=source_round)
                if current is None:
                    continue
                current["receipt_refs"].add("transfer:" + receipt.artifact_hash)
                current["move_kinds"].append((1, "transfer"))
                if current["stagnation_type"] == "None":
                    current["stagnation_type"] = default_stagnation
                slot_id = str(candidate_by_id[candidate_id].metadata.get("branch_slot_id") or "")
                if slot_id in profile_by_slot:
                    current["sampling_profile"] = profile_by_slot[slot_id]
    return {
        candidate_id: data
        for candidate_id, data in supports.items()
        if data["receipt_refs"] and data["move_kinds"]
    }


def _preferred_move_kind(support: Mapping[str, Any]) -> str:
    ranked = sorted(
        (
            (int(priority), _canonical_move_kind(move_kind))
            for priority, move_kind in support.get("move_kinds", [])
            if _canonical_move_kind(move_kind)
        ),
        key=lambda item: (item[0], item[1]),
    )
    return ranked[0][1]


def _canonical_move_kind(value: Any) -> str:
    token = normalize_token(value)
    if "repair" in token or token in {"discharge_obligation", "case_split"}:
        return "repair"
    if "transfer" in token:
        return "transfer"
    if "cross" in token or "blend" in token:
        return "crossover"
    return token or "deepen"


def _challenge_context(
    candidate: CandidateGenome,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    metadata = coerce_dict(candidate.metadata)
    evidence_state = coerce_dict(metadata.get("evidence_state"))
    challenges: set[str] = set()
    for raw in (
        evidence_state.get("target_challenge_ids"),
        metadata.get("target_challenge_ids"),
    ):
        if isinstance(raw, list):
            challenges.update(normalize_token(item) for item in raw if normalize_token(item))
    failure = coerce_dict(metadata.get("failure_classification") or metadata.get("failure_class"))
    for key in ("class", "kind", "reason", "category"):
        token = normalize_token(failure.get(key))
        if token:
            challenges.add(token)
    obligation_categories: set[str] = set()
    obligation_ids: list[str] = []
    for item in candidate.proof_obligations:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "pending").strip().lower()
        if status in {"passed", "discharged", "refuted", "resolved"}:
            continue
        obligation_id = str(item.get("id") or item.get("obligation_id") or "").strip()
        source = normalize_token(item.get("source") or item.get("category") or item.get("kind"))
        if source:
            challenges.add(source)
        if source == "blend_receipt":
            obligation_categories.add(source)
            if obligation_id:
                obligation_ids.append(obligation_id)
    return (
        tuple(sorted(challenges)),
        tuple(sorted(obligation_categories)),
        tuple(dict.fromkeys(obligation_ids)),
    )


def _entry_similarity(entry: MoveReplayEntry, query: MoveReplayQuery) -> float:
    score = 4.0 * jaccard_similarity(entry.descriptor_tokens, query.descriptor_tokens)
    score += float(entry.key.stagnation_type == query.stagnation_type)
    score += 2.0 * float(entry.key.parent_mechanism_family == query.parent_mechanism_family)
    score += 2.0 * float(entry.key.descriptor_cell == query.descriptor_cell)
    score += 2.0 * float(entry.key.failed_challenge_category == query.failed_challenge_category)
    score += float(entry.key.artifact_type == query.artifact_type)
    if set(entry.obligation_categories) & set(query.obligation_categories):
        score += 3.0
    return score


def _outcome_class(reward: float, risk: float, reason_codes: Iterable[str]) -> str:
    if float(reward) > 0.0:
        return "success"
    grounded_failures = {
        "unbound_branch_slot",
        "observed_evaluator_or_patch_failure",
        "exact_phenotype_duplicate",
        "terminal_or_verification_failure",
    }
    if float(risk) > 0.0 or grounded_failures.intersection(str(item) for item in reason_codes):
        return "failure"
    return ""


def _grounded_cost(*, parent: CandidateGenome) -> float:
    metadata = coerce_dict(parent.metadata)
    evaluator = coerce_dict(metadata.get("evaluator"))
    costs = [coerce_dict(evaluator.get("cost"))]
    records = metadata.get("evidence_records")
    if isinstance(records, list):
        costs.extend(coerce_dict(item.get("cost")) for item in records if isinstance(item, dict))
    values: list[float] = []
    for cost in costs:
        for key in ("normalized", "penalty", "risk"):
            try:
                value = float(cost.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(max(0.0, min(1.0, value)))
                break
    return max(values, default=0.0)


def _entry_summary(
    *,
    move_kind: str,
    outcome: str,
    reasons: tuple[str, ...],
    stagnation_type: str,
    challenge_category: str,
    net_credit: float,
) -> str:
    reason = ",".join(reasons[:3]) or "grounded_outcome"
    return (
        f"move={move_kind}; outcome={outcome}; grounded={reason}; "
        f"context={normalize_token(stagnation_type) or 'none'}/{challenge_category}; "
        f"net_credit={round(net_credit, 6)}"
    )


def _history_round(record: dict[str, Any], plan: dict[str, Any]) -> int:
    try:
        return int(record.get("round", plan.get("round_index", -1)))
    except (TypeError, ValueError):
        return -1


def _credited_transfer_hashes(
    history: Iterable[dict[str, Any]],
    *,
    before_round: int | None,
) -> set[str]:
    hashes: set[str] = set()
    for record in history:
        if not isinstance(record, dict):
            continue
        plan = coerce_dict(record.get("generation_plan"))
        source_round = _history_round(record, plan)
        if before_round is not None and source_round >= before_round:
            continue
        allocation = coerce_dict(plan.get("productive_branch_allocation"))
        hashes.update(
            str(item)
            for item in allocation.get("credited_transfer_artifact_hashes", [])
            if str(item)
        )
    return hashes


__all__ = [
    "EmitterKey",
    "EmitterStats",
    "MoveReplayEntry",
    "MoveReplayKey",
    "MoveReplayQuery",
    "MoveReplayView",
    "ReplaySelection",
    "derive_move_replay_view",
    "replay_query_for_parent",
    "select_contextual_replay",
]
