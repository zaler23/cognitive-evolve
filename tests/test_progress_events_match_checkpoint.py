from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.events.progress import EvolutionProgressEvent, PipelineProgressEvent
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore
from cognitive_evolve_runtime.persistence.event_store import EventStore


def test_progress_events_match_checkpoint(tmp_path: Path) -> None:
    progress = EvolutionProgressEvent(
        round=3,
        max_rounds=5,
        population_size=2,
        active_candidates=1,
        dormant_candidates=1,
        archive_elites=1,
        tool_calls=4,
        best_answer_candidate="C1",
        search_diagnosis="DiversityCollapse",
        next_action="rare_inject",
    ).to_dict()
    event_store = EventStore(tmp_path / "events.jsonl")
    event_store.append(progress)

    checkpoint = CheckpointStore(tmp_path / "checkpoint.json").save_state(
        round=3,
        max_rounds=5,
        population=CandidatePopulation([CandidateGenome(id="C1")]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        progress_event=progress,
    )
    loaded = CheckpointStore(tmp_path / "checkpoint.json").load()

    assert loaded is not None
    assert loaded.round == checkpoint.round == progress["round"]
    assert event_store.read_all()[0]["type"] == "evolution_progress"
    assert PipelineProgressEvent(stage="candidate_evolution", stage_index=6, stage_count=9, stage_progress=0.56).to_dict()["stage_progress"] == 0.56


def test_normal_checkpoint_still_rejects_progress_round_mismatch(tmp_path: Path) -> None:
    progress = EvolutionProgressEvent(
        round=2,
        max_rounds=5,
        population_size=1,
        active_candidates=1,
        dormant_candidates=0,
        archive_elites=0,
        tool_calls=0,
    ).to_dict()

    with pytest.raises(ValueError, match="checkpoint round and progress event round differ"):
        CheckpointStore(tmp_path / "checkpoint.json").save_state(
            round=3,
            max_rounds=5,
            population=CandidatePopulation([CandidateGenome(id="C1")]),
            archives=ArchiveManager(),
            policy=EvolutionPolicy(),
            progress_event=progress,
        )


def test_error_checkpoint_repairs_progress_round_mismatch(tmp_path: Path) -> None:
    progress = EvolutionProgressEvent(
        round=2,
        max_rounds=5,
        population_size=1,
        active_candidates=1,
        dormant_candidates=0,
        archive_elites=0,
        tool_calls=0,
    ).to_dict()

    checkpoint = CheckpointStore(tmp_path / "checkpoint.json").save_state(
        round=3,
        max_rounds=5,
        population=CandidatePopulation([CandidateGenome(id="C1")]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        progress_event=progress,
        allow_progress_round_repair=True,
    )
    loaded = CheckpointStore(tmp_path / "checkpoint.json").load()

    assert loaded is not None
    assert checkpoint.round == loaded.round == 3
    assert loaded.progress_event["round"] == 3
    assert loaded.progress_event["metadata"]["repaired_progress_event_round"]["from"] == 2


def test_event_store_append_once_prevents_duplicate_round_events(tmp_path: Path) -> None:
    event_store = EventStore(tmp_path / "events.jsonl")
    event = EvolutionProgressEvent(
        round=1,
        max_rounds=2,
        population_size=1,
        active_candidates=1,
        dormant_candidates=0,
        archive_elites=0,
        tool_calls=0,
    ).to_dict()

    assert event_store.append_once(event) is not None
    assert event_store.append_once({**event, "at": "later"}) is None

    assert len(event_store.read_all()) == 1
