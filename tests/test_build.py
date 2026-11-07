import json
import os
from pathlib import Path

import pandas as pd
import pytest

from eeg_pipeline import build as pipeline
from eeg_pipeline.config import PREDICTORS


def _candidates():
    return pd.DataFrame({
        "subject": ["sub-003", "sub-003"],
        "trial": [1, 2],
        "event_id": ["a", "b"],
        "block": [1, 1],
        "onset_s": [1.5, 3.0],
        "condition": ["standard", "function_dev"],
        "target": [0, 1],
        "source_available": [True, False],
    })


def test_validator_reconciles_source_metadata_and_every_candidate():
    candidates = _candidates()
    ledger = candidates.assign(included=[True, False], reason=["", "source_unavailable"])
    features = candidates.iloc[:1].assign(**{name: 0.0 for name in PREDICTORS})
    pipeline.validate_tables(features, ledger, candidates)
    with pytest.raises(ValueError, match="Ledger and source candidates"):
        pipeline.validate_tables(features, ledger.iloc[:1], candidates)
    with pytest.raises(ValueError, match="Features and included ledger metadata"):
        pipeline.validate_tables(features.assign(target=1, condition="function_dev"), ledger, candidates)
    with pytest.raises(ValueError, match="Invalid stimulus condition"):
        pipeline.validate_tables(features.assign(target=1, condition="typo"), ledger[["event_id", "included", "reason"]])


def test_build_lock_excludes_a_second_writer_and_cleans_up(tmp_path, monkeypatch):
    lock = tmp_path / ".build.lock"

    def fail_under_lock(data_dir, output_dir, subjects):
        assert int(lock.read_text(encoding="utf-8")) == os.getpid()
        with pytest.raises(RuntimeError, match="Build lock exists"):
            pipeline.build(data_dir, output_dir, subjects)
        raise ValueError("mock build failure")

    monkeypatch.setattr(pipeline, "_build", fail_under_lock)
    with pytest.raises(ValueError, match="mock build failure"):
        pipeline.build(tmp_path, tmp_path)
    assert not lock.exists()


@pytest.fixture
def severely_truncated(monkeypatch):
    candidates = _candidates().assign(source_available=False)
    sources = pd.DataFrame({"subject": ["sub-003"], "group": ["Cl"], "source_eligible": [False]})
    monkeypatch.setattr(pipeline, "audit", lambda data_dir, subjects: (sources, candidates))
    monkeypatch.setattr(
        pipeline, "pipeline_identity",
        lambda: ("test", {pipeline.MANIFEST.name: "test"}, {"python": "test"}),
    )
    return candidates


def test_all_excluded_build_publishes_accounting(tmp_path, severely_truncated):
    (tmp_path / "example.npz").write_bytes(b"stale")
    pipeline.build(tmp_path, tmp_path, ["sub-003"])

    features = pd.read_parquet(tmp_path / "events.parquet")
    assert features.empty
    assert list(features) == [*severely_truncated, *PREDICTORS, "group"]
    assert all(features.dtypes[name] == float for name in PREDICTORS)
    ledger = pd.read_csv(tmp_path / "event_ledger.csv.gz")
    assert list(ledger.event_id) == list(severely_truncated.event_id)
    assert ledger.reason.eq("source_severely_truncated").all()
    participant = pd.read_csv(tmp_path / "participant_audit.csv").iloc[0]
    assert participant.n_candidates == 2 and participant.n_included == 0
    assert not (tmp_path / "example.npz").exists()
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["included_events"] == 0
    assert "example.npz" not in provenance["files"]
    for name, checksum in provenance["files"].items():
        assert pipeline.file_checksum(tmp_path / name, "sha256") == checksum


def test_interrupted_publication_cannot_leave_old_provenance(tmp_path, severely_truncated, monkeypatch):
    (tmp_path / "provenance.json").write_text('{"old": true}', encoding="utf-8")
    replace = Path.replace

    def interrupted(path, destination):
        if destination.name == "participant_audit.csv":
            raise OSError("interrupted")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        pipeline.build(tmp_path, tmp_path, ["sub-003"])
    assert not (tmp_path / "provenance.json").exists()
    assert not (tmp_path / ".build.lock").exists()
    assert not list(tmp_path.glob(".publish-*"))
