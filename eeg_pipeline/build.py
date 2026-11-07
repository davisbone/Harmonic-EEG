"""Rebuild the published table, reusing only verified per-participant results."""
import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PREDICTORS, SETTINGS
from .source import CONDITIONS, MANIFEST, audit, file_checksum, load_subject


def pipeline_identity():
    root = Path(__file__).resolve().parent
    files = [root / name for name in ("config.py", "source.py", "features.py", "processing.py")]
    files.extend([MANIFEST, root.parent / "uv.lock"])
    hashes = {path.name: file_checksum(path, "sha256") for path in files}
    packages = ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn",
                "mne", "mne-icalabel", "onnxruntime", "pyprep", "python-picard")
    runtime = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "packages": {name: version(name) for name in packages},
    }
    identity = json.dumps({"hashes": hashes, "runtime": runtime}, sort_keys=True).encode()
    return hashlib.sha256(identity).hexdigest(), hashes, runtime


def _match_metadata(actual, expected, columns, description):
    try:
        pd.testing.assert_frame_equal(
            actual[columns].sort_values("event_id").reset_index(drop=True),
            expected[columns].sort_values("event_id").reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except (AssertionError, KeyError) as exc:
        raise ValueError(f"{description} differ: {exc}") from exc


def validate_tables(features, ledger, candidates=None):
    """Check the scientific table contract, not just whether files were written."""
    if features.event_id.duplicated().any() or ledger.event_id.duplicated().any():
        raise ValueError("Duplicate event identities")
    included = ledger.loc[ledger.included]
    if set(features.event_id) != set(included.event_id):
        raise ValueError("Included ledger rows do not match feature rows")
    shared = [column for column in features if column in ledger]
    if candidates is not None:
        shared = list(candidates)
        _match_metadata(ledger, candidates, shared, "Ledger and source candidates")
    _match_metadata(features, included, shared, "Features and included ledger metadata")
    if not np.isfinite(features[PREDICTORS].to_numpy()).all():
        raise ValueError("EEG predictors contain missing or infinite values")
    if [c for c in features if c.startswith("erp_")] != PREDICTORS:
        raise ValueError("Predictor columns differ from the fixed allowlist")
    if not features.condition.isin(CONDITIONS.values()).all():
        raise ValueError("Invalid stimulus condition")
    if not features.target.isin([0, 1]).all():
        raise ValueError("Invalid binary target")
    if not (features.target == features.condition.ne("standard").astype(int)).all():
        raise ValueError("Target and condition disagree")
    if not (ledger.included == ledger.reason.eq("")).all():
        raise ValueError("Every exclusion needs a reason; included events must have none")


def _save_cache(directory, features, ledger, qc, example, identity):
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        features.to_parquet(temporary / "features.parquet", index=False)
        ledger.to_parquet(temporary / "ledger.parquet", index=False)
        (temporary / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8", newline="\n")
        np.savez_compressed(temporary / "example.npz", **example)
        hashes = {p.name: file_checksum(p, "sha256") for p in temporary.iterdir()}
        (temporary / "complete.json").write_text(
            json.dumps({"identity": identity, "files": hashes}, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_cache(directory, identity):
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if complete["identity"] != identity:
        raise ValueError(f"Unexpected cache identity: {directory}")
    for name, expected in complete["files"].items():
        if file_checksum(directory / name, "sha256") != expected:
            raise ValueError(f"Cached output changed: {directory / name}")
    return (pd.read_parquet(directory / "features.parquet"),
            pd.read_parquet(directory / "ledger.parquet"),
            json.loads((directory / "qc.json").read_text(encoding="utf-8")))


def _source_exclusion(trials):
    ledger = trials.copy()
    ledger["boundary_eligible"] = False
    ledger["signal_pass"] = False
    ledger["included"] = False
    ledger["signal_reason"] = ""
    ledger["max_ptp_uv"] = np.nan
    ledger["worst_channel"] = ""
    ledger["reason"] = "source_severely_truncated"
    return ledger


def build(data_dir, output_dir, subjects=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".build.lock"
    try:
        stream = lock.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise RuntimeError(f"Build lock exists: {lock}") from exc
    try:
        with stream:
            stream.write(f"{os.getpid()}\n")
        _build(data_dir, output_dir, subjects)
    finally:
        lock.unlink(missing_ok=True)


def _build(data_dir, output_dir, subjects):
    print("Verifying pinned source files and event timing…", flush=True)
    sources, candidates = audit(data_dir, subjects)
    fingerprint, code_hashes, runtime = pipeline_identity()
    tables, ledgers, records = [], [], []
    example_path = None
    for source in sources.to_dict("records"):
        subject = source["subject"]
        trials = candidates[candidates.subject == subject].reset_index(drop=True)
        if not source["source_eligible"]:
            ledger = _source_exclusion(trials)
            ledgers.append(ledger)
            records.append({**source, "filter_guard_s": None, "dead_channels": "[]",
                            "bad_channels": "[]", "intermittent_channels": "[]", "ica_components": 0,
                            "ica_excluded": "[]", "ica_labels": "[]", "ica_probabilities": "[]",
                            "ica_converged": None, "ica_rank": None, "ica_iterations": None,
                            "retention": None, "n_candidates": len(trials), "n_source_available": 0,
                            "n_edge_eligible": 0, "n_signal_pass": 0, "n_included": 0,
                            "participant_reason": "source_severely_truncated"})
            print(f"{subject}: severely truncated source; {len(trials)} events accounted for", flush=True)
            continue
        identity = f"{fingerprint}:{source['source_identity']}"
        cache = output_dir / ".cache" / fingerprint / subject
        if cache.exists():
            features, ledger, qc = _load_cache(cache, identity)
            print(f"{subject}: reusing verified result", flush=True)
        else:
            from .processing import process_subject

            print(f"{subject}: processing EEG", flush=True)
            raw, loaded_trials, blocks = load_subject(data_dir, subject)
            features, ledger, qc, example = process_subject(raw, loaded_trials, blocks)
            del raw
            validate_tables(features, ledger, trials)
            _save_cache(cache, features, ledger, qc, example, identity)
        features["group"] = source["group"]
        tables.append(features)
        ledgers.append(ledger)
        # Keep list-valued diagnostics in the CSV as JSON, rather than Python repr.
        summary = {key: json.dumps(value) if isinstance(value, (list, dict)) else value
                   for key, value in qc.items()}
        records.append({**source, **summary})
        if len(features) and (example_path is None or subject == "sub-002"):
            with np.load(cache / "example.npz") as example:
                if "included" in example and example["included"].any():
                    example_path = cache / "example.npz"
        print(f"{subject}: {len(features):,} included / {len(trials):,} candidate events", flush=True)

    ledger = pd.concat(ledgers, ignore_index=True).sort_values(["subject", "trial"])
    if tables:
        features = pd.concat(tables, ignore_index=True).sort_values(["subject", "trial"])
    else:
        features = candidates.iloc[:0].assign(
            **{name: pd.Series(dtype=float) for name in PREDICTORS},
            group=pd.Series(dtype=sources["group"].dtype),
        )
    validate_tables(features, ledger, candidates)
    participant_audit = pd.DataFrame(records)
    # Publish only after the whole selected cohort has completed successfully.
    staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=output_dir))
    try:
        features.to_parquet(staging / "events.parquet", index=False)
        ledger.to_csv(
            staging / "event_ledger.csv.gz",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            compression={"method": "gzip", "mtime": 0},
        )
        participant_audit.to_csv(
            staging / "participant_audit.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        if example_path:
            shutil.copyfile(example_path, staging / "example.npz")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        provenance = {
            "dataset": "NEMAR on003570", "version": "1.0.0",
            "source_url": "https://www.nemar.org/dataset/on003570",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "subjects_audited": sources.subject.tolist(),
            "full_release_bytes": sum(entry["size"] for entry in manifest),
            "pipeline_identity": fingerprint, "code_hashes": code_hashes,
            "manifest_sha256": code_hashes[MANIFEST.name], "runtime": runtime,
            "settings": SETTINGS, "predictors": PREDICTORS,
            "candidate_events": len(ledger), "included_events": len(features),
            "included_participants": features.subject.nunique(),
            "files": {p.name: file_checksum(p, "sha256") for p in staging.iterdir()},
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8", newline="\n"
        )
        # A generation is valid only when the provenance marker and its hashes agree.
        (output_dir / "provenance.json").unlink(missing_ok=True)
        if example_path is None:
            (output_dir / "example.npz").unlink(missing_ok=True)
        for path in sorted(staging.iterdir(), key=lambda p: p.name == "provenance.json"):
            path.replace(output_dir / path.name)
    finally:
        shutil.rmtree(staging)
    print(f"Published {len(features):,} events from {features.subject.nunique()} participants to {output_dir}", flush=True)
