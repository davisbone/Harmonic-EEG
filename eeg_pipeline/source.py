"""Download, verify, and read the pinned NEMAR on003570 v1.0.0 release."""
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import EPOCH

MANIFEST = Path(__file__).with_name("manifest-v1.0.0.json")
TASK = "AuditoryOddballChords"
SUBJECTS = tuple(f"sub-{index:03d}" for index in range(1, 41))
PREFIXES = {"sub-001", "sub-030"}
SEVERE = {
    "sub-003", "sub-004", "sub-005", "sub-007", "sub-010",
    "sub-013", "sub-019", "sub-022", "sub-025", "sub-026",
}
MIDDLE_CHORD = re.compile(r"(?:fourRoot|four6|fiveRoot|five6)_(.+)")
CONDITIONS = {
    "S": "standard",
    "Sincorrect": "standard",
    "deviantF": "function_dev",
    "deviantFcorrect_F": "function_dev",
    "deviantE": "exemplar_dev",
    "deviantEcorrect_E": "exemplar_dev",
}
CHANNEL_COUNT = 64
SOURCE_RATE = 2048
FDT_DTYPE = np.dtype("<f4")


def source_path(data_dir, subject, suffix):
    return Path(data_dir) / subject / "eeg" / f"{subject}_task-{TASK}_{suffix}"


def file_checksum(path, algorithm):
    """Use the release's Git blob SHA-1 or its annex file checksum."""
    path = Path(path)
    digest = hashlib.new("sha1" if algorithm == "git" else algorithm)
    if algorithm == "git":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, lambda: digest).hexdigest()


def _verify_file(path, entry):
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned source file: {path}")
    if (
        path.stat().st_size != entry["size"]
        or file_checksum(path, entry["checksum_algorithm"]) != entry["checksum"]
    ):
        raise ValueError(f"Size or checksum differs from the pinned release: {path}")


def _selection(subjects):
    selected = SUBJECTS if subjects is None else tuple(subjects)
    if not selected or set(selected) - set(SUBJECTS) or len(set(selected)) != len(selected):
        raise ValueError("Subjects must be unique dataset IDs from sub-001 to sub-040")
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root_entries = []
    per_subject = {subject: [] for subject in selected}
    for entry in entries:
        folder = entry["path"].split("/")[0]
        if not folder.startswith("sub-"):
            root_entries.append(entry)
        elif folder in per_subject:
            per_subject[folder].append(entry)
    return root_entries, per_subject


def download(data_dir, subjects=None):
    """Fetch missing files and verify both downloaded and pre-existing files."""
    root_entries, per_subject = _selection(subjects)
    for entry in root_entries + [entry for group in per_subject.values() for entry in group]:
        path = Path(data_dir) / entry["path"]
        if path.exists():
            _verify_file(path, entry)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            print(f"Downloading {entry['path']}", flush=True)
            with urlopen(entry["bytes_url"], timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            _verify_file(temporary, entry)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def parse_events(events, subject):
    """Extract the middle chord and condition from each three-chord trial."""
    onsets = events["onset"].to_numpy(dtype=float)
    if not np.isfinite(onsets).all() or (onsets < 0).any() or (np.diff(onsets) < 0).any():
        raise ValueError(f"{subject} has invalid or unsorted event onsets")

    markers = events["value"].astype(str)
    blocks = markers.eq("boundary").cumsum() + 1
    stimulus = markers.isin(["two", "one"]) | markers.str.startswith(("four", "five"))
    chords = markers[stimulus].to_numpy()
    times = onsets[stimulus.to_numpy()]
    chord_blocks = blocks[stimulus].to_numpy()
    if not len(chords) or len(chords) % 3:
        raise ValueError(f"{subject} has incomplete chord sequences")

    rows = []
    for index in range(0, len(chords), 3):
        first, middle, last = chords[index:index + 3]
        first_time, onset, last_time = times[index:index + 3]
        first_block, block, last_block = chord_blocks[index:index + 3]
        match = MIDDLE_CHORD.fullmatch(middle)
        if first != "two" or last != "one" or match is None or match[1] not in CONDITIONS:
            raise ValueError(f"{subject} has an invalid chord sequence: {first}, {middle}, {last}")
        if first_block != last_block or not np.allclose(
            [onset - first_time, last_time - onset], 0.4, atol=2 / SOURCE_RATE, rtol=0,
        ):
            raise ValueError(f"{subject} has invalid within-trial timing at {onset:.6f} s")

        condition = CONDITIONS[match[1]]
        trial = index // 3 + 1
        rows.append({
            "subject": subject,
            "trial": trial,
            "event_id": f"{subject}_trial-{trial:04d}",
            "block": int(block),
            "onset_s": float(onset),
            "condition": condition,
            "target": int(condition != "standard"),
        })
    return pd.DataFrame(rows)


def _frame_geometry(fdt_bytes, expected_frames):
    """Return complete interleaved sample frames and any final incomplete bytes."""
    frames, trailing_bytes = divmod(fdt_bytes, CHANNEL_COUNT * FDT_DTYPE.itemsize)
    if frames > expected_frames or (frames == expected_frames and trailing_bytes):
        raise ValueError("FDT bytes exceed the recording geometry")
    return frames, trailing_bytes


def _source_status(subject, recorded_fraction, header_readable):
    if subject in SEVERE and recorded_fraction < 0.5:
        return "severely_truncated"
    if header_readable:
        if subject in PREFIXES and 0.92 <= recorded_fraction < 1:
            return "verified_prefix"
        if subject not in PREFIXES | SEVERE and recorded_fraction == 1:
            return "complete"
    raise ValueError(f"{subject} has unexpected recording coverage or unreadable header")


def _block_intervals(boundaries, available_frames):
    # EEGLAB places a boundary halfway between the last and first samples.
    cuts = np.ceil(boundaries * SOURCE_RATE).astype(int)
    cuts = cuts[cuts < available_frames]
    edges = np.r_[0, cuts, available_frames] / SOURCE_RATE
    return list(zip(edges[:-1], edges[1:]))


def _check_embedded_events(subject, events, embedded):
    markers = [str(event["type"]) for event in embedded]
    if markers != events["value"].tolist():
        raise ValueError(f"{subject} embedded and TSV event labels differ")
    latencies = np.array([float(event["latency"]) for event in embedded])
    # EEGLAB latencies are one-based. The TSV stores seconds to ten decimals.
    if not np.allclose((latencies - 1) / SOURCE_RATE, events["onset"], atol=1e-8, rtol=0):
        raise ValueError(f"{subject} embedded and TSV event timestamps differ")


def _read_fdt(path, available_frames):
    disk = np.memmap(path, dtype=FDT_DTYPE, mode="r", shape=(available_frames, CHANNEL_COUNT))
    volts = np.array(disk.T, dtype=np.float64, order="C")
    volts *= 1e-6  # EEGLAB stores microvolts. MNE RawArray accepts volts.
    return volts


def _inspect_subject(data_dir, subject):
    from scipy.io import loadmat

    fdt_path = source_path(data_dir, subject, "eeg.fdt")
    sidecar = json.loads(source_path(data_dir, subject, "eeg.json").read_text(encoding="utf-8"))
    if (sidecar["EEGChannelCount"], sidecar["SamplingFrequency"]) != (CHANNEL_COUNT, SOURCE_RATE):
        raise ValueError(f"{subject} has unexpected sidecar recording geometry")

    try:
        header = loadmat(
            source_path(data_dir, subject, "eeg.set"), simplify_cells=True, variable_names=["EEG"],
        )["EEG"]
    except OSError:
        if subject not in SEVERE:
            raise
        header = None

    channels = pd.read_csv(source_path(data_dir, subject, "channels.tsv"), sep="\t")["name"].tolist()
    if header is None:
        # The pinned release includes unreadable SET files for severely truncated participants.
        expected = round(sidecar["RecordingDuration"] * SOURCE_RATE)
    else:
        if (
            (header["nbchan"], header["srate"], header["trials"]) != (CHANNEL_COUNT, SOURCE_RATE, 1)
            or header["data"] != fdt_path.name
        ):
            raise ValueError(f"{subject} has unexpected SET recording geometry or companion filename")
        if [str(channel["labels"]) for channel in header["chanlocs"]] != channels:
            raise ValueError(f"{subject} SET and TSV channel orders differ")
        expected = int(header["pnts"])

    fdt_bytes = fdt_path.stat().st_size
    available, trailing_bytes = _frame_geometry(fdt_bytes, expected)
    fraction = available / expected
    status = _source_status(subject, fraction, header is not None)
    eligible = status != "severely_truncated"

    events = pd.read_csv(
        source_path(data_dir, subject, "events.tsv"), sep="\t", usecols=["onset", "value"],
        dtype={"value": str}, keep_default_na=False,
    )
    trials = parse_events(events, subject)
    boundaries = events.loc[events["value"].eq("boundary"), "onset"].to_numpy()
    if (
        len(boundaries) != 7
        or np.any(np.diff(boundaries) <= 0)
        or events["onset"].max() >= expected / SOURCE_RATE
        or trials.groupby("block").size().tolist() != [180] * 8
    ):
        raise ValueError(f"{subject} has unexpected recording boundaries or block counts")
    if header is not None:
        _check_embedded_events(subject, events, header["event"])

    # The baseline and signal window must fit the source. Internal block edges
    # and extra filter support are handled separately by preprocessing.
    trials["source_available"] = (
        eligible
        & (trials["onset_s"] + EPOCH[0] >= 0)
        & (trials["onset_s"] + EPOCH[1] <= available / SOURCE_RATE)
    )
    record = {
        "subject": subject,
        "source_status": status,
        "source_eligible": eligible,
        "sample_rate_hz": SOURCE_RATE,
        "expected_frames": expected,
        "available_frames": available,
        "recorded_fraction": fraction,
        "fdt_bytes": fdt_bytes,
        "trailing_bytes": trailing_bytes,
        "set_readable": header is not None,
        "embedded_events_verified": header is not None,
        "candidate_count": len(trials),
    }
    return record, trials, channels, _block_intervals(boundaries, available)


def audit(data_dir, subjects=None):
    """Verify pinned bytes and return participant source coverage and candidate events."""
    data_dir = Path(data_dir)
    root_entries, per_subject = _selection(subjects)
    for entry in root_entries:
        _verify_file(data_dir / entry["path"], entry)
    groups = pd.read_csv(data_dir / "participants.tsv", sep="\t").set_index("participant_id")["group"]
    records, candidates = [], []
    for subject, entries in per_subject.items():
        for entry in entries:
            _verify_file(data_dir / entry["path"], entry)
        record, trials, _, _ = _inspect_subject(data_dir, subject)
        record["group"] = groups.loc[subject]
        record["source_identity"] = hashlib.sha256(
            json.dumps(root_entries + entries, sort_keys=True).encode(),
        ).hexdigest()
        records.append(record)
        candidates.append(trials)
    return pd.DataFrame(records), pd.concat(candidates, ignore_index=True)


def load_subject(data_dir, subject):
    """Load one source-eligible participant from a previously audited directory."""
    import mne

    record, trials, channels, intervals = _inspect_subject(data_dir, subject)
    if not record["source_eligible"]:
        raise ValueError(f"{subject} has insufficient source EEG")
    channels = ["AFz" if name == "Afz" else name for name in channels]
    info = mne.create_info(channels, SOURCE_RATE, "eeg")
    volts = _read_fdt(source_path(data_dir, subject, "eeg.fdt"), record["available_frames"])
    raw = mne.io.RawArray(volts, info, verbose=False)
    raw.set_montage("biosemi64")
    raw.set_annotations(mne.Annotations([start for start, _ in intervals[1:]], 0.0, "EDGE boundary"))
    return raw, trials, intervals
