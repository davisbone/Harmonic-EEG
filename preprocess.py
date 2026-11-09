"""Turn the raw NEMAR on003570 EEG recordings into one table row per middle chord.

CS1090A Group 18, Milestone 2. The notebook (ms2_data.ipynb) walks through these
steps on one participant; build() runs them for everyone and writes results/.

Steps, one function each:
  1. load_participant      read the raw EEG, the chord events, and the 8 recording blocks
  2. downsample_and_filter 2048 -> 256 Hz, remove slow drift (high-pass) and power-line hum (notch)
  3. find_bad_channels     dead electrodes, plus PyPREP's noisy-electrode checks
  4. repair                rebuild bad electrodes from their neighbours, then average-reference
  5. fit_ica               find eye/muscle/heart/line-noise components (ICA + ICLabel)
  6. clean_epochs          remove those components, low-pass at 40 Hz, cut -100..400 ms epochs
  7. quality_check         reject epochs with >150 uV peak-to-peak (or flat / non-finite)
  8. erp_features          21 numbers per epoch: 7 scalp regions x 3 time windows

No step looks at the labels (standard vs deviant). Needs: uv sync --extra preprocessing
"""
import json
import re
import shutil
import warnings
from pathlib import Path
from urllib.request import urlopen

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep import NoisyChannels
from scipy.signal import firwin, resample_poly

# ---------------------------------------------------------------- settings
TASK = "AuditoryOddballChords"
MANIFEST = Path(__file__).with_name("data_manifest.json")  # file list of the pinned v1.0.0 release
SOURCE_SFREQ, SFREQ = 2048, 256  # recorded rate, working rate (Hz)
EPOCH = (-0.100, 0.400)          # seconds around the middle chord
BASELINE = (-0.100, 0.0)
ROIS = {
    "Lfront": ["AF3", "F3", "F5", "F7", "FC3", "FC5"],
    "Rfront": ["AF4", "F4", "F6", "F8", "FC4", "FC6"],
    "FCmid": ["Fz", "F1", "F2", "FC1", "FCz", "FC2"],
    "Cmid": ["C1", "Cz", "C2", "CP1", "CPz", "CP2"],
    "Pmid": ["P1", "Pz", "P2", "PO3", "POz", "PO4"],
    "Ltemp": ["T7", "TP7", "P7", "P9"],
    "Rtemp": ["T8", "TP8", "P8", "P10"],
}
WINDOWS = {"080_150": (0.080, 0.150), "150_250": (0.150, 0.250), "250_400": (0.250, 0.400)}
PREDICTORS = [f"erp_{region}_{window}_uv" for region in ROIS for window in WINDOWS]

MAX_BAD_CHANNELS = 6     # more than this and the participant is dropped
REJECT_UV, FLAT_UV = 150.0, 1.0
MIN_RETENTION = 0.6      # participant needs >= 60% of epochs to pass QC
EDGE_MARGIN_S = 1.0      # extra safety margin on top of the filter length
ICA_COMPONENTS, ICA_SEED, ICA_MAX_ITER, ICA_CONFIDENCE = 30, 97, 500, 0.8
ICA_REJECT = ["eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"]

# Filters. Zero-phase FIRs from MNE; the notch matches MNE's default 60/120 Hz notch.
RESAMPLE_TAPS = firwin(2 * 10 * 8 + 1, 1 / 8, window=("kaiser", 5.0))
HP = {"l_freq": 0.1, "h_freq": None}
LP = {"l_freq": None, "h_freq": 40.0}
ICA_BAND = {"l_freq": 1.0, "h_freq": 100.0}
NOTCH = {"l_freq": [f + f / 400 + 0.5 for f in (60.0, 120.0)],
         "h_freq": [f - f / 400 - 0.5 for f in (60.0, 120.0)],
         "l_trans_bandwidth": 0.5, "h_trans_bandwidth": 0.5}

# Middle-chord markers, e.g. "fiveRoot_deviantF". The suffix gives the condition.
MIDDLE_CHORD = re.compile(r"(?:fourRoot|four6|fiveRoot|five6)_(.+)")
CONDITIONS = {"S": "standard", "Sincorrect": "standard",
              "deviantF": "function_dev", "deviantFcorrect_F": "function_dev",
              "deviantE": "exemplar_dev", "deviantEcorrect_E": "exemplar_dev"}


# ---------------------------------------------------------------- getting the data
def download(data_dir, subjects=None):
    """Download the release files (skips files already present at the right size)."""
    for entry in json.loads(MANIFEST.read_text(encoding="utf-8")):
        folder = entry["path"].split("/")[0]
        if subjects and folder.startswith("sub-") and folder not in subjects:
            continue
        path = Path(data_dir) / entry["path"]
        if path.exists() and path.stat().st_size == entry["size"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading", entry["path"], flush=True)
        partial = path.with_name(path.name + ".part")
        with urlopen(entry["bytes_url"], timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, 8 * 1024 * 1024)
        if partial.stat().st_size != entry["size"]:
            raise IOError(f"Incomplete download: {entry['path']}")
        partial.replace(path)


def _file(data_dir, subject, suffix):
    return Path(data_dir) / subject / "eeg" / f"{subject}_task-{TASK}_{suffix}"


def parse_events(events, subject):
    """One row per trial (two -> middle chord -> one), labelled by the middle chord."""
    markers = events["value"].astype(str)
    block = markers.eq("boundary").cumsum() + 1  # a "boundary" marker starts a new block
    chord = markers.isin(["two", "one"]) | markers.str.startswith(("four", "five"))
    names, onsets, blocks = markers[chord].to_numpy(), events["onset"][chord].to_numpy(), block[chord].to_numpy()
    rows = []
    for i in range(0, len(names), 3):
        first, middle, last = names[i:i + 3]
        match = MIDDLE_CHORD.fullmatch(middle)
        if first != "two" or last != "one" or match is None:
            raise ValueError(f"{subject}: unexpected chord sequence {first}, {middle}, {last}")
        condition = CONDITIONS[match[1]]
        trial = i // 3 + 1
        rows.append({"subject": subject, "trial": trial, "event_id": f"{subject}_trial-{trial:04d}",
                     "block": int(blocks[i + 1]), "onset_s": float(onsets[i + 1]),
                     "condition": condition, "target": int(condition != "standard")})
    return pd.DataFrame(rows)


def load_participant(data_dir, subject):
    """Return (raw EEG or None, trials table, block start/end in seconds, recorded fraction).

    Ten recordings in the release are cut off (less than half the samples exist); those
    return raw=None and every trial is marked source_available=False.
    """
    events = pd.read_csv(_file(data_dir, subject, "events.tsv"), sep="\t", usecols=["onset", "value"],
                         dtype={"value": str}, keep_default_na=False)
    trials = parse_events(events, subject)
    sidecar = json.loads(_file(data_dir, subject, "eeg.json").read_text(encoding="utf-8"))
    fdt = _file(data_dir, subject, "eeg.fdt")
    available = fdt.stat().st_size // (64 * 4)  # complete frames of 64 float32 values
    fraction = available / (sidecar["RecordingDuration"] * SOURCE_SFREQ)
    usable = fraction >= 0.5
    trials["source_available"] = (usable & (trials.onset_s + EPOCH[0] >= 0)
                                  & (trials.onset_s + EPOCH[1] <= available / SOURCE_SFREQ))
    # EEGLAB puts a boundary between two samples; the next sample starts the new block.
    cuts = np.ceil(events.onset[events.value == "boundary"].to_numpy() * SOURCE_SFREQ).astype(int)
    edges = np.r_[0, cuts[cuts < available], available] / SOURCE_SFREQ
    blocks = list(zip(edges[:-1], edges[1:]))
    if not usable:
        return None, trials, blocks, fraction
    channels = pd.read_csv(_file(data_dir, subject, "channels.tsv"), sep="\t")["name"].tolist()
    channels = ["AFz" if name == "Afz" else name for name in channels]
    volts = np.array(np.memmap(fdt, dtype="<f4", mode="r", shape=(available, 64)).T, dtype=np.float64)
    volts *= 1e-6  # the file stores microvolts; MNE works in volts
    raw = mne.io.RawArray(volts, mne.create_info(channels, SOURCE_SFREQ, "eeg"), verbose=False)
    raw.set_montage("biosemi64")
    return raw, trials, blocks, fraction


# ---------------------------------------------------------------- cleaning
def edge_guard(*final_filters):
    """Seconds to stay away from a block edge so no filter runs off the end of the data."""
    half = (len(RESAMPLE_TAPS) - 1) / (2 * SOURCE_SFREQ)
    for spec in (HP, NOTCH, *final_filters):
        half += (len(mne.filter.create_filter(None, SFREQ, **spec, verbose=False)) - 1) / (2 * SFREQ)
    return half + EDGE_MARGIN_S


def downsample_and_filter(raw, blocks):
    """Each block separately: 2048 -> 256 Hz, then 0.1 Hz high-pass and 60/120 Hz notch."""
    info = mne.create_info(raw.ch_names, SFREQ, "eeg")
    info.set_montage(raw.get_montage())
    out = []
    for start, stop in blocks:
        native = raw.get_data(start=round(start * SOURCE_SFREQ), stop=round(stop * SOURCE_SFREQ))
        data = resample_poly(native, 1, 8, axis=1, window=RESAMPLE_TAPS, padtype="reflect")
        block = mne.io.RawArray(data, info, verbose=False)
        out.append(block.filter(**HP, verbose=False).filter(**NOTCH, verbose=False))
    return out


def find_bad_channels(blocks):
    """Dead electrodes (no signal) plus electrodes PyPREP flags in >= 50% of the recording."""
    channels = blocks[0].ch_names
    mad = []  # median absolute deviation: a robust "how much does this electrode move"
    for name in channels:
        values = np.concatenate([b.get_data(picks=[name]).ravel() for b in blocks])
        mad.append(np.median(np.abs(values - np.median(values))) if np.isfinite(values).all() else np.nan)
    mad = np.array(mad)
    typical = np.median(mad[np.isfinite(mad)])
    dead = {n for n, m in zip(channels, mad) if not np.isfinite(m) or m == 0 or m < 0.1 * typical}
    if len(dead) > MAX_BAD_CHANNELS:
        return [n for n in channels if n in dead]
    margin = int(np.ceil((edge_guard() - EDGE_MARGIN_S) * SFREQ))  # skip filter edge effects
    picks = [n for n in channels if n not in dead]
    flagged, tested = {}, 0
    for block in blocks:
        if block.n_times - 2 * margin < 2 * SFREQ:
            continue
        data = block.get_data(picks=picks, start=margin, stop=block.n_times - margin)
        info = mne.pick_info(block.info, [channels.index(n) for n in picks])
        noisy = NoisyChannels(mne.io.RawArray(data, info, verbose=False), do_detrend=False,
                              ransac=False, average_reference=True, random_state=ICA_SEED)
        noisy.find_bad_by_deviation(deviation_threshold=5.0)
        noisy.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
        noisy.find_bad_by_correlation(frac_bad=0.2)
        for name in noisy.get_bads():
            flagged[name] = flagged.get(name, 0) + data.shape[1]
        tested += data.shape[1]
    persistent = {n for n, samples in flagged.items() if samples / tested >= 0.5}
    return [n for n in channels if n in dead | persistent]


def repair(inst, bads):
    """Interpolate bad electrodes from their neighbours, then use the average reference."""
    inst.info["bads"] = list(bads)
    if bads:
        inst.interpolate_bads(reset_bads=True, verbose=False)
    inst.set_eeg_reference("average", verbose=False)


def fit_ica(blocks):
    """Fit ICA on 1-100 Hz copies of the blocks; mark artifact components with ICLabel."""
    length = 5 * SFREQ  # fit on 5-second segments that stay inside one block
    margin = int(np.ceil(edge_guard(ICA_BAND) * SFREQ))
    counts = [max(0, (b.n_times - 2 * margin) // length) for b in blocks]
    segments = []
    for block, count in zip(blocks, counts):
        if count:
            fit = block.copy().filter(**ICA_BAND, verbose=False)
            data = fit.get_data(start=margin, stop=margin + count * length)
            segments.append(data.reshape(len(fit.ch_names), count, length).transpose(1, 0, 2))
    fit = mne.EpochsArray(np.concatenate(segments), fit.info, verbose=False)
    rank = int(mne.compute_rank(fit, verbose=False)["eeg"])
    ica = ICA(n_components=min(ICA_COMPONENTS, rank), method="picard", random_state=ICA_SEED,
              fit_params={"ortho": False, "extended": True}, max_iter=ICA_MAX_ITER)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Picard did not converge.*", category=UserWarning)
        ica.fit(fit, decim=1, verbose=False)
    labels = label_components(fit, ica, method="iclabel")
    ica.exclude = [i for i, (label, p) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
                   if label in ICA_REJECT and p > ICA_CONFIDENCE]
    return ica, labels


def epoch_centers(trials, blocks, guard):
    """Sample index of each middle chord within its block, or -1 if too near a block edge."""
    ledger = trials.copy()
    centers = np.full(len(ledger), -1)
    ledger["reason"] = "source_unavailable"
    ledger["boundary_eligible"] = False
    for number, (start, stop) in enumerate(blocks, start=1):
        first, last = round(start * SOURCE_SFREQ), round(stop * SOURCE_SFREQ)
        rows = ledger.index[ledger.source_available & (ledger.block == number)]
        local = ledger.loc[rows, "onset_s"].to_numpy() - first / SOURCE_SFREQ
        center = np.rint(local * SFREQ).astype(int)
        left = np.minimum(local, center / SFREQ) + EPOCH[0]
        right = np.maximum(local, center / SFREQ) + EPOCH[1]
        duration = (last - first) / SOURCE_SFREQ
        inside = (left >= 0) & (right <= duration)
        safe = (left >= guard) & (right <= duration - guard)
        ledger.loc[rows, "reason"] = "block_boundary_overlap"
        ledger.loc[rows[inside], "reason"] = "filter_edge_guard"
        ledger.loc[rows[safe], "reason"] = ""
        ledger.loc[rows[safe], "boundary_eligible"] = True
        centers[rows[safe]] = center[safe]
    return ledger, centers


def _cut(block, centers, offsets):
    index = centers[:, None] + offsets
    if not len(index):
        return np.empty((0, len(block.ch_names), len(offsets)))
    first = int(index.min())
    return block.get_data(start=first, stop=int(index.max()) + 1)[:, index - first].transpose(1, 0, 2)


def clean_epochs(blocks, ica, ledger, centers):
    """Remove ICA artifacts, low-pass at 40 Hz, and cut epochs (events x channels x samples)."""
    offsets = epoch_offsets()
    eligible = ledger.index[ledger.boundary_eligible]
    epochs = np.empty((len(eligible), len(blocks[0].ch_names), len(offsets)))
    for number, block in enumerate(blocks, start=1):
        rows = eligible[ledger.loc[eligible, "block"] == number]
        if ica.exclude:
            ica.apply(block, verbose=False)
        block.filter(**LP, verbose=False)
        epochs[eligible.get_indexer(rows)] = _cut(block, centers[rows], offsets)
    return epochs


def epoch_offsets():
    return np.arange(*(int(np.ceil(t * SFREQ)) for t in EPOCH))


def repair_intermittent(epochs, raw, bad):
    """One extra repair pass for an electrode that is the worst offender in >5% of epochs."""
    channels = raw.ch_names
    ptp = np.ptp(epochs, axis=2) * 1e6
    ptp[~np.isfinite(ptp)] = -np.inf
    worst = ptp.argmax(axis=1)[ptp.max(axis=1) > REJECT_UV]
    counts = np.bincount(worst, minlength=len(channels))
    extra = [n for n, c in zip(channels, counts) if n not in bad and c > 0.05 * len(epochs)]
    total = [n for n in channels if n in set(bad + extra)]
    if extra and len(total) <= MAX_BAD_CHANNELS:
        info = mne.create_info(channels, SFREQ, "eeg")
        info.set_montage(raw.get_montage())
        repaired = mne.EpochsArray(epochs, info, tmin=epoch_offsets()[0] / SFREQ, verbose=False)
        repair(repaired, total)
        return repaired.get_data(copy=False), total, extra
    return epochs - epochs.mean(axis=1, keepdims=True), total, extra  # average reference


def quality_check(epochs):
    """Flag epochs with non-finite values, >150 uV or <1 uV peak-to-peak on any of the 64 channels."""
    finite = np.isfinite(epochs).all(axis=(1, 2))
    ptp = np.ptp(epochs, axis=2) * 1e6
    high, flat = (ptp > REJECT_UV).any(axis=1), (ptp < FLAT_UV).any(axis=1)
    reason = [";".join(r for r, bad in [("nonfinite", not f), ("amplitude", h), ("flat", z)] if bad)
              for f, h, z in zip(finite, high, flat)]
    return pd.DataFrame({"signal_reason": reason, "max_ptp_uv": ptp.max(axis=1),
                         "worst_channel_index": ptp.argmax(axis=1)})


# ---------------------------------------------------------------- features
def erp_features(epochs, channels):
    """Subtract each channel's -100..0 ms baseline, then average over region and time window (uV)."""
    times = epoch_offsets() / SFREQ
    base = (times >= BASELINE[0]) & (times < BASELINE[1])
    epochs = epochs - epochs[:, :, base].mean(axis=2, keepdims=True)
    values = {}
    for region, names in ROIS.items():
        regional = epochs[:, [channels.index(n) for n in names], :].mean(axis=1)
        for window, (start, stop) in WINDOWS.items():
            samples = (times >= start) & (times < stop)
            values[f"erp_{region}_{window}_uv"] = regional[:, samples].mean(axis=1) * 1e6
    return pd.DataFrame(values, columns=PREDICTORS)


# ---------------------------------------------------------------- one participant, then everyone
def process_participant(raw, trials, blocks):
    """Run steps 2-8. Returns (feature rows, ledger with a reason for every event, summary)."""
    ledger, centers = epoch_centers(trials, blocks, edge_guard(LP))
    ledger["signal_pass"] = False
    ledger["included"] = False
    ledger["signal_reason"] = ""
    ledger["max_ptp_uv"] = np.nan
    ledger["worst_channel"] = ""
    eligible = ledger.index[ledger.boundary_eligible]
    summary = {"bad_channels": [], "intermittent_channels": [], "ica_removed": 0}
    epochs = None
    if len(eligible):
        down = downsample_and_filter(raw, blocks)
        bad = find_bad_channels(down)
        summary["bad_channels"] = bad
        if len(bad) <= MAX_BAD_CHANNELS:
            for block in down:
                repair(block, bad)
            ica, _ = fit_ica(down)
            summary["ica_removed"] = len(ica.exclude)
            epochs = clean_epochs(down, ica, ledger, centers)
            epochs, total, extra = repair_intermittent(epochs, raw, bad)
            summary.update(bad_channels=total, intermittent_channels=extra)
            qc = quality_check(epochs)
            ledger.loc[eligible, "signal_reason"] = qc.signal_reason.to_numpy()
            ledger.loc[eligible, "signal_pass"] = (qc.signal_reason == "").to_numpy()
            ledger.loc[eligible, "max_ptp_uv"] = qc.max_ptp_uv.to_numpy()
            ledger.loc[eligible, "worst_channel"] = np.asarray(raw.ch_names)[qc.worst_channel_index]
    n_eligible = int(ledger.boundary_eligible.sum())
    retention = int((ledger.signal_pass & ledger.boundary_eligible).sum()) / n_eligible if n_eligible else 0.0
    drop = ("too_many_bad_channels" if len(summary["bad_channels"]) > MAX_BAD_CHANNELS
            else "insufficient_signal_retention" if retention < MIN_RETENTION else "")
    failed = ledger.signal_reason != ""
    ledger.loc[failed, "reason"] = "signal:" + ledger.loc[failed, "signal_reason"]
    rest = ledger.boundary_eligible & ~failed
    if drop:
        ledger.loc[rest, "reason"] = "participant:" + drop
    else:
        ledger.loc[rest, "included"] = True
    kept = ledger.index[ledger.included]
    values = pd.DataFrame(columns=PREDICTORS, dtype=float)
    if len(kept):
        values = erp_features(epochs[eligible.get_indexer(kept)], raw.ch_names)
    features = pd.concat([ledger.loc[kept, trials.columns].reset_index(drop=True), values], axis=1)
    summary.update(retention=retention, participant_reason=drop)
    return features, ledger, summary


def _ledger_columns(ledger):
    for column, value in [("boundary_eligible", False), ("signal_pass", False), ("included", False),
                          ("signal_reason", ""), ("max_ptp_uv", np.nan), ("worst_channel", "")]:
        if column not in ledger:
            ledger[column] = value
    order = ["subject", "trial", "event_id", "block", "onset_s", "condition", "target", "source_available",
             "reason", "boundary_eligible", "signal_pass", "included", "signal_reason", "max_ptp_uv",
             "worst_channel"]
    return ledger[order]


def build(data_dir, output_dir="results", subjects=None):
    """Process every participant and write events.parquet, event_ledger.csv.gz, participant_audit.csv."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(data_dir / "participants.tsv", sep="\t").set_index("participant_id")["group"]
    subjects = subjects or sorted(groups.index)
    tables, ledgers, audit = [], [], []
    for subject in subjects:
        raw, trials, blocks, fraction = load_participant(data_dir, subject)
        status = "severely_truncated" if raw is None else "partial" if fraction < 0.99 else "complete"
        if raw is None:
            ledger = trials.assign(reason="source_severely_truncated")
            summary = {"bad_channels": [], "intermittent_channels": [], "ica_removed": 0,
                       "retention": None, "participant_reason": "source_severely_truncated"}
        else:
            features, ledger, summary = process_participant(raw, trials, blocks)
            tables.append(features.assign(group=groups[subject]))
            del raw
        ledgers.append(_ledger_columns(ledger))
        audit.append({"subject": subject, "group": groups[subject], "source_status": status,
                      "recorded_fraction": round(fraction, 4),
                      "n_candidates": len(ledger), "n_source_available": int(ledger.source_available.sum()),
                      "n_edge_eligible": int(ledger.boundary_eligible.sum()),
                      "n_signal_pass": int(ledger.signal_pass.sum()), "n_included": int(ledger.included.sum()),
                      **{k: json.dumps(v) if isinstance(v, list) else v for k, v in summary.items()}})
        print(f"{subject}: {status}, {int(ledger.included.sum()):,} of {len(ledger):,} events kept", flush=True)
    events = pd.concat(tables, ignore_index=True).sort_values(["subject", "trial"], ignore_index=True)
    ledger = pd.concat(ledgers, ignore_index=True).sort_values(["subject", "trial"], ignore_index=True)
    assert set(events.event_id) == set(ledger.event_id[ledger.included])
    assert (ledger.included == ledger.reason.eq("")).all(), "every excluded event needs a reason"
    assert np.isfinite(events[PREDICTORS].to_numpy()).all()
    events.to_parquet(output_dir / "events.parquet", index=False)
    ledger.to_csv(output_dir / "event_ledger.csv.gz", index=False, lineterminator="\n",
                  compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(audit).to_csv(output_dir / "participant_audit.csv", index=False, lineterminator="\n")
    print(f"Wrote {len(events):,} events from {events.subject.nunique()} participants to {output_dir}")


if __name__ == "__main__":
    # python preprocess.py DATA_DIR OUTPUT_DIR [sub-002 ...]
    import sys
    build(sys.argv[1], sys.argv[2], subjects=sys.argv[3:] or None)
