"""Label-blind cleaning and trial accounting for one recording."""
import warnings

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep import NoisyChannels
from scipy.signal import firwin, resample_poly

from .config import EPOCH, PREDICTORS, SETTINGS, SFREQ
from .features import baseline_correct, feature_table, quality

SOURCE_SFREQ = 2048
DOWNSAMPLE = SOURCE_SFREQ // SFREQ
RESAMPLE_TAPS = firwin(2 * 10 * DOWNSAMPLE + 1, 1 / DOWNSAMPLE, window=("kaiser", 5.0))
HP = {"l_freq": SETTINGS["analysis_band_hz"][0], "h_freq": None}
LP = {"l_freq": None, "h_freq": SETTINGS["analysis_band_hz"][1]}
ICA_BAND = dict(zip(("l_freq", "h_freq"), SETTINGS["ica_band_hz"]))
# The same stopbands and transitions as MNE's default FIR notch filter.
NOTCH = {"l_freq": [f + f / 400 + 0.5 for f in SETTINGS["notch_hz"]],
         "h_freq": [f - f / 400 - 0.5 for f in SETTINGS["notch_hz"]],
         "l_trans_bandwidth": 0.5, "h_trans_bandwidth": 0.5}


def _epoch_offsets(sfreq):
    return np.arange(*(int(np.ceil(t * sfreq)) for t in EPOCH))


def _guard(*final_filters):
    resample_half_support = (len(RESAMPLE_TAPS) - 1) / (2 * SOURCE_SFREQ)
    fir_half_support = sum(
        (len(mne.filter.create_filter(None, SFREQ, **spec, verbose=False)) - 1) / (2 * SFREQ)
        for spec in (HP, NOTCH, *final_filters)
    )
    return resample_half_support + fir_half_support + SETTINGS["edge_margin_s"]


def _epoch_locations(ledger, blocks, guard):
    """Identify source and edge exclusions before looking at the signal."""
    centers = np.full(len(ledger), -1, dtype=int)
    ledger["reason"] = "source_unavailable"
    ledger["boundary_eligible"] = False
    for block, (first, last) in enumerate(blocks, start=1):
        rows = ledger.index[ledger.source_available & (ledger.block == block)]
        local = ledger.loc[rows, "onset_s"].to_numpy() - first / SOURCE_SFREQ
        center = np.rint(local * SFREQ).astype(int)
        # Guard both the intended interval and the interval at the nearest sample.
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
    return centers


def _downsample_block(raw, first, last, info):
    native = raw.get_data(start=first, stop=last)
    data = resample_poly(native, 1, DOWNSAMPLE, axis=1, window=RESAMPLE_TAPS, padtype="reflect")
    del native
    return mne.io.RawArray(data, info, verbose=False).filter(**HP, verbose=False).filter(**NOTCH, verbose=False)


def _flagged_duration_fractions(block_reports):
    durations = {}
    total = sum(report["n_tested_samples"] for report in block_reports)
    for report in block_reports:
        for name in report["bad_channels"]:
            durations[name] = durations.get(name, 0) + report["n_tested_samples"]
    return {name: duration / total for name, duration in sorted(durations.items())}


def _bad_channels(blocks):
    channels = blocks[0].ch_names
    mad = []
    for name in channels:
        values = np.concatenate([block.get_data(picks=[name]).ravel() for block in blocks])
        mad.append(np.median(np.abs(values - np.median(values))) if np.isfinite(values).all() else np.nan)
    mad = np.array(mad)
    finite = np.isfinite(mad)
    typical = np.median(mad[finite]) if finite.any() else 0.0
    dead = {name for name, value in zip(channels, mad)
            if not np.isfinite(value) or value == 0 or value < SETTINGS["dead_mad_ratio"] * typical}
    block_reports = []
    if len(dead) <= SETTINGS["max_bad_channels"]:
        margin = int(np.ceil((_guard() - SETTINGS["edge_margin_s"]) * SFREQ))
        picks = [name for name in channels if name not in dead]
        for number, block in enumerate(blocks, start=1):
            # Ignore edge transients when scoring the already high-passed blocks.
            if block.n_times - 2 * margin < 2 * SFREQ:
                continue
            trimmed = block.get_data(picks=picks, start=margin, stop=block.n_times - margin)
            info = mne.pick_info(block.info, [channels.index(name) for name in picks])
            noisy = NoisyChannels(mne.io.RawArray(trimmed, info, verbose=False),
                                  do_detrend=False, ransac=False, average_reference=True,
                                  random_state=SETTINGS["ica_seed"])
            noisy.find_bad_by_deviation(deviation_threshold=5.0)
            noisy.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
            noisy.find_bad_by_correlation(frac_bad=SETTINGS["bad_correlation_fraction"])
            block_reports.append(dict(block=number, n_tested_samples=trimmed.shape[1],
                                      bad_channels=sorted(noisy.get_bads())))
    fractions = _flagged_duration_fractions(block_reports)
    bad = dead | {name for name, fraction in fractions.items()
                  if fraction >= SETTINGS["bad_block_persistence"]}
    report = dict(dead_channels=[name for name in channels if name in dead],
                  pyprep_blocks=block_reports, pyprep_flagged_duration_fraction=fractions)
    return [name for name in channels if name in bad], report


def _repair(inst, bads):
    inst.info["bads"] = list(bads)
    if bads:
        inst.interpolate_bads(reset_bads=True, verbose=False)
    inst.set_eeg_reference("average", verbose=False)


def _fit_ica(blocks):
    # Fixed five-second segments keep ICLabel's spectral and autocorrelation
    # inputs from straddling splices or padding. No event selection enters ICA.
    length = 5 * SFREQ
    margin = int(np.ceil(_guard(ICA_BAND) * SFREQ))
    n_segments = [max(0, (block.n_times - 2 * margin) // length) for block in blocks]
    if not sum(n_segments):
        raise ValueError("No block is long enough for ICA fitting")
    samples = np.empty((sum(n_segments), len(blocks[0].ch_names), length))
    first = 0
    for block, count in zip(blocks, n_segments):
        if count:
            fit = block.copy().filter(**ICA_BAND, verbose=False)
            fit_info = fit.info.copy()
            data = fit.get_data(start=margin, stop=margin + count * length)
            samples[first:first + count] = data.reshape(len(fit.ch_names), count, length).transpose(1, 0, 2)
            del fit, data
            first += count
    fit = mne.EpochsArray(samples, fit_info, verbose=False)
    del samples
    rank = int(mne.compute_rank(fit, verbose=False)["eeg"])
    ica = ICA(n_components=min(SETTINGS["ica_components"], rank), method="picard",
              fit_params={"ortho": False, "extended": True}, random_state=SETTINGS["ica_seed"],
              max_iter=SETTINGS["ica_max_iter"])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Picard did not converge.*", category=UserWarning)
        ica.fit(fit, decim=1, verbose=False)
    labels = label_components(fit, ica, method="iclabel")
    probabilities = [float(value) for value in labels["y_pred_proba"]]
    ica.exclude = [i for i, (label, p) in enumerate(zip(labels["labels"], probabilities))
                   if label in SETTINGS["ica_reject_labels"] and p > SETTINGS["ica_confidence"]]
    return ica, dict(ica_rank=rank, ica_components=ica.n_components_, ica_iterations=ica.n_iter_,
                     ica_converged=True, ica_labels=labels["labels"], ica_probabilities=probabilities,
                     ica_excluded=list(ica.exclude))


def _extract(block, centers, offsets):
    indices = centers[:, None] + offsets
    if not len(indices):
        return np.empty((0, len(block.ch_names), len(offsets)))
    first, last = int(indices.min()), int(indices.max()) + 1
    return block.get_data(start=first, stop=last)[:, indices - first].transpose(1, 0, 2)


def _intermittent_channels(epochs, channels, excluded):
    # A flat or nonfinite epoch can fail the QC screen, but the largest finite
    # peak-to-peak channel only diagnoses intermittent high-amplitude noise.
    ptp_uv = np.ptp(epochs, axis=2) * 1e6
    ptp_uv[~np.isfinite(ptp_uv)] = -np.inf
    worst = ptp_uv.argmax(axis=1)
    counts = np.bincount(worst[ptp_uv.max(axis=1) > SETTINGS["reject_uv"]], minlength=len(channels))
    return [name for name, count in zip(channels, counts)
            if name not in excluded and count > SETTINGS["intermittent_fraction"] * len(epochs)]


def _retention(signal_pass, boundary_eligible):
    eligible = np.count_nonzero(boundary_eligible)
    return np.count_nonzero(signal_pass & boundary_eligible) / eligible if eligible else 0.0


def _examples(raw, bounds, ledger, epochs, offsets, previews):
    # Show a retained trial if one exists and a few fixed ICA diagnostic trials.
    rows = ledger.index[ledger.boundary_eligible]
    kept = ledger.index[ledger.included]
    selected = list(dict.fromkeys([*kept[:1], *previews]))
    if not selected:
        return {}
    source_offsets = _epoch_offsets(SOURCE_SFREQ)
    raw_epochs = []
    for row in selected:
        trial = ledger.loc[row]
        start = bounds[trial.block - 1][0]
        center = start + int(np.rint((trial.onset_s - start / SOURCE_SFREQ) * SOURCE_SFREQ))
        raw_epochs.append(_extract(raw, np.array([center]), source_offsets)[0])
    clean = epochs[rows.get_indexer(selected)]
    return dict(event_ids=ledger.loc[selected, "event_id"].to_numpy(dtype=str),
                included=ledger.loc[selected, "included"].to_numpy(),
                channel_names=np.asarray(raw.ch_names), raw_times_s=source_offsets / SOURCE_SFREQ,
                times_s=offsets / SFREQ, raw_uv=np.stack(raw_epochs) * 1e6, clean_uv=clean * 1e6,
                baseline_corrected_uv=baseline_correct(clean, offsets / SFREQ) * 1e6,
                ica_event_ids=ledger.loc[list(previews), "event_id"].to_numpy(dtype=str),
                pre_ica_uv=np.stack([pair[0] for pair in previews.values()]) * 1e6,
                post_ica_uv=np.stack([pair[1] for pair in previews.values()]) * 1e6)


def process_subject(raw, trials, blocks):
    """Return the retained feature rows, complete ledger, QC summary, and compact traces."""
    if raw.info["sfreq"] != SOURCE_SFREQ or len(raw.ch_names) != 64:
        raise ValueError("Expected 64 EEG channels sampled at 2048 Hz")
    if not blocks:
        raise ValueError("No recording blocks")
    bounds = [(round(a * SOURCE_SFREQ), round(b * SOURCE_SFREQ)) for a, b in blocks]
    if (any(not 0 <= a < b <= raw.n_times for a, b in bounds)
            or any(left[1] > right[0] for left, right in zip(bounds, bounds[1:]))):
        raise ValueError("Invalid source block bounds")
    ledger = trials.reset_index(drop=True).copy()
    if (ledger.source_available & ~ledger.block.between(1, len(bounds))).any():
        raise ValueError("A source-available trial has no recording block")
    guard = _guard(LP)
    centers = _epoch_locations(ledger, bounds, guard)
    ledger["signal_pass"] = False
    ledger["included"] = False
    ledger["signal_reason"] = ""
    ledger["max_ptp_uv"] = np.nan
    ledger["worst_channel"] = ""
    channels = raw.ch_names
    offsets = _epoch_offsets(SFREQ)
    eligible = ledger.index[ledger.boundary_eligible]
    epochs = np.empty((len(eligible), len(channels), len(offsets)))
    summary = dict(subject=str(ledger.subject.iloc[0]), filter_guard_s=guard, dead_channels=[],
                   bad_channels=[], intermittent_channels=[], pyprep_blocks=[],
                   pyprep_flagged_duration_fraction={}, ica_components=0, ica_excluded=[],
                   ica_labels=[], ica_probabilities=[], ica_converged=None, ica_rank=None, ica_iterations=None)
    previews = {}
    if len(eligible):
        info = mne.create_info(channels, SFREQ, "eeg")
        info.set_montage(raw.get_montage())
        down = [_downsample_block(raw, a, b, info) for a, b in bounds]
        initial, report = _bad_channels(down)
        summary.update(bad_channels=initial, **report)
        if len(initial) <= SETTINGS["max_bad_channels"]:
            for block in down:
                _repair(block, initial)
            ica, report = _fit_ica(down)
            summary.update(report)
            for number, block in enumerate(down, start=1):
                rows = eligible[ledger.loc[eligible, "block"] == number]
                if len(rows) and len(previews) < 4:
                    before = _extract(block, centers[rows[:1]], offsets)[0]
                else:
                    before = None
                if ica.exclude:
                    ica.apply(block, verbose=False)
                if before is not None:
                    after = _extract(block, centers[rows[:1]], offsets)[0]
                    previews[rows[0]] = (before, after)
                block.filter(**LP, verbose=False)
                epochs[eligible.get_indexer(rows)] = _extract(block, centers[rows], offsets)
            del down
            intermittent = _intermittent_channels(epochs, channels, initial)
            bad_set = set(initial + intermittent)
            total = [name for name in channels if name in bad_set]
            summary.update(bad_channels=total, intermittent_channels=intermittent)
            if intermittent and len(total) <= SETTINGS["max_bad_channels"]:
                repaired = mne.EpochsArray(epochs, info, tmin=offsets[0] / SFREQ, verbose=False)
                _repair(repaired, total)
                epochs = repaired.get_data(copy=False)
            else:
                epochs -= epochs.mean(axis=1, keepdims=True)
            assessed = quality(epochs)
            ledger.loc[eligible, "signal_reason"] = assessed.signal_reason.to_numpy()
            ledger.loc[eligible, "signal_pass"] = (assessed.signal_reason == "").to_numpy()
            ledger.loc[eligible, "max_ptp_uv"] = assessed.max_ptp_uv.to_numpy()
            ledger.loc[eligible, "worst_channel"] = np.asarray(channels)[assessed.worst_channel_index]
    retention = _retention(ledger.signal_pass, ledger.boundary_eligible)
    participant_reason = ""
    if len(summary["bad_channels"]) > SETTINGS["max_bad_channels"]:
        participant_reason = "too_many_bad_channels"
    elif retention < SETTINGS["min_retention"]:
        participant_reason = "insufficient_signal_retention"
    failed_signal = ledger.signal_reason != ""
    ledger.loc[failed_signal, "reason"] = "signal:" + ledger.loc[failed_signal, "signal_reason"]
    eligible_unclassified = ledger.boundary_eligible & ~failed_signal
    if participant_reason:
        ledger.loc[eligible_unclassified, "reason"] = "participant:" + participant_reason
    else:
        ledger.loc[eligible_unclassified, "included"] = True
    accepted = ledger.index[ledger.included]
    metadata = ledger.loc[accepted, trials.columns].reset_index(drop=True)
    values = pd.DataFrame(columns=PREDICTORS, dtype=float)
    if len(accepted):
        corrected = baseline_correct(epochs[eligible.get_indexer(accepted)], offsets / SFREQ)
        values = feature_table(corrected, offsets / SFREQ, channels)
    features = pd.concat([metadata, values], axis=1)
    examples = _examples(raw, bounds, ledger, epochs, offsets, previews)
    summary.update(n_candidates=len(ledger), n_source_available=int(ledger.source_available.sum()),
                   n_edge_eligible=len(eligible), n_signal_pass=int(ledger.signal_pass.sum()),
                   n_included=len(accepted), retention=retention, participant_reason=participant_reason)
    return features, ledger, summary, examples
