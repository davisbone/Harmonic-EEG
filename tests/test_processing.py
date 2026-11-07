from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("pyprep")
pytest.importorskip("mne_icalabel")

from eeg_pipeline import processing as p
from eeg_pipeline.config import EPOCH, PREDICTORS, SETTINGS
from eeg_pipeline.features import quality


def test_filtering_and_resampling_do_not_cross_splice():
    sfreq = p.SOURCE_SFREQ
    samples = 36 * sfreq + 5  # The block size need not be divisible by eight.
    rng = np.random.default_rng(97)
    first = rng.normal(size=(1, samples)) * 1e-6
    info = mne.create_info(["Cz"], sfreq, "eeg")
    down_info = mne.create_info(["Cz"], p.SFREQ, "eeg")

    def process_first(second):
        raw = mne.io.RawArray(np.concatenate([first, second], axis=1), info, verbose=False)
        block = p._downsample_block(raw, 0, samples, down_info)
        block.filter(**p.LP, verbose=False)
        assert block.n_times == (samples + p.DOWNSAMPLE - 1) // p.DOWNSAMPLE
        return block.get_data()

    np.testing.assert_array_equal(process_first(np.zeros_like(first)),
                                  process_first(np.full_like(first, 1.0)))


def test_entire_half_open_epoch_must_clear_actual_guard():
    guard = p._guard(p.LP)
    expected = SETTINGS["edge_margin_s"] + (len(p.RESAMPLE_TAPS) - 1) / (2 * p.SOURCE_SFREQ)
    expected += sum((len(mne.filter.create_filter(None, p.SFREQ, **spec, verbose=False)) - 1)
                    / (2 * p.SFREQ) for spec in (p.HP, p.NOTCH, p.LP))
    assert guard == expected
    offsets = p._epoch_offsets(p.SFREQ)
    assert len(offsets) == 128
    assert (offsets / p.SFREQ >= EPOCH[0]).all() and (offsets / p.SFREQ < EPOCH[1]).all()

    sfreq = p.SOURCE_SFREQ
    ledger = pd.DataFrame(dict(source_available=[True, True, True, True, True, False],
                               block=[1, 1, 1, 1, 2, 2],
                               onset_s=[guard + 0.05, guard + 0.2, 100 - guard - 0.3,
                                        99.9, 100 + guard + 0.2, 190]))
    centers = p._epoch_locations(ledger, [(0, 100 * sfreq), (100 * sfreq, 180 * sfreq)], guard)
    assert ledger.reason.tolist() == ["filter_edge_guard", "", "filter_edge_guard",
                                       "block_boundary_overlap", "", "source_unavailable"]
    assert centers[1] == centers[4] == round((guard + 0.2) * p.SFREQ)


def test_prep_persistence_uses_tested_duration():
    reports = [dict(n_tested_samples=n, bad_channels=bads)
               for n, bads in [(1, ["Fp1", "Fpz"]), (4, ["Fp1"]), (5, ["Fz"])]]
    assert p._flagged_duration_fractions(reports) == {"Fp1": 0.5, "Fpz": 0.1, "Fz": 0.5}


def test_all_channel_quality_and_one_pass_interpolation():
    montage = mne.channels.make_standard_montage("biosemi64")
    names = montage.ch_names
    fp1 = names.index("Fp1")
    offsets = p._epoch_offsets(p.SFREQ)
    signal = np.sin(2 * np.pi * 5 * offsets / p.SFREQ) * 5e-6
    epochs = np.broadcast_to(signal, (20, len(names), len(offsets))).copy()
    epochs[0, fp1, :] *= 30
    epochs[1, fp1, :] *= 30
    assert quality(epochs).signal_reason.tolist()[:3] == ["amplitude", "amplitude", ""]
    assert p._intermittent_channels(epochs[1:], names, []) == ["Fp1"]
    single = epochs.copy()
    single[1, fp1, :] /= 30
    assert p._intermittent_channels(single, names, []) == []  # Exactly 5% is below the strict rule.
    assert p._intermittent_channels(epochs, names, ["Fp1"]) == []
    info = mne.create_info(names, p.SFREQ, "eeg").set_montage(montage)
    inst = mne.EpochsArray(epochs, info, verbose=False)
    p._repair(inst, ["Fp1"])
    np.testing.assert_allclose(inst.get_data()[:, fp1], 0, atol=1e-15)
    np.testing.assert_allclose(inst.get_data().mean(axis=1), 0, atol=1e-15)


def test_retention_uses_source_and_edge_eligible_denominator(monkeypatch):
    montage = mne.channels.make_standard_montage("biosemi64")
    names = montage.ch_names
    seconds = 62
    info = mne.create_info(names, p.SOURCE_SFREQ, "eeg").set_montage(montage)
    raw = mne.io.RawArray(np.zeros((64, seconds * p.SOURCE_SFREQ)), info, verbose=False)
    times = np.arange(seconds * p.SFREQ) / p.SFREQ
    # Mock ICA and persistent screening. A flat interval affects only one trial,
    # without triggering the high-amplitude intermittent repair.
    data = np.linspace(-4, 4, 64)[:, None] * np.sin(2 * np.pi * 7 * times) * 1e-6
    data[:, (times >= 32.5) & (times < 34)] = 0
    data += np.where(np.arange(64) % 2, 1, -1)[:, None] * 2e-6 * np.sin(2 * np.pi * 9 * times)
    data[:, (times >= 32.5) & (times < 34)] = 0
    monkeypatch.setattr(p, "_downsample_block", lambda raw, a, b, info: mne.io.RawArray(data.copy(), info, verbose=False))
    monkeypatch.setattr(p, "_bad_channels", lambda blocks: ([], {}))
    monkeypatch.setattr(p, "_fit_ica", lambda blocks: (SimpleNamespace(exclude=[]), {}))
    trials = pd.DataFrame(dict(subject="test", trial=np.arange(6), event_id=list("abcdef"),
                               block=[1, 1, 1, 1, 1, 2], onset_s=[1, 25, 30, 33, 60, 70],
                               condition="standard", target=0, source_available=[True] * 5 + [False]))
    features, ledger, qc, example = p.process_subject(raw, trials, [(0, seconds)])
    assert qc["n_edge_eligible"] == 3
    assert qc["n_signal_pass"] == qc["n_included"] == 2
    assert qc["retention"] == 2 / 3
    assert ledger.reason.tolist() == ["filter_edge_guard", "", "", "signal:flat",
                                       "filter_edge_guard", "source_unavailable"]
    assert features.event_id.tolist() == ["b", "c"]
    assert features.columns.tolist() == trials.columns.tolist() + PREDICTORS
    assert example["event_ids"][0] == "b"
    assert example["baseline_corrected_uv"].shape[1:] == (64, 128)


def test_unassessed_trials_retain_source_edge_and_participant_reasons(monkeypatch):
    montage = mne.channels.make_standard_montage("biosemi64")
    names = montage.ch_names
    seconds = 62
    info = mne.create_info(names, p.SOURCE_SFREQ, "eeg").set_montage(montage)
    raw = mne.io.RawArray(np.zeros((64, seconds * p.SOURCE_SFREQ)), info, verbose=False)
    trials = pd.DataFrame(dict(subject="test", trial=[1, 2, 3], event_id=list("abc"), block=[1, 1, 2],
                               onset_s=[1, 25, 80], condition="standard", target=0,
                               source_available=[True, True, False]))
    monkeypatch.setattr(p, "_downsample_block", lambda *args: None)
    monkeypatch.setattr(p, "_bad_channels", lambda blocks: (names[:7], {}))
    features, ledger, qc, example = p.process_subject(raw, trials, [(0, seconds)])
    assert features.empty and not example
    assert qc["participant_reason"] == "too_many_bad_channels"
    assert qc["n_edge_eligible"] == 1 and qc["n_included"] == 0
    assert ledger.reason.tolist() == ["filter_edge_guard", "participant:too_many_bad_channels", "source_unavailable"]
    features, ledger, qc, example = p.process_subject(raw, trials.drop(index=1), [(0, seconds)])
    assert features.empty and not example
    assert qc["participant_reason"] == "insufficient_signal_retention"
    assert qc["n_edge_eligible"] == qc["n_included"] == 0
    assert ledger.reason.tolist() == ["filter_edge_guard", "source_unavailable"]
