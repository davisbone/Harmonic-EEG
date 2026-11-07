"""Simple averages turn an event's waveform into 21 named predictors."""
import numpy as np
import pandas as pd

from .config import BASELINE, PREDICTORS, ROIS, SETTINGS, WINDOWS


def baseline_correct(epochs, times):
    """epochs: events x channels x samples, in volts; times: relative seconds."""
    baseline = (times >= BASELINE[0]) & (times < BASELINE[1])
    if not baseline.any():
        raise ValueError("No pre-event samples for baseline correction")
    return epochs - epochs[:, :, baseline].mean(axis=2, keepdims=True)


def feature_table(epochs, times, channels):
    """Input is baseline-corrected; output is mean voltage in microvolts."""
    values = {}
    for region, names in ROIS.items():
        indices = [channels.index(name) for name in names]
        regional = epochs[:, indices, :].mean(axis=1)
        for window, (start, stop) in WINDOWS.items():
            samples = (times >= start) & (times < stop)
            if not samples.any():
                raise ValueError(f"No samples in feature window {window}")
            values[f"erp_{region}_{window}_uv"] = regional[:, samples].mean(axis=1) * 1e6
    return pd.DataFrame(values, columns=PREDICTORS)


def quality(epochs):
    """Check every EEG channel, not only those used to construct features."""
    finite = np.isfinite(epochs).all(axis=(1, 2))
    ptp_uv = np.ptp(epochs, axis=2) * 1e6
    high = (ptp_uv > SETTINGS["reject_uv"]).any(axis=1)
    flat = (ptp_uv < SETTINGS["flat_uv"]).any(axis=1)
    reasons = [";".join(reason for reason, applies in
                       [("nonfinite", not f), ("amplitude", h), ("flat", z)] if applies)
               for f, h, z in zip(finite, high, flat)]
    return pd.DataFrame({"signal_reason": reasons, "max_ptp_uv": ptp_uv.max(axis=1),
                         "worst_channel_index": ptp_uv.argmax(axis=1)})
