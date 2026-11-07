"""Fixed analysis choices, in seconds, hertz, and microvolts as noted."""

SFREQ = 256
SEED = 97
BASELINE = (-0.100, 0.0)
EPOCH = (-0.100, 0.400)  # Half-open on the aligned sample grid; alignment error <= 1.95 ms.
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

SETTINGS = {
    "sfreq": SFREQ,
    "analysis_band_hz": [0.1, 40.0],
    "notch_hz": [60.0, 120.0],
    "ica_band_hz": [1.0, 100.0],
    "ica_components": 30,
    "ica_seed": SEED,
    "ica_max_iter": 500,
    "ica_confidence": 0.8,
    "ica_reject_labels": ["eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"],
    "dead_mad_ratio": 0.1,
    "bad_correlation_fraction": 0.2,
    "bad_block_persistence": 0.5,
    "intermittent_fraction": 0.05,
    "max_bad_channels": 6,
    "reject_uv": 150.0,
    "flat_uv": 1.0,
    "min_retention": 0.6,
    "edge_margin_s": 1.0,
    "baseline_s": BASELINE,
    "epoch_s": EPOCH,
    "regions": ROIS,
    "windows_s": WINDOWS,
}
