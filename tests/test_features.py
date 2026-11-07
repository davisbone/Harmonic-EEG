import numpy as np

from eeg_pipeline.config import PREDICTORS, ROIS
from eeg_pipeline.features import baseline_correct, feature_table, quality


def test_known_window_means_in_microvolts():
    channels = list(dict.fromkeys(name for names in ROIS.values() for name in names))
    times = np.arange(-25, 103) / 256
    signal = np.full(len(times), 2e-6)
    signal[(times >= .080) & (times < .150)] = 4e-6
    signal[(times >= .150) & (times < .250)] = 6e-6
    signal[(times >= .250) & (times < .400)] = 8e-6
    epochs = np.tile(signal, (2, len(channels), 1))
    corrected = baseline_correct(epochs, times)
    result = feature_table(corrected, times, channels)
    assert list(result) == PREDICTORS
    np.testing.assert_allclose(result.to_numpy(), np.tile([2., 4., 6.], (2, 7)), atol=1e-12)


def test_quality_checks_non_feature_channels_and_nonfinite():
    epochs = np.tile(np.linspace(-2e-6, 2e-6, 128), (4, 64, 1))
    epochs[1, 63, 0] = 200e-6
    epochs[2, 63, :] = 0
    epochs[3, 63, 0] = np.nan
    assert quality(epochs).signal_reason.tolist() == ["", "amplitude", "flat", "nonfinite"]


