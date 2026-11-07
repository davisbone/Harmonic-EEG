import hashlib
import io

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline import source


def test_trial_labels_keep_condition_independent_of_response():
    events = pd.DataFrame({
        "onset": [1, 1.4, 1.8, 1.9, 2, 3, 3.4, 3.8, 5, 5.4, 5.8],
        "value": [
            "two", "four6_Sincorrect", "one", "Incorrect - Standard!", "boundary",
            "two", "fiveRoot_deviantFcorrect_F", "one",
            "two", "five6_deviantEcorrect_E", "one",
        ],
    })
    trials = source.parse_events(events, "sub-001")
    assert trials["event_id"].tolist() == [f"sub-001_trial-{index:04d}" for index in range(1, 4)]
    assert trials["block"].tolist() == [1, 2, 2]
    assert trials["onset_s"].tolist() == [1.4, 3.4, 5.4]
    assert trials["condition"].tolist() == ["standard", "function_dev", "exemplar_dev"]
    assert trials["target"].tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="invalid chord sequence"):
        source.parse_events(events.assign(value=events["value"].replace({
            "fiveRoot_deviantFcorrect_F": "fiveRoot_deviantFcorrect_E",
        })), "sub-001")


def test_embedded_latency_is_one_based():
    events = pd.DataFrame({"onset": [0, 1.5 / source.SOURCE_RATE], "value": ["two", "boundary"]})
    embedded = [{"type": "two", "latency": 1}, {"type": "boundary", "latency": 2.5}]
    source._check_embedded_events("sub-001", events, embedded)
    embedded[1]["latency"] = 3.5
    with pytest.raises(ValueError, match="timestamps differ"):
        source._check_embedded_events("sub-001", events, embedded)


def test_complete_frames_and_microvolts(tmp_path):
    fdt = source.source_path(tmp_path, "sub-001", "eeg.fdt")
    fdt.parent.mkdir(parents=True)
    samples = np.arange(2 * source.CHANNEL_COUNT, dtype="<f4").reshape(2, -1)
    fdt.write_bytes(samples.tobytes() + b"\x00" * 108)
    frames, trailing = source._frame_geometry(fdt.stat().st_size, expected_frames=3)
    assert (frames, trailing) == (2, 108)
    np.testing.assert_array_equal(source._read_fdt(fdt, frames), samples.T.astype(float) * 1e-6)
    with pytest.raises(ValueError, match="exceed"):
        source._frame_geometry(fdt.stat().st_size, expected_frames=2)


def test_block_cuts_use_next_sample_and_stop_at_prefix():
    rate = source.SOURCE_RATE
    boundaries = np.array([3.5, 9.5, 19.5]) / rate
    intervals = source._block_intervals(boundaries, available_frames=13)
    assert intervals == [(0, 4 / rate), (4 / rate, 10 / rate), (10 / rate, 13 / rate)]


def test_only_two_known_long_prefixes_are_eligible():
    assert source._source_status("sub-001", 0.92, True) == "verified_prefix"
    assert source._source_status("sub-030", 0.93, True) == "verified_prefix"
    assert source._source_status("sub-003", 0.017, False) == "severely_truncated"
    assert source._source_status("sub-002", 1.0, True) == "complete"
    with pytest.raises(ValueError, match="coverage"):
        source._source_status("sub-002", 0.93, True)


def test_verify_missing_and_wrong_bytes(tmp_path):
    path = tmp_path / "tiny"
    entry = {
        "size": 3,
        "checksum_algorithm": "git",
        "checksum": hashlib.sha1(b"blob 3\0abc").hexdigest(),
    }
    with pytest.raises(FileNotFoundError, match="Missing pinned source"):
        source._verify_file(path, entry)
    path.write_bytes(b"abc")
    source._verify_file(path, entry)
    path.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="checksum"):
        source._verify_file(path, entry)


def test_download_only_publishes_verified_files(tmp_path, monkeypatch):
    entry = {
        "path": "fixture.txt",
        "size": 3,
        "checksum_algorithm": "md5",
        "checksum": hashlib.md5(b"abc").hexdigest(),
        "bytes_url": "https://example.invalid/fixture.txt",
    }
    monkeypatch.setattr(source, "_selection", lambda subjects: ([entry], {}))
    monkeypatch.setattr(source, "urlopen", lambda url, timeout: io.BytesIO(b"xyz"))
    with pytest.raises(ValueError, match="checksum"):
        source.download(tmp_path)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(source, "urlopen", lambda url, timeout: io.BytesIO(b"abc"))
    source.download(tmp_path)
    assert (tmp_path / "fixture.txt").read_bytes() == b"abc"


def test_manifest_selection_keeps_release_metadata():
    root, entries = source._selection(None)
    assert len(entries) == 40
    assert len(root) + sum(map(len, entries.values())) == 328
    selected_root, selected_entries = source._selection(["sub-001"])
    assert root == selected_root
    assert selected_entries == {"sub-001": entries["sub-001"]}
    with pytest.raises(ValueError, match="unique dataset IDs"):
        source._selection(["sub-001", "sub-001"])
