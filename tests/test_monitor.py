"""Checks that the drift report flags changed features and ignores unchanged ones."""
import numpy as np
import pandas as pd
import yaml

from monitor.drift_report import run_drift
from monitor.simulate_traffic import phone_quality, predicted_per_window

CFG = yaml.safe_load(open("configs/monitor.yaml"))


def fake_log(n, rng, sample_rate=16000, rms=-30.0, conf=0.9):
    shares = rng.dirichlet(np.ones(6), size=n)
    labels = ["speech", "music", "speech_over_music", "applause", "laughter", "noise"]
    df = pd.DataFrame({"original_sample_rate": sample_rate, "original_channels": 1,
                       "duration_s": rng.uniform(16, 24, n), "mean_rms_db": rng.normal(rms, 2, n),
                       "silent_share": rng.uniform(0, 0.05, n),
                       "mean_confidence": rng.normal(conf, 0.03, n),
                       "low_confidence_share": rng.uniform(0, 0.1, n)})
    for i, label in enumerate(labels):
        df[f"label_share_{label}"] = shares[:, i]
    return df


def test_detects_phone_like_drift_only_in_changed_features():
    rng = np.random.default_rng(0)
    ref = fake_log(40, rng)
    cur = fake_log(30, rng, sample_rate=8000, rms=-42, conf=0.75)
    table, _ = run_drift(ref, cur, CFG)
    drifted = set(table[table.drifted].feature)
    assert {"original_sample_rate", "mean_rms_db", "mean_confidence"} <= drifted
    # Each test has a 5% false-alarm rate, so 0-2 of the 10 unchanged features may flag by chance.
    assert len(drifted - {"original_sample_rate", "mean_rms_db", "mean_confidence"}) <= 2


def test_phone_quality_is_8khz_and_quieter():
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(32000)).astype(np.float32)
    out, sr = phone_quality(audio, CFG["phone"], rng)
    assert sr == 8000 and len(out) == 16000
    assert np.sqrt(np.mean(out ** 2)) < np.sqrt(np.mean(audio ** 2))


def test_window_labels_read_from_segments():
    segs = [{"start": 0.0, "end": 4.0, "label": "speech"}, {"start": 4.0, "end": 6.0, "label": "music"}]
    assert predicted_per_window(segs, 3) == ["speech", "speech", "music"]