"""Unit tests for the timeline logic (no model needed)."""
import numpy as np

from serve.timeline import LABELS, SR, make_windows, normalise, rms_db, smooth, tag

CFG = {"hop_s": 2.0, "smooth_windows": 3, "batch_size": 4, "low_confidence": 0.6}


def tone(seconds: float, level: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (level * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def fixed_model(label: str):
    """A stand-in model that always predicts one label."""
    def predict(x):
        p = np.full((len(x), len(LABELS)), 0.02, dtype=np.float32)
        p[:, LABELS.index(label)] = 0.9
        return p
    return predict


def test_windows_cover_audio_and_pad_last_piece():
    starts, w = make_windows(tone(5.0), hop_s=2.0)
    assert list(starts) == [0.0, 2.0, 4.0]          # last 1 s piece kept (>= 0.5 s), padded
    assert w.shape == (3, 32000)


def test_short_tail_is_ignored():
    starts, _ = make_windows(tone(4.3), hop_s=2.0)
    assert list(starts) == [0.0, 2.0]               # 0.3 s tail is too short


def test_normalise_reaches_target_loudness():
    assert abs(rms_db(normalise(tone(2.0, 0.001), -25)) - (-25)) < 0.5


def test_silence_skips_the_model_and_segments_merge():
    audio = np.concatenate([tone(4.0), np.zeros(4 * SR, dtype=np.float32), tone(2.0)])
    out = tag(audio, fixed_model("music"), CFG, silence_db=-55, target_db=-25)
    assert [s["label"] for s in out["segments"]] == ["music", "silence", "music"]
    assert out["segments"][1] == {"start": 4.0, "end": 8.0, "label": "silence", "confidence": 1.0}
    assert out["seconds_per_label"] == {"music": 6.0, "silence": 4.0}


def test_smoothing_removes_a_single_window_flicker():
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.9, 0.1]])
    smoothed = smooth(probs, np.ones(3, dtype=bool), k=3)
    assert smoothed.argmax(axis=1).tolist() == [0, 0, 0]