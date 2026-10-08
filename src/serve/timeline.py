"""Turn a recording of any length into a timeline of labelled segments.

Steps (same preparation as training):
  1. Decode audio to 16 kHz mono.
  2. Cut into 2 s windows.
  3. Windows quieter than the silence level are labelled "silence" without the model.
  4. Every other window is loudness-normalised (as in training) and sent to the model.
  5. Probabilities are averaged with neighbouring windows to avoid flicker.
  6. Consecutive windows with the same label are merged into segments.
"""
import io

import librosa
import numpy as np
import soundfile as sf

LABELS = ["speech", "music", "speech_over_music", "applause", "laughter", "noise"]
SR = 16000
WINDOW_S = 2.0
WIN = int(SR * WINDOW_S)
MIN_TAIL_S = 0.5            # a last piece shorter than this is ignored


def load_audio(data: bytes) -> tuple[np.ndarray, dict]:
    """Bytes of a wav/flac/ogg/mp3 file -> 16 kHz mono float32, plus info about the original."""
    info = sf.info(io.BytesIO(data))
    audio, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True)
    return audio.astype(np.float32), {"original_sample_rate": info.samplerate,
                                      "original_channels": info.channels,
                                      "format": info.format}


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10))


def normalise(x: np.ndarray, target_db: float) -> np.ndarray:
    """Same loudness normalisation as src/data/build_dataset.py."""
    y = x * 10 ** ((target_db - rms_db(x)) / 20)
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * 0.99 / peak
    return y.astype(np.float32)


def make_windows(audio: np.ndarray, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Cut audio into 2 s windows every hop_s seconds. The last partial window is zero-padded."""
    hop = int(SR * hop_s)
    starts = list(range(0, max(len(audio) - WIN, 0) + 1, hop))
    last_end = starts[-1] + WIN if starts else 0
    if len(audio) - last_end >= MIN_TAIL_S * SR or not starts:
        starts.append(last_end if starts else 0)
    windows = np.zeros((len(starts), WIN), dtype=np.float32)
    for i, s in enumerate(starts):
        piece = audio[s:s + WIN]
        windows[i, :len(piece)] = piece
    return np.array(starts) / SR, windows


def smooth(probs: np.ndarray, keep: np.ndarray, k: int) -> np.ndarray:
    """Average each window's probabilities with its k-window neighbourhood (non-silent windows only)."""
    if k <= 1:
        return probs
    kernel = np.ones(k)
    weighted = np.stack([np.convolve(probs[:, c] * keep, kernel, mode="same")
                         for c in range(probs.shape[1])], axis=1)
    counts = np.convolve(keep.astype(float), kernel, mode="same")[:, None]
    return np.where(counts > 0, weighted / np.maximum(counts, 1e-9), probs)


def tag(audio: np.ndarray, predict, cfg: dict, silence_db: float, target_db: float) -> dict:
    """predict: function taking (n, 32000) float32 windows and returning (n, 6) probabilities."""
    duration = len(audio) / SR
    starts, windows = make_windows(audio, cfg["hop_s"])
    levels = np.array([rms_db(w) for w in windows])
    keep = levels >= silence_db

    probs = np.zeros((len(windows), len(LABELS)), dtype=np.float32)
    idx = np.where(keep)[0]
    for i in range(0, len(idx), cfg["batch_size"]):
        batch = idx[i:i + cfg["batch_size"]]
        probs[batch] = predict(np.stack([normalise(windows[j], target_db) for j in batch]))
    probs = smooth(probs, keep, cfg["smooth_windows"])

    labels = np.where(keep, np.array(LABELS)[probs.argmax(axis=1)], "silence")
    conf = np.where(keep, probs.max(axis=1), 1.0)

    segments, i = [], 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        end = starts[j + 1] if j + 1 < len(labels) else duration
        segments.append({"start": round(float(starts[i]), 2), "end": round(float(min(end, duration)), 2),
                         "label": str(labels[i]), "confidence": round(float(conf[i:j + 1].mean()), 3)})
        i = j + 1

    seconds = {}
    for s in segments:
        seconds[s["label"]] = round(seconds.get(s["label"], 0) + s["end"] - s["start"], 2)

    return {
        "duration_s": round(duration, 2),
        "n_windows": int(len(windows)),
        "segments": segments,
        "seconds_per_label": seconds,
        # Extra numbers for monitoring (not part of the user-facing answer).
        "_stats": {"mean_rms_db": round(float(levels.mean()), 1),
                   "silent_share": round(float((~keep).mean()), 3),
                   "label_share": {lab: round(float((labels == lab).mean()), 3)
                                   for lab in LABELS + ["silence"]},
                   "mean_confidence": round(float(probs.max(axis=1)[keep].mean()), 3) if keep.any() else None,
                   "low_confidence_share": round(float((probs.max(axis=1)[keep] < cfg["low_confidence"]).mean()), 3)
                   if keep.any() else None},
    }