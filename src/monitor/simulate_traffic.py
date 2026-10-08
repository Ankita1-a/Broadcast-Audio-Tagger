"""Step 12a: Send realistic traffic to the running API, with a known right answer.

Builds recordings from TEST-set clips (never used for training), optionally
degrades them (e.g. phone quality), sends each to POST /tag, and compares the
returned timeline with the true labels. In production you rarely have true labels;
here they let us check whether the drift monitor flags situations where accuracy
really drops.

Examples (API must be running, e.g. `docker compose up`):
    python src/monitor/simulate_traffic.py --scenario normal --n 40 --fresh
    python src/monitor/simulate_traffic.py --scenario phone --n 30 --fresh

--fresh archives the current request log first, so the log only holds this batch.
"""
import argparse
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import requests
import soundfile as sf
import yaml
from scipy.signal import butter, sosfilt
from tqdm import tqdm

CONFIG = Path("configs/monitor.yaml")
MANIFEST = Path("data/processed/windows.csv")
REQUEST_LOG = Path("logs/requests.jsonl")
RUNS_LOG = Path("monitoring/traffic_runs.jsonl")
LABELS = ["speech", "music", "speech_over_music", "applause", "laughter", "noise"]
SR = 16000


def rotate_log() -> None:
    """Move the live request log into logs/archive/ so the next batch starts clean."""
    if REQUEST_LOG.exists():
        archive = REQUEST_LOG.parent / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(REQUEST_LOG), archive / f"requests_{stamp}.jsonl")
        print(f"Archived the previous request log to {archive}/requests_{stamp}.jsonl")


def make_recording(test: pd.DataFrame, cfg: dict, weights: np.ndarray, rng) -> tuple[np.ndarray, list]:
    """Join a few same-label runs of 2 s test clips. Returns audio and the true label per window."""
    t = cfg["traffic"]
    clips, truth = [], []
    for _ in range(t["segments_per_recording"]):
        label = rng.choice(LABELS, p=weights)
        n = int(rng.integers(t["clips_per_segment"][0], t["clips_per_segment"][1] + 1))
        pool = test[test.label == label]
        for path in pool.path.iloc[rng.choice(len(pool), size=n, replace=False)]:
            clips.append(sf.read(path, dtype="float32")[0])
            truth.append(label)
    return np.concatenate(clips), truth


def phone_quality(audio: np.ndarray, p: dict, rng) -> tuple[np.ndarray, int]:
    """Simulate a phone line: 300-3400 Hz band, 8 kHz, quieter, with hiss."""
    sos = butter(4, p["band_hz"], btype="band", fs=SR, output="sos")
    x = sosfilt(sos, audio) * 10 ** (p["gain_db"] / 20)
    noise_rms = np.sqrt(np.mean(x ** 2)) / 10 ** (p["noise_snr_db"] / 20)
    x = x + rng.normal(0, noise_rms, len(x))
    x = librosa.resample(x.astype(np.float32), orig_sr=SR, target_sr=p["sample_rate"])
    return np.clip(x, -1, 1).astype(np.float32), p["sample_rate"]


def predicted_per_window(segments: list, n: int) -> list:
    """Label of each 2 s window, read from the returned segments (at the window centre)."""
    out = []
    for i in range(n):
        centre = 2.0 * i + 1.0
        seg = next((s for s in segments if s["start"] <= centre < s["end"]), None)
        out.append(seg["label"] if seg else "none")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--n", type=int, default=30, help="number of recordings to send")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fresh", action="store_true", help="archive the request log first")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    scen = cfg["traffic"]["scenarios"][args.scenario]
    url = cfg["api_url"]
    try:
        model = requests.get(f"{url}/health", timeout=5).json()["model"]
    except Exception:
        raise SystemExit(f"API not reachable at {url}. Start it first (docker compose up).")

    if args.fresh:
        rotate_log()

    w = scen.get("label_weights", {})
    weights = np.array([w.get(label, 1.0) for label in LABELS], dtype=float)
    weights /= weights.sum()

    m = pd.read_csv(MANIFEST)
    test = m[m.split == "test"]
    rng = np.random.default_rng(args.seed)

    truth_all, pred_all = [], []
    for _ in tqdm(range(args.n), desc=f"Sending '{args.scenario}' recordings"):
        audio, truth = make_recording(test, cfg, weights, rng)
        sr = SR
        if scen["degrade"] == "phone":
            audio, sr = phone_quality(audio, cfg["phone"], rng)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        r = requests.post(f"{url}/tag", files={"file": ("rec.wav", buf.getvalue(), "audio/wav")},
                          timeout=300)
        r.raise_for_status()
        truth_all += truth
        pred_all += predicted_per_window(r.json()["segments"], len(truth))

    truth_all, pred_all = np.array(truth_all), np.array(pred_all)
    accuracy = float((truth_all == pred_all).mean())
    per_class = {label: round(float((pred_all[truth_all == label] == label).mean()), 3)
                 for label in LABELS if (truth_all == label).any()}

    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "scenario": args.scenario, "n_recordings": args.n, "seed": args.seed,
                            "model_version": model.get("version"), "window_accuracy": round(accuracy, 4),
                            "recall_per_class": per_class}) + "\n")

    print(f"\nScenario '{args.scenario}': {args.n} recordings, {len(truth_all)} windows")
    print(f"Window accuracy vs true labels: {accuracy:.3f}")
    print("Recall per class: " + ", ".join(f"{k} {v:.2f}" for k, v in per_class.items()))
    print(f"Requests were logged to {REQUEST_LOG}")


if __name__ == "__main__":
    main()