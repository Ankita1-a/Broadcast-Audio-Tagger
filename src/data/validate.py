"""Step 5: Check the built dataset before any model sees it.

Three groups of checks:
  1. Table checks (Pandera): every row in windows.csv has valid values.
  2. Dataset checks: no leakage between splits, every class present in every split.
  3. Audio checks: every file is 16 kHz, mono, exactly 2 seconds, not silent, not clipped.

Writes a summary to reports/data_validation.json.
Exits with an error code if anything fails, so later pipelines (DVC, CI) stop.

Run from the project root:
    python src/data/validate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

try:                                   # newer Pandera versions
    import pandera.pandas as pa
except ImportError:                    # older Pandera versions
    import pandera as pa

CONFIG = Path("configs/data.yaml")
MANIFEST = Path("data/processed/windows.csv")
REPORT = Path("reports/data_validation.json")

LABELS = ["speech", "music", "speech_over_music", "applause", "laughter", "noise"]
SPLITS = ["train", "val", "test"]
MIN_PER_CLASS_AND_SPLIT = 15


# ---------- 1. table checks ----------

def manifest_schema(cfg: dict) -> pa.DataFrameSchema:
    lo, hi = cfg["speech_over_music"]["snr_db"]
    file_exists = pa.Check(lambda s: s.map(lambda p: Path(p).is_file()), error="audio file missing")
    return pa.DataFrameSchema({
        "window_id":      pa.Column(unique=True),
        "path":           pa.Column(checks=file_exists),
        "label":          pa.Column(checks=pa.Check.isin(LABELS)),
        "split":          pa.Column(checks=pa.Check.isin(SPLITS)),
        "source_dataset": pa.Column(checks=pa.Check.isin(["musan", "esc50", "musan_mix"])),
        "recording_id":   pa.Column(nullable=False),
        "rms_db":         pa.Column(float, coerce=True, checks=pa.Check.ge(cfg["silence_db"])),
        "snr_db":         pa.Column(float, coerce=True, nullable=True,
                                    checks=pa.Check.in_range(lo, hi)),
    })


def check_table(m: pd.DataFrame, cfg: dict) -> list[str]:
    try:
        manifest_schema(cfg).validate(m, lazy=True)   # lazy = collect all errors, not just the first
        return []
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        return [f"{r.column}: {r.check} (e.g. {r.failure_case})"
                for r in fc.drop_duplicates(["column", "check"]).itertuples()]


# ---------- 2. dataset checks ----------

def check_dataset(m: pd.DataFrame) -> list[str]:
    problems = []
    base = m[m.label != "speech_over_music"]
    mixes = m[m.label == "speech_over_music"]

    # No recording in more than one split.
    leaks = base.groupby("recording_id").split.nunique().gt(1)
    if leaks.any():
        problems.append(f"{leaks.sum()} recordings appear in more than one split")

    # Each mix must use speech and music from its own split.
    split_of = base.drop_duplicates("recording_id").set_index("recording_id").split
    for r in mixes.itertuples():
        parts = r.recording_id.split("+")
        if any(split_of.get(p) != r.split for p in parts):
            problems.append(f"mix {r.window_id} uses audio from another split")
            break

    # Every class has enough examples in every split.
    counts = pd.crosstab(m.label, m.split).reindex(index=LABELS, columns=SPLITS, fill_value=0)
    for label in LABELS:
        for split in SPLITS:
            if counts.loc[label, split] < MIN_PER_CLASS_AND_SPLIT:
                problems.append(f"{label}/{split} has only {counts.loc[label, split]} windows")
    return problems


# ---------- 3. audio checks ----------

def check_audio(m: pd.DataFrame, cfg: dict) -> list[str]:
    sr = cfg["sample_rate"]
    n_frames = int(cfg["window_s"] * sr)
    problems = []
    for p in tqdm(m.path, desc="Checking audio files"):
        info = sf.info(p)
        if info.samplerate != sr or info.channels != 1 or info.frames != n_frames:
            problems.append(f"{p}: {info.samplerate} Hz, {info.channels} ch, {info.frames} frames")
            continue
        x, _ = sf.read(p, dtype="float32")        # one file at a time: low memory
        rms = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10)
        if not np.isfinite(x).all():
            problems.append(f"{p}: contains NaN or infinite values")
        elif rms < cfg["silence_db"]:
            problems.append(f"{p}: near-silent ({rms:.1f} dB)")
        elif np.abs(x).max() >= 1.0:
            problems.append(f"{p}: clipped")
    return problems


# ---------- main ----------

def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    m = pd.read_csv(MANIFEST)

    results = {
        "table": check_table(m, cfg),
        "dataset": check_dataset(m),
        "audio": check_audio(m, cfg),
    }

    print()
    for name, problems in results.items():
        status = "PASS" if not problems else f"FAIL ({len(problems)} problems)"
        print(f"{name:8s} checks: {status}")
        for p in problems[:5]:
            print(f"   - {p}")

    passed = not any(results.values())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "passed": passed,
        "n_windows": len(m),
        "minutes_of_audio": round(len(m) * cfg["window_s"] / 60, 1),
        "windows_per_class_and_split": pd.crosstab(m.label, m.split).to_dict(),
        "problems": {k: v[:20] for k, v in results.items()},
    }, indent=2))
    print(f"\n{'All checks passed.' if passed else 'Validation FAILED.'} Report: {REPORT}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()