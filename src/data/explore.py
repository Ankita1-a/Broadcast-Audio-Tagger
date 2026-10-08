"""Step 3: Explore the raw datasets.

Reads only file headers (not the audio itself) for the inventory, so it uses
very little memory. Nothing in data/raw is changed.

Run from the project root:
    python src/data/explore.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

RAW = Path("data/raw")
ESC = RAW / "ESC-50-master"
MUSAN = RAW / "musan"
OUT = Path("data/processed/inventory.csv")


def musan_inventory() -> pd.DataFrame:
    """One row per MUSAN file. Folder names give the label and source."""
    rows = []
    for path in sorted(MUSAN.rglob("*.wav")):
        info = sf.info(path)
        rel = path.relative_to(MUSAN)
        rows.append({
            "dataset": "musan",
            "path": str(path),
            "label": rel.parts[0],      # speech / music / noise
            "subset": rel.parts[1],     # e.g. librivox, fma, free-sound
            "recording_id": f"musan_{path.stem}",
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_s": info.duration,
        })
    return pd.DataFrame(rows)


def esc50_inventory() -> pd.DataFrame:
    """One row per ESC-50 clip, using the official metadata table."""
    meta = pd.read_csv(ESC / "meta" / "esc50.csv")
    rows = []
    for r in meta.itertuples():
        path = ESC / "audio" / r.filename
        info = sf.info(path)
        rows.append({
            "dataset": "esc50",
            "path": str(path),
            "label": r.category,        # e.g. clapping, laughing, rain
            "subset": f"fold{r.fold}",
            # Clips cut from the same original Freesound recording share src_file.
            # We will keep them in the same split later to avoid leakage.
            "recording_id": f"esc50_{r.src_file}",
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_s": info.duration,
        })
    return pd.DataFrame(rows)


def silent_window_share(path: str, win_s: float = 2.0, threshold_db: float = -60.0) -> float:
    """Share of 2-second windows in a clip that are almost silent."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    win = int(win_s * sr)
    n = len(audio) // win
    if n == 0:
        return float("nan")
    frames = audio[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-10
    return float((20 * np.log10(rms) < threshold_db).mean())


def summarise(df: pd.DataFrame, name: str) -> None:
    print(f"\n===== {name} =====")
    print(f"Files: {len(df):,}   Total hours: {df.duration_s.sum() / 3600:.1f}")
    print("Sample rates:", df.sample_rate.value_counts().to_dict())
    print("Channels:    ", df.channels.value_counts().to_dict())
    d = df.duration_s
    print(f"Duration (s): min {d.min():.1f} | median {d.median():.1f} | max {d.max():.1f}")


def main() -> None:
    musan = musan_inventory()
    esc = esc50_inventory()

    summarise(musan, "MUSAN")
    by_source = (
        musan.groupby(["label", "subset"])
        .agg(files=("path", "count"), hours=("duration_s", lambda s: round(s.sum() / 3600, 1)))
    )
    print("\nMUSAN by label and source:\n", by_source.to_string())

    summarise(esc, "ESC-50")
    print("\nESC-50 categories (40 clips each):")
    print(", ".join(sorted(esc.label.unique())))

    print("\nSilence check (share of 2-second windows that are near-silent):")
    for cat in ["clapping", "laughing"]:
        paths = esc.loc[esc.label == cat, "path"]
        share = np.nanmean([silent_window_share(p) for p in paths])
        print(f"  {cat:10s} {share:.0%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([musan, esc], ignore_index=True).to_csv(OUT, index=False)
    print(f"\nSaved inventory of all files to {OUT}")


if __name__ == "__main__":
    main()