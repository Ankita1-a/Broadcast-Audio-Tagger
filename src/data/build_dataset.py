"""Step 4: Build the clean, windowed dataset.

Order matters:
  1. Split whole recordings into train / val / test FIRST (prevents leakage).
  2. Cut 2-second windows, drop silent ones, convert to 16 kHz mono, normalise loudness.
  3. Create speech_over_music by mixing speech + music windows from the SAME split.

Outputs:
  data/processed/windows/<split>/<label>/<window_id>.wav
  data/processed/windows.csv   (one row per window: label, split, source, recording, ...)

Run from the project root:
    python src/data/build_dataset.py
"""
from pathlib import Path
import shutil

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

CONFIG = Path("configs/data.yaml")
INVENTORY = Path("data/processed/inventory.csv")
OUT_DIR = Path("data/processed/windows")
MANIFEST = Path("data/processed/windows.csv")


# ---------- audio helpers ----------

def rms_db(x: np.ndarray) -> float:
    """Loudness of a signal in dBFS (0 = maximum, more negative = quieter)."""
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10))


def normalise(x: np.ndarray, target_db: float) -> np.ndarray:
    """Bring a window to a fixed loudness so the model can't use volume as a shortcut."""
    y = x * 10 ** ((target_db - rms_db(x)) / 20)
    peak = np.abs(y).max()
    if peak > 0.99:                       # avoid clipping
        y = y * 0.99 / peak
    return y.astype(np.float32)


def load_window(path: str, start_s: float, window_s: float, sr_out: int) -> np.ndarray:
    """Read only the needed slice of a file (memory-friendly), as mono at sr_out."""
    sr = sf.info(path).samplerate
    audio, _ = sf.read(path, start=int(round(start_s * sr)), frames=int(round(window_s * sr)),
                       dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sr_out:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_out)
    n = int(window_s * sr_out)
    return np.pad(audio[:n], (0, max(0, n - len(audio))))


# ---------- step 1: split by recording ----------

def assign_splits(inv: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    inv = inv.copy()

    # ESC-50: official folds almost always keep clips from the same source recording together.
    fold_to_split = {f: s for s, folds in cfg["splits"]["esc50_folds"].items() for f in folds}
    esc = inv.dataset == "esc50"
    folds = inv.loc[esc, "subset"].str.replace("fold", "").astype(int)
    inv.loc[esc, "split"] = folds.map(fold_to_split)

    # A few source recordings are spread over two folds. Move all their clips into
    # train, so the validation and test sets stay completely unseen.
    spread = inv[esc].groupby("recording_id").split.nunique()
    conflicts = spread[spread > 1].index
    if len(conflicts):
        inv.loc[esc & inv.recording_id.isin(conflicts), "split"] = "train"
        print(f"Moved {len(conflicts)} ESC-50 recordings that span two folds into train: "
              f"{list(conflicts)}")

    # MUSAN: shuffle whole recordings, separately per label + source,
    # so every split gets a fair mix of sources (e.g. both audiobooks and hearings).
    r = cfg["splits"]["musan"]
    for _, group in inv[~esc].groupby(["label", "subset"]):
        ids = rng.permutation(group.recording_id.unique())
        n_tr = int(round(len(ids) * r["train"]))
        n_va = int(round(len(ids) * r["val"]))
        split_of = {i: "train" for i in ids[:n_tr]}
        split_of.update({i: "val" for i in ids[n_tr:n_tr + n_va]})
        split_of.update({i: "test" for i in ids[n_tr + n_va:]})
        inv.loc[group.index, "split"] = group.recording_id.map(split_of)
    return inv


# ---------- step 2: candidate windows, then pick + save ----------

def candidate_windows(inv: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    w = cfg["window_s"]
    label_map = cfg["esc50_label_map"]
    rows = []
    for r in inv.itertuples():
        if r.dataset == "musan":
            if r.duration_s < w:                      # too short for one window
                continue
            starts = np.arange(0, r.duration_s - w + 1e-6, w)   # non-overlapping
            k = min(cfg["musan_windows_per_recording"], len(starts))
            starts = rng.choice(starts, size=k, replace=False)
            label = r.label
        else:
            if r.label not in label_map:              # ESC-50 class we don't use
                continue
            starts = cfg["esc50_window_starts"]
            label = label_map[r.label]
        for s in starts:
            rows.append({"src_path": r.path, "label": label, "split": r.split,
                         "source_dataset": r.dataset, "subset": r.subset,
                         "recording_id": r.recording_id, "start_s": float(s)})
    return pd.DataFrame(rows)


def split_quota(target, dataset: str, split: str, cfg: dict):
    """How many windows of one class/source this split should get."""
    if target == "all":
        return None
    if dataset == "musan":
        share = cfg["splits"]["musan"][split]
    else:
        share = len(cfg["splits"]["esc50_folds"][split]) / 5
    return int(round(target * share))


def select_and_save(cands: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int]:
    sr, w = cfg["sample_rate"], cfg["window_s"]
    kept, dropped = [], 0

    groups = list(cands.groupby(["label", "source_dataset", "split"]))
    for (label, dataset, split), g in tqdm(groups, desc="Cutting windows"):
        quota = split_quota(cfg["targets"][label][dataset], dataset, split, cfg)

        # Round-robin: take 1st window of every recording, then 2nd, ...
        # so windows are spread across as many recordings as possible.
        g = g.sample(frac=1, random_state=cfg["seed"])
        g = g.assign(rank=g.groupby("recording_id").cumcount()).sort_values("rank", kind="stable")

        n = 0
        for r in g.itertuples():
            if quota is not None and n >= quota:
                break
            audio = load_window(r.src_path, r.start_s, w, sr)
            level = rms_db(audio)
            if level < cfg["silence_db"]:
                dropped += 1
                continue
            window_id = f"{label}_{split}_{len(kept):05d}"
            out = OUT_DIR / split / label / f"{window_id}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, normalise(audio, cfg["target_rms_db"]), sr, subtype="PCM_16")
            kept.append({"window_id": window_id, "path": str(out), "label": label,
                         "split": split, "source_dataset": dataset, "subset": r.subset,
                         "recording_id": r.recording_id, "src_path": r.src_path,
                         "start_s": r.start_s, "rms_db": round(level, 1), "snr_db": np.nan})
            n += 1
        if quota is not None and n < quota:
            tqdm.write(f"  warning: {label}/{dataset}/{split} got {n} of {quota} windows")
    return pd.DataFrame(kept), dropped


# ---------- step 3: speech over music ----------

def make_mixes(manifest: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    sr, mix_cfg = cfg["sample_rate"], cfg["speech_over_music"]
    rows = []
    for split, share in cfg["splits"]["musan"].items():
        speech = manifest[(manifest.label == "speech") & (manifest.split == split)]
        music = manifest[(manifest.label == "music") & (manifest.split == split)]
        n_mixes = int(round(mix_cfg["total"] * share))
        seen = set()
        while len(seen) < n_mixes:
            s = speech.iloc[rng.integers(len(speech))]
            m = music.iloc[rng.integers(len(music))]
            if (s.window_id, m.window_id) in seen:
                continue
            seen.add((s.window_id, m.window_id))

            xs, _ = sf.read(s.path, dtype="float32")
            xm, _ = sf.read(m.path, dtype="float32")
            snr = float(rng.uniform(*mix_cfg["snr_db"]))
            xm = xm * 10 ** ((rms_db(xs) - rms_db(xm) - snr) / 20)   # music snr dB below speech
            mix = normalise(xs + xm, cfg["target_rms_db"])

            window_id = f"speech_over_music_{split}_{len(rows):05d}"
            out = OUT_DIR / split / "speech_over_music" / f"{window_id}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, mix, sr, subtype="PCM_16")
            rows.append({"window_id": window_id, "path": str(out), "label": "speech_over_music",
                         "split": split, "source_dataset": "musan_mix", "subset": "mix",
                         "recording_id": f"{s.recording_id}+{m.recording_id}",
                         "src_path": f"{s.path}+{m.path}", "start_s": np.nan,
                         "rms_db": round(rms_db(mix), 1), "snr_db": round(snr, 1)})
    return pd.DataFrame(rows)


# ---------- main ----------

def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    rng = np.random.default_rng(cfg["seed"])

    if OUT_DIR.exists():                  # rebuild from scratch every time
        shutil.rmtree(OUT_DIR)

    inv = pd.read_csv(INVENTORY)
    inv = assign_splits(inv, cfg, rng)

    # Safety check: no recording may appear in more than one split.
    leaks = inv.groupby("recording_id").split.nunique().gt(1)
    if leaks.any():
        raise SystemExit(f"Leakage: {leaks.sum()} recordings appear in more than one split, "
                         f"e.g. {list(leaks[leaks].index[:3])}")
    print("Leakage check passed: every recording is in exactly one split.")

    cands = candidate_windows(inv, cfg, rng)
    print(f"Candidate windows: {len(cands):,}")

    manifest, dropped = select_and_save(cands, cfg)
    print(f"Dropped as silent: {dropped}")

    print("Mixing speech_over_music ...")
    manifest = pd.concat([manifest, make_mixes(manifest, cfg, rng)], ignore_index=True)
    manifest.to_csv(MANIFEST, index=False)

    print("\nWindows per class and split:")
    print(pd.crosstab(manifest.label, manifest.split, margins=True)
          .reindex(columns=["train", "val", "test", "All"]).to_string())
    minutes = len(manifest) * cfg["window_s"] / 60
    print(f"\nTotal: {len(manifest):,} windows ({minutes:.0f} minutes of audio)")
    print(f"Saved audio to {OUT_DIR}/ and the list of windows to {MANIFEST}")


if __name__ == "__main__":
    main()