"""Shared helpers used by every model (A, B, C, D), so all are judged the same way."""
import random
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ["speech", "music", "speech_over_music", "applause", "laughter", "noise"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
MANIFEST = Path("data/processed/windows.csv")
MLFLOW_URI = "sqlite:///mlflow.db"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_manifest() -> pd.DataFrame:
    m = pd.read_csv(MANIFEST)
    m["y"] = m.label.map(LABEL_TO_ID)
    return m


# ---------- lineage: which data and code produced a model ----------

def data_version() -> str:
    """Fingerprint of the dataset from dvc.lock, so every run records its exact data."""
    lock = yaml.safe_load(Path("dvc.lock").read_text())
    for out in lock["stages"]["build"]["outs"]:
        if out["path"] == "data/processed/windows":
            return out["md5"]
    return "unknown"


def git_commit() -> str:
    try:
        run = dict(text=True, stderr=subprocess.DEVNULL)
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], **run).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], **run).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---------- metrics ----------

def compute_metrics(y_true, y_pred, prefix: str) -> dict:
    """Macro-F1 is the main score: every class counts equally, even rare ones."""
    out = {
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(len(LABELS))),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(LABELS)), zero_division=0)
    out.update({f"{prefix}_f1_{label}": float(f) for label, f in zip(LABELS, per_class)})
    return out


def text_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=range(len(LABELS)),
                                 target_names=LABELS, digits=3, zero_division=0)


# ---------- confusion matrix ----------

# One-hue sequential ramp (light -> dark blue): darker cell = larger share.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def plot_confusion(y_true, y_pred, path: Path, title: str) -> None:
    """Each row = true class, normalised to 100%, so rare classes are as visible as common ones."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "#ffffff" if share[i, j] > 0.55 else "#0b0b0b"
            ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=8, color=color)
    names = [label.replace("_", " ") for label in LABELS]
    ax.set_xticks(range(len(LABELS)), names, rotation=35, ha="right", color="#52514e", fontsize=9)
    ax.set_yticks(range(len(LABELS)), names, color="#52514e", fontsize=9)
    ax.set_xlabel("Predicted", color="#52514e")
    ax.set_ylabel("True", color="#52514e")
    ax.set_title(title, color="#0b0b0b", fontsize=11, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)