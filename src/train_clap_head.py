"""Step 8: Model C, a small classifier trained on frozen CLAP embeddings, plus a learning curve.

Part 1: train on all training clips, pick the regularisation strength on validation,
        evaluate once on test, log the model to MLflow.
Part 2: learning curve. Train with only 1, 2, 5, ... clips per class (5 random picks
        each) to find how much labelled data beats zero-shot.

Run from the project root (after extract_clap.py and zero_shot.py):
    python src/train_clap_head.py
"""
import json
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import (LABELS, MLFLOW_URI, compute_metrics, data_version, git_commit,
                    load_manifest, plot_confusion, text_report)

CONFIG = Path("configs/clap_head.yaml")
EMB_DIR = Path("data/processed/clap")


def fit(X, y, C: float, cfg: dict, seed: int) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight=cfg["class_weight"], max_iter=3000,
                              random_state=seed).fit(X, y)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=range(len(LABELS)))


def pick_k_per_class(train: pd.DataFrame, k: int, rng: np.random.Generator) -> np.ndarray:
    """k clips per class, taken from as many different recordings as possible."""
    picked = []
    for label in range(len(LABELS)):
        rows = train[train.y == label].sample(frac=1, random_state=int(rng.integers(1e9)))
        rows = rows.assign(rank=rows.groupby("recording_id").cumcount()).sort_values("rank", kind="stable")
        picked.extend(rows.index[:k])
    return np.array(picked)


def reference_scores(names: dict) -> dict:
    """Latest test/val macro-F1 of earlier runs (zero-shot, CNN), looked up in MLflow."""
    out = {}
    for key, run_name in names.items():
        runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'",
                                  order_by=["start_time DESC"], max_results=1)
        if len(runs) and "metrics.test_macro_f1" in runs:
            out[key] = {"val": float(runs["metrics.val_macro_f1"].iloc[0]),
                        "test": float(runs["metrics.test_macro_f1"].iloc[0])}
    return out


# ---------- learning-curve chart ----------

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3


def plot_learning_curve(curve: pd.DataFrame, refs: dict, path: Path) -> None:
    x = np.arange(len(curve))
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    mean, std = curve.test_mean.to_numpy(), curve.test_std.to_numpy()
    ax.fill_between(x, mean - std, mean + std, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=BLUE, linewidth=2, solid_capstyle="round",
            label="C: CLAP + classifier (mean of 5 picks, band = ±1 std)")
    ax.scatter(x, mean, s=40, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(f"{mean[-1]:.3f}", (x[-1] + 0.1, mean[-1]), textcoords="offset points",
                xytext=(6, 0), va="center", color=INK, fontsize=9)

    for key, color, name in [("zero_shot", ORANGE, "B: CLAP zero-shot (0 examples)"),
                             ("cnn", AQUA, "A: CNN from scratch (all examples)")]:
        if key in refs:
            v = refs[key]["test"]
            ax.hlines(v, -0.4, x[-1] + 0.1, color=color, linewidth=2, label=name, zorder=2)
            ax.annotate(f"{v:.3f}", (x[-1] + 0.1, v), textcoords="offset points", xytext=(6, 0),
                        va="center", color=INK_2, fontsize=9)

    ax.set_xticks(x, curve.shots_label, color=INK_2)
    ax.set_xlabel("Labelled clips per class used for training", color=INK_2)
    ax.set_ylabel("Test macro-F1", color=INK_2)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.4, len(curve) - 0.4)          # room on the right for value labels
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("How much labelled data does it take to beat zero-shot?",
                 color=INK, fontsize=11, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False,
              fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    info = json.loads((EMB_DIR / "info.json").read_text())
    saved = np.load(EMB_DIR / "embeddings.npz", allow_pickle=True)
    m = load_manifest()
    X = pd.DataFrame(saved["emb"], index=saved["window_id"]).loc[m.window_id].to_numpy()

    tr, va, te = [(m.split == s).to_numpy() for s in ["train", "val", "test"]]
    ytr, yva, yte = [m.y[s].to_numpy() for s in (tr, va, te)]
    seed = cfg["seed"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["experiment"])
    refs = reference_scores(cfg["reference_runs"])

    # ----- Part 1: all training data, choose C on validation -----
    grid = {C: macro_f1(yva, fit(X[tr], ytr, C, cfg, seed).predict(X[va])) for C in cfg["c_grid"]}
    best_C = max(grid, key=grid.get)
    head = fit(X[tr], ytr, best_C, cfg, seed)
    val_pred, test_pred = head.predict(X[va]), head.predict(X[te])
    test_conf = head.predict_proba(X[te]).max(axis=1)

    start = time.perf_counter()
    for _ in range(200):
        head.predict_proba(X[te][:1])
    head_ms = (time.perf_counter() - start) / 200 * 1000

    with mlflow.start_run(run_name="clap-head"):
        mlflow.set_tags({"model_family": "C_clap_frozen_head", "git_commit": git_commit()})
        mlflow.log_params({"clap_model": info["model_name"], "head": "logistic_regression",
                           "best_C": best_C, "class_weight": cfg["class_weight"],
                           "data_version": data_version(), "n_train": int(tr.sum())})
        mlflow.log_metrics({f"val_macro_f1_C_{C}": f for C, f in grid.items()})
        mlflow.log_metrics({**compute_metrics(yva, val_pred, "val"),
                            **compute_metrics(yte, test_pred, "test"),
                            "head_ms_per_window": head_ms,
                            "cpu_ms_per_window": info["cpu_ms_per_window"] + head_ms,
                            "model_size_mb": info["audio_size_mb"] + head.coef_.nbytes / 1e6})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_confusion(yte, test_pred, tmp / "confusion_test.png", "Test: model C (CLAP + classifier)")
            plot_confusion(yva, val_pred, tmp / "confusion_val.png", "Validation: model C (CLAP + classifier)")
            (tmp / "report_test.txt").write_text(text_report(yte, test_pred))
            rows = m[te].reset_index(drop=True)
            wrong = test_pred != yte
            pd.DataFrame({"window_id": rows.window_id[wrong], "path": rows.path[wrong],
                          "true": [LABELS[i] for i in yte[wrong]],
                          "predicted": [LABELS[i] for i in test_pred[wrong]],
                          "confidence": test_conf[wrong].round(3)}) \
                .sort_values("confidence", ascending=False).to_csv(tmp / "test_mistakes.csv", index=False)
            mlflow.log_artifacts(str(tmp), "evaluation")
        try:
            mlflow.sklearn.log_model(head, name="head", input_example=X[te][:1])
        except TypeError:
            mlflow.sklearn.log_model(head, artifact_path="head", input_example=X[te][:1])

    print(f"Validation macro-F1 per C: { {C: round(f, 3) for C, f in grid.items()} } -> best C = {best_C}")
    print(f"\n===== Model C, all training data =====\n{text_report(yte, test_pred)}")

    # ----- Part 2: learning curve -----
    train_rows = m[tr]
    rng = np.random.default_rng(seed)
    rows = []
    for k in cfg["learning_curve"]["shots"]:
        scores = []
        for _ in range(cfg["learning_curve"]["repeats"]):
            idx = pick_k_per_class(train_rows, k, rng)
            clf = fit(X[idx], m.y[idx].to_numpy(), best_C, cfg, seed)
            scores.append((macro_f1(yva, clf.predict(X[va])), macro_f1(yte, clf.predict(X[te]))))
        s = np.array(scores)
        rows.append({"shots": k, "shots_label": str(k), "val_mean": s[:, 0].mean(),
                     "val_std": s[:, 0].std(), "test_mean": s[:, 1].mean(), "test_std": s[:, 1].std()})
    rows.append({"shots": int(np.bincount(ytr).min()), "shots_label": "all",
                 "val_mean": macro_f1(yva, val_pred), "val_std": 0.0,
                 "test_mean": macro_f1(yte, test_pred), "test_std": 0.0})
    curve = pd.DataFrame(rows)

    with mlflow.start_run(run_name="clap-head-learning-curve"):
        mlflow.set_tags({"model_family": "C_clap_frozen_head", "git_commit": git_commit()})
        mlflow.log_params({"best_C": best_C, "repeats": cfg["learning_curve"]["repeats"],
                           "data_version": data_version()})
        for r in curve.itertuples():
            mlflow.log_metrics({f"test_macro_f1_{r.shots_label}_shots": r.test_mean,
                                f"val_macro_f1_{r.shots_label}_shots": r.val_mean})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_learning_curve(curve, refs, tmp / "learning_curve.png")
            curve.round(4).to_csv(tmp / "learning_curve.csv", index=False)
            mlflow.log_artifacts(str(tmp), "learning_curve")

    print("===== Learning curve (macro-F1, mean ± std over 5 random picks) =====")
    table = pd.DataFrame({
        "clips per class": curve.shots_label,
        "validation": [f"{a:.3f} ± {b:.3f}" for a, b in zip(curve.val_mean, curve.val_std)],
        "test": [f"{a:.3f} ± {b:.3f}" for a, b in zip(curve.test_mean, curve.test_std)]})
    print(table.to_string(index=False))
    for key, name in [("zero_shot", "B zero-shot"), ("cnn", "A CNN (all data)")]:
        if key in refs:
            print(f"Reference {name:18s} validation {refs[key]['val']:.3f} | test {refs[key]['test']:.3f}")
    if "zero_shot" in refs:
        beats = curve[curve.val_mean > refs["zero_shot"]["val"]]
        first = beats.shots_label.iloc[0] if len(beats) else "never"
        print(f"\nFewest clips per class that beat zero-shot on validation: {first}")


if __name__ == "__main__":
    main()
