"""Step 9: Model D, knowledge distillation. Train the small CNN to imitate model C.

Teacher = model C (CLAP + classifier): accurate but ~35x slower and ~95x bigger.
Student = the same small CNN as model A: fast and tiny.

Model A learned from hard labels only ("this clip is applause").
Model D also learns from the teacher's soft predictions ("70% applause, 25% noise,
5% laughter"), which carry extra information about how classes relate.

Run from the project root:
    python src/train_distilled.py
"""
import argparse
import copy
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from torch import nn

from common import (LABELS, MLFLOW_URI, compute_metrics, data_version, get_device, git_commit,
                    load_manifest, plot_confusion, set_seed, text_report)
from models.cnn import SmallCNN
from train_cnn import (augment, cpu_ms_per_window, flatten, load_features, log_pytorch_model,
                       predict_logits)

EMB_DIR = Path("data/processed/clap")


# ---------- teacher ----------

def load_teacher(run_name: str):
    """Model C's classifier and settings, looked up in MLflow by run name."""
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["start_time DESC"], max_results=1)
    if not len(runs):
        raise SystemExit(f"No MLflow run named '{run_name}'. Run src/train_clap_head.py first.")
    run = runs.iloc[0]
    head = mlflow.sklearn.load_model(f"runs:/{run.run_id}/head")
    return run.run_id, head, float(run["params.best_C"]), run["params.class_weight"]


def teacher_logits(m: pd.DataFrame, emb: np.ndarray, head, best_C: float, class_weight: str,
                   folds: int, seed: int) -> np.ndarray:
    """Teacher scores for every clip.
    Train clips: out-of-fold (each scored by a classifier trained on the other folds,
    grouped by recording so no recording is on both sides).
    Val/test clips: the real model C classifier (it never trained on them)."""
    logits = head.decision_function(emb).astype(np.float32)
    tr = np.where(m.split == "train")[0]
    for fit_idx, score_idx in GroupKFold(n_splits=folds).split(tr, groups=m.recording_id.iloc[tr]):
        clf = LogisticRegression(C=best_C, class_weight=class_weight, max_iter=3000,
                                 random_state=seed).fit(emb[tr[fit_idx]], m.y.iloc[tr[fit_idx]])
        logits[tr[score_idx]] = clf.decision_function(emb[tr[score_idx]])
    return logits


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distill.yaml")
    args = parser.parse_args()
    dcfg = yaml.safe_load(Path(args.config).read_text())
    cfg = yaml.safe_load(Path(dcfg["base_config"]).read_text())
    T, alpha = dcfg["temperature"], dcfg["alpha"]

    set_seed(cfg["seed"])
    device = get_device()
    m = load_manifest()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["experiment"])

    # ----- teacher soft labels -----
    teacher_run_id, head, best_C, class_weight = load_teacher(dcfg["teacher_run"])
    saved = np.load(EMB_DIR / "embeddings.npz", allow_pickle=True)
    emb = pd.DataFrame(saved["emb"], index=saved["window_id"]).loc[m.window_id].to_numpy()
    t_logits = teacher_logits(m, emb, head, best_C, class_weight,
                              dcfg["teacher_folds"], cfg["seed"])

    split = {s: (m.split == s).to_numpy() for s in ["train", "val", "test"]}
    y = m.y.to_numpy(copy=True)
    teacher_train_acc = float((t_logits[split["train"]].argmax(1) == y[split["train"]]).mean())
    print(f"Teacher (out-of-fold) accuracy on training clips: {teacher_train_acc:.3f}")

    # ----- student data (same spectrograms as model A) -----
    X = load_features(m.path, cfg["features"])
    Xtr, ytr, ttr = X[split["train"]], y[split["train"]], t_logits[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    test_rows = m[split["test"]].reset_index(drop=True)

    model = SmallCNN(len(LABELS), cfg["features"]["n_mels"], **cfg["model"])
    model.mean.copy_(torch.from_numpy(Xtr.mean(axis=(0, 2))).view(1, -1, 1))
    model.std.copy_(torch.from_numpy(Xtr.std(axis=(0, 2)) + 1e-6).view(1, -1, 1))
    model.to(device)

    tc = cfg["train"]
    counts = np.bincount(ytr, minlength=len(LABELS))
    weights = counts.sum() / (len(LABELS) * counts) if tc["class_weights"] else np.ones(len(LABELS))
    hard_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimiser = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=tc["max_epochs"])

    with mlflow.start_run(run_name=dcfg["run_name"]) as run:
        mlflow.set_tags({"model_family": "D_cnn_distilled", "git_commit": git_commit(),
                         "teacher_run_id": teacher_run_id})
        mlflow.log_params({**flatten(cfg), "temperature": T, "alpha": alpha,
                           "teacher_run": dcfg["teacher_run"], "teacher_folds": dcfg["teacher_folds"],
                           "data_version": data_version(), "device": str(device)})
        mlflow.log_metric("teacher_train_accuracy_oof", teacher_train_acc)
        mlflow.log_artifact(args.config, "config")
        mlflow.log_artifact(dcfg["base_config"], "config")

        best_f1, best_epoch, best_state, waited = -1.0, 0, None, 0
        g = torch.Generator().manual_seed(cfg["seed"])
        for epoch in range(1, tc["max_epochs"] + 1):
            model.train()
            order = torch.randperm(len(ytr), generator=g).numpy()
            totals = np.zeros(3)
            for i in range(0, len(order), tc["batch_size"]):
                idx = order[i:i + tc["batch_size"]]
                xb = torch.from_numpy(Xtr[idx]).to(device)
                yb = torch.from_numpy(ytr[idx]).to(device)
                tb = torch.from_numpy(ttr[idx]).to(device)
                logits = model.classify(augment(model.normalise(xb), cfg["augment"]))

                # Match the teacher's softened predictions (KL divergence, scaled by T^2)...
                soft = F.kl_div(F.log_softmax(logits / T, dim=1), F.softmax(tb / T, dim=1),
                                reduction="batchmean") * T * T
                # ...and the true labels.
                hard = hard_loss.to(device)(logits, yb)
                loss = alpha * soft + (1 - alpha) * hard

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                totals += np.array([loss.item(), soft.item(), hard.item()]) * len(idx)
            scheduler.step()

            val_pred = predict_logits(model, Xva, device).argmax(1).numpy()
            val = compute_metrics(yva, val_pred, "val")
            mlflow.log_metrics({"train_loss": totals[0] / len(ytr), "train_soft_loss": totals[1] / len(ytr),
                                "train_hard_loss": totals[2] / len(ytr),
                                "val_macro_f1": val["val_macro_f1"], "val_accuracy": val["val_accuracy"],
                                "lr": scheduler.get_last_lr()[0]}, step=epoch)
            print(f"epoch {epoch:3d} | loss {totals[0] / len(ytr):.3f} "
                  f"(teacher {totals[1] / len(ytr):.3f}, labels {totals[2] / len(ytr):.3f}) | "
                  f"val macro-F1 {val['val_macro_f1']:.3f}")

            if val["val_macro_f1"] > best_f1:
                best_f1, best_epoch, waited = val["val_macro_f1"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= tc["patience"]:
                    print(f"Early stop: no improvement for {tc['patience']} epochs.")
                    break

        # ----- final evaluation -----
        model.load_state_dict(best_state)
        val_pred = predict_logits(model, Xva, device).argmax(1).numpy()
        test_probs = predict_logits(model, Xte, device).softmax(1).numpy()
        test_pred = test_probs.argmax(1)
        teacher_test_pred = t_logits[split["test"]].argmax(1)

        final = {**compute_metrics(yva, val_pred, "val"), **compute_metrics(yte, test_pred, "test")}
        final.update({"best_epoch": best_epoch,
                      "test_agreement_with_teacher": float((test_pred == teacher_test_pred).mean()),
                      "cpu_ms_per_window": cpu_ms_per_window(model, Xte[:1]),
                      "model_size_mb": sum(t.numel() * t.element_size()
                                           for t in model.state_dict().values()) / 1e6})
        mlflow.log_metrics(final, step=epoch + 1)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_confusion(yva, val_pred, tmp / "confusion_val.png", "Validation: model D (distilled CNN)")
            plot_confusion(yte, test_pred, tmp / "confusion_test.png", "Test: model D (distilled CNN)")
            (tmp / "report_test.txt").write_text(text_report(yte, test_pred))
            wrong = test_pred != yte
            pd.DataFrame({"window_id": test_rows.window_id[wrong], "path": test_rows.path[wrong],
                          "true": [LABELS[i] for i in yte[wrong]],
                          "predicted": [LABELS[i] for i in test_pred[wrong]],
                          "teacher_predicted": [LABELS[i] for i in teacher_test_pred[wrong]],
                          "confidence": test_probs.max(1)[wrong].round(3)}) \
                .sort_values("confidence", ascending=False).to_csv(tmp / "test_mistakes.csv", index=False)
            mlflow.log_artifacts(str(tmp), "evaluation")

        log_pytorch_model(model.cpu().eval(), Xte[:1])

    print(f"\nBest epoch: {best_epoch}")
    print(f"Validation macro-F1:          {final['val_macro_f1']:.3f}")
    print(f"Test macro-F1:                {final['test_macro_f1']:.3f}")
    print(f"Agrees with teacher on test:  {final['test_agreement_with_teacher']:.1%}")
    print(f"CPU speed:                    {final['cpu_ms_per_window']:.2f} ms per 2 s window")
    print(f"Model size:                   {final['model_size_mb']:.2f} MB")
    print(f"\nTest report:\n{text_report(yte, test_pred)}")
    print(f"MLflow run id: {run.info.run_id}")


if __name__ == "__main__":
    main()