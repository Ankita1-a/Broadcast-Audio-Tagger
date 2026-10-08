"""Step 6: Train model A (small CNN from scratch) and log everything to MLflow.

What gets recorded for every run:
  - settings (from the config), data version (from dvc.lock), Git commit
  - training curves: loss and validation macro-F1 per epoch
  - final validation and test scores, per-class F1
  - confusion matrices, a text report, and the list of test mistakes
  - speed (ms per window on CPU), model size, and the model itself

Run from the project root:
    python src/train_cnn.py
"""
import argparse
import copy
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from torch import nn
from tqdm import tqdm

from common import (LABELS, MLFLOW_URI, compute_metrics, data_version, get_device, git_commit,
                    load_manifest, plot_confusion, set_seed, text_report)
from models.cnn import SmallCNN, log_mel


# ---------- data ----------

def load_features(paths, feat_cfg: dict) -> np.ndarray:
    """Turn every 2 s clip into a log-mel spectrogram. Small enough to keep in memory (~70 MB)."""
    specs = []
    for p in tqdm(paths, desc="Computing spectrograms"):
        audio, sr = sf.read(p, dtype="float32")
        specs.append(log_mel(audio, sr, feat_cfg))
    return np.stack(specs)


def augment(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Random changes during training only, so the model learns sounds, not exact clips."""
    x = x.clone()
    b, n_freq, n_time = x.shape
    for i in range(b):
        if cfg["time_shift"]:
            x[i] = torch.roll(x[i], int(torch.randint(0, n_time, (1,))), dims=1)
        f = int(torch.randint(0, cfg["freq_mask"] + 1, (1,)))
        f0 = int(torch.randint(0, n_freq - f + 1, (1,)))
        x[i, f0:f0 + f, :] = 0            # 0 = average value after normalisation
        t = int(torch.randint(0, cfg["time_mask"] + 1, (1,)))
        t0 = int(torch.randint(0, n_time - t + 1, (1,)))
        x[i, :, t0:t0 + t] = 0
    return x


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, device, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu()
           for i in range(0, len(X), batch_size)]
    return torch.cat(out)


# ---------- helpers ----------

def flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


def cpu_ms_per_window(model: nn.Module, x_one: np.ndarray, runs: int = 50) -> float:
    """Average time for one 2 s window on CPU (the serving setup), spectrogram excluded."""
    m = copy.deepcopy(model).cpu().eval()
    x = torch.from_numpy(x_one)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        start = time.perf_counter()
        for _ in range(runs):
            m(x)
    return (time.perf_counter() - start) / runs * 1000


def log_pytorch_model(model: nn.Module, example: np.ndarray) -> None:
    """Save the model in MLflow. "pickle" keeps it usable with any batch size;
    code_paths ships src/models with it so it can be loaded from anywhere."""
    kwargs = dict(input_example=example, code_paths=["src/models"])
    for extra in (dict(name="model", serialization_format="pickle"),   # newer MLflow 3
                  dict(name="model"),                                   # older MLflow 3
                  dict(artifact_path="model")):                         # MLflow 2
        try:
            mlflow.pytorch.log_model(model, **extra, **kwargs)
            return
        except TypeError:
            continue


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_cnn.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    set_seed(cfg["seed"])
    device = get_device()
    m = load_manifest()
    X = load_features(m.path, cfg["features"])
    split = {s: (m.split == s).to_numpy() for s in ["train", "val", "test"]}
    Xtr, ytr = X[split["train"]], m.y[split["train"]].to_numpy(copy=True)
    Xva, yva = X[split["val"]], m.y[split["val"]].to_numpy(copy=True)
    Xte, yte = X[split["test"]], m.y[split["test"]].to_numpy(copy=True)
    test_rows = m[split["test"]].reset_index(drop=True)

    model = SmallCNN(len(LABELS), cfg["features"]["n_mels"], **cfg["model"])
    model.mean.copy_(torch.from_numpy(Xtr.mean(axis=(0, 2))).view(1, -1, 1))
    model.std.copy_(torch.from_numpy(Xtr.std(axis=(0, 2)) + 1e-6).view(1, -1, 1))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    tc = cfg["train"]
    counts = np.bincount(ytr, minlength=len(LABELS))
    weights = counts.sum() / (len(LABELS) * counts) if tc["class_weights"] else np.ones(len(LABELS))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimiser = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=tc["max_epochs"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.set_tags({"model_family": "A_cnn_scratch", "git_commit": git_commit()})
        mlflow.log_params(flatten(cfg))
        mlflow.log_params({"data_version": data_version(), "device": str(device),
                           "n_params": n_params, "n_train": len(ytr), "n_val": len(yva),
                           "n_test": len(yte), "class_weights_values": np.round(weights, 2).tolist()})
        mlflow.log_artifact(args.config, "config")
        mlflow.log_artifact("configs/data.yaml", "config")

        best_f1, best_epoch, best_state, waited = -1.0, 0, None, 0
        g = torch.Generator().manual_seed(cfg["seed"])
        for epoch in range(1, tc["max_epochs"] + 1):
            model.train()
            order = torch.randperm(len(ytr), generator=g).numpy()
            total = 0.0
            for i in range(0, len(order), tc["batch_size"]):
                idx = order[i:i + tc["batch_size"]]
                xb = torch.from_numpy(Xtr[idx]).to(device)
                yb = torch.from_numpy(ytr[idx]).to(device)
                logits = model.classify(augment(model.normalise(xb), cfg["augment"]))
                loss = criterion.to(device)(logits, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += loss.item() * len(idx)
            scheduler.step()

            val_logits = predict_logits(model, Xva, device)
            val_loss = criterion.cpu()(val_logits, torch.from_numpy(yva)).item()
            val = compute_metrics(yva, val_logits.argmax(1).numpy(), "val")
            mlflow.log_metrics({"train_loss": total / len(ytr), "val_loss": val_loss,
                                "val_macro_f1": val["val_macro_f1"],
                                "val_accuracy": val["val_accuracy"],
                                "lr": scheduler.get_last_lr()[0]}, step=epoch)
            print(f"epoch {epoch:3d} | train loss {total / len(ytr):.3f} | "
                  f"val loss {val_loss:.3f} | val macro-F1 {val['val_macro_f1']:.3f}")

            if val["val_macro_f1"] > best_f1:
                best_f1, best_epoch, waited = val["val_macro_f1"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= tc["patience"]:
                    print(f"Early stop: no improvement for {tc['patience']} epochs.")
                    break

        # ----- final evaluation with the best epoch's weights -----
        model.load_state_dict(best_state)
        val_pred = predict_logits(model, Xva, device).argmax(1).numpy()
        test_probs = predict_logits(model, Xte, device).softmax(1).numpy()
        test_pred = test_probs.argmax(1)

        final = {**compute_metrics(yva, val_pred, "val"), **compute_metrics(yte, test_pred, "test")}
        final.update({"best_epoch": best_epoch,
                      "cpu_ms_per_window": cpu_ms_per_window(model, Xte[:1]),
                      "model_size_mb": sum(t.numel() * t.element_size()
                                           for t in model.state_dict().values()) / 1e6})
        # Logged one step after the last epoch, so MLflow's "latest value" is the final score
        # (not the last epoch's validation score).
        mlflow.log_metrics(final, step=epoch + 1)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_confusion(yva, val_pred, tmp / "confusion_val.png", "Validation: model A (CNN)")
            plot_confusion(yte, test_pred, tmp / "confusion_test.png", "Test: model A (CNN)")
            (tmp / "report_test.txt").write_text(text_report(yte, test_pred))
            wrong = test_pred != yte
            pd.DataFrame({
                "window_id": test_rows.window_id[wrong], "path": test_rows.path[wrong],
                "true": [LABELS[i] for i in yte[wrong]],
                "predicted": [LABELS[i] for i in test_pred[wrong]],
                "confidence": test_probs.max(1)[wrong].round(3),
            }).sort_values("confidence", ascending=False).to_csv(tmp / "test_mistakes.csv", index=False)
            mlflow.log_artifacts(str(tmp), "evaluation")

        log_pytorch_model(model.cpu().eval(), Xte[:1])

    print(f"\nBest epoch: {best_epoch}")
    print(f"Validation macro-F1: {final['val_macro_f1']:.3f}")
    print(f"Test macro-F1:       {final['test_macro_f1']:.3f}")
    print(f"CPU speed:           {final['cpu_ms_per_window']:.2f} ms per 2 s window")
    print(f"Model size:          {final['model_size_mb']:.2f} MB ({n_params:,} parameters)")
    print(f"\nTest report:\n{text_report(yte, test_pred)}")
    print(f"MLflow run id: {run.info.run_id}")


if __name__ == "__main__":
    main()