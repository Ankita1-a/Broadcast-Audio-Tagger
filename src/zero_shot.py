"""Step 7b: Model B, CLAP zero-shot. Classify clips from text descriptions alone.

For each class we embed its description(s) with CLAP's text side. Each clip is
labelled with the class whose description is closest to the clip's embedding.
No training examples are used at all.

Logs one MLflow run per prompt set ("single" and "ensemble").

Run from the project root (after extract_clap.py):
    python src/zero_shot.py
"""
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from clap_utils import embed_text, load_clap
from common import (LABELS, MLFLOW_URI, compute_metrics, data_version, git_commit,
                    load_manifest, plot_confusion, text_report)

CONFIG = Path("configs/clap.yaml")
EMB_DIR = Path("data/processed/clap")


def class_text_embeddings(processor, model, prompts: dict) -> np.ndarray:
    """One vector per class: the average of its prompt embeddings."""
    vecs = []
    for label in LABELS:
        e = embed_text(processor, model, prompts[label], "cpu").mean(axis=0)
        vecs.append(e / np.linalg.norm(e))
    return np.stack(vecs)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    info = json.loads((EMB_DIR / "info.json").read_text())
    saved = np.load(EMB_DIR / "embeddings.npz", allow_pickle=True)

    m = load_manifest()
    emb = pd.DataFrame(saved["emb"], index=saved["window_id"]).loc[m.window_id].to_numpy()
    processor, model = load_clap(cfg["model_name"], "cpu")       # only the text side is used here

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["experiment"])
    summary = []
    for name, prompts in cfg["prompt_sets"].items():
        text = class_text_embeddings(processor, model, prompts)
        scores = emb @ text.T                                    # cosine similarity per class
        pred = scores.argmax(axis=1)

        results = {}
        for split in ["val", "test"]:
            mask = (m.split == split).to_numpy()
            results.update(compute_metrics(m.y[mask], pred[mask], split))

        with mlflow.start_run(run_name=f"clap-zero-shot-{name}"):
            mlflow.set_tags({"model_family": "B_clap_zero_shot", "git_commit": git_commit()})
            mlflow.log_params({"model_name": cfg["model_name"], "prompt_set": name,
                               "prompts_per_class": len(prompts[LABELS[0]]),
                               "data_version": data_version(), "training_examples": 0})
            mlflow.log_metrics({**results, "cpu_ms_per_window": info["cpu_ms_per_window"],
                                "model_size_mb": info["audio_size_mb"]})
            mask = (m.split == "test").to_numpy()
            yte, pte = m.y[mask].to_numpy(), pred[mask]
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                plot_confusion(yte, pte, tmp / "confusion_test.png", f"Test: model B (CLAP zero-shot, {name})")
                (tmp / "report_test.txt").write_text(text_report(yte, pte))
                (tmp / "prompts.yaml").write_text(yaml.safe_dump(prompts, sort_keys=False))
                mlflow.log_artifacts(str(tmp), "evaluation")

        summary.append({"prompt_set": name, "val_macro_f1": results["val_macro_f1"],
                        "test_macro_f1": results["test_macro_f1"],
                        "test_f1_applause": results["test_f1_applause"]})
        print(f"\n===== Prompt set: {name} =====\n{text_report(yte, pte)}")

    print("Summary:")
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print(f"\nCLAP speed on CPU: {info['cpu_ms_per_window']} ms per window, "
          f"audio encoder {info['audio_size_mb']} MB")


if __name__ == "__main__":
    main()