"""Step 10a: Package models A, B and C in one standard format and register them.

For each candidate training run:
  1. Gather what the model needs at prediction time (weights, CLAP audio encoder,
     prompts or classifier) into one MLflow model with the same interface.
  2. Register it as a new version of "audio-tagger" in the MLflow Model Registry.
  3. Load the REGISTERED version back and score it on validation and test audio.
     This "packaging check" proves the packaged model behaves like the trained one.
  4. Store the scores, speed, size and lineage as tags on the version.

Already-registered runs are skipped, so it is safe to run again.

Run from the project root:
    python src/register_models.py
"""
import json
import logging
import shutil
import tempfile
import time
import warnings
from importlib.metadata import version as pkg_version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import soundfile as sf
import torch
import yaml
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema, TensorSpec
from sklearn.metrics import f1_score

from clap_utils import embed_text, load_clap
from common import LABELS, MLFLOW_URI, load_manifest
from tagger_pyfunc import AudioTagger

# Expected, harmless messages: we give an explicit signature instead of an input example,
# and the wrapper is our own code (so CloudPickle is fine).
warnings.filterwarnings("ignore", message=".*input example was not provided.*")
logging.getLogger("mlflow.pyfunc").setLevel(logging.ERROR)

CONFIG = Path("configs/registry.yaml")
CLAP_CONFIG = Path("configs/clap.yaml")
WINDOW = 32000                     # 2 s at 16 kHz

SIGNATURE = ModelSignature(
    inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, WINDOW))]),
    outputs=Schema([ColSpec("double", label) for label in LABELS]))

REQUIREMENTS = [f"{p}=={pkg_version(p)}" for p in
                ["mlflow", "torch", "transformers", "librosa", "scikit-learn",
                 "soundfile", "numpy", "pandas", "joblib"]]


# ---------- helpers ----------

def latest_run(run_name: str):
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["start_time DESC"], max_results=1)
    if not len(runs):
        raise SystemExit(f"No MLflow run named '{run_name}'.")
    return runs.iloc[0]


def already_registered(client, name: str, run_id: str):
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.tags.get("source_run_id") == run_id:
            return mv
    return None


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def save_clap_audio_encoder(model_name: str, out: Path) -> None:
    """Only the audio half of CLAP (~113 MB instead of ~600 MB) is needed at prediction time."""
    from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor
    ClapAudioModelWithProjection.from_pretrained(model_name).save_pretrained(out)
    ClapFeatureExtractor.from_pretrained(model_name).save_pretrained(out)


# ---------- packaging per model type ----------

def package(kind: str, run, work: Path, clap_dir: Path, clap_name: str) -> dict:
    """Write everything the model needs into `work`; return the artifact map."""
    run_id = run.run_id
    meta = {"kind": kind, "labels": LABELS, "sample_rate": 16000, "window_s": 2.0}
    artifacts = {}

    if kind == "cnn":
        cfg_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config",
                                                           dst_path=str(work)))
        cfg = next(c for c in (yaml.safe_load(p.read_text()) for p in cfg_dir.glob("*.yaml"))
                   if "features" in c and "model" in c)
        net = mlflow.pytorch.load_model(f"runs:/{run_id}/model", map_location="cpu")
        torch.save(net.state_dict(), work / "weights.pt")
        meta.update({"features": cfg["features"], "model": cfg["model"]})
        artifacts["weights"] = str(work / "weights.pt")

    elif kind == "clap_zero_shot":
        prompts_file = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="evaluation/prompts.yaml", dst_path=str(work))
        prompts = yaml.safe_load(Path(prompts_file).read_text())
        processor, full = load_clap(clap_name, "cpu")
        vecs = []
        for label in LABELS:
            e = embed_text(processor, full, prompts[label], "cpu").mean(axis=0)
            vecs.append(e / np.linalg.norm(e))
        np.save(work / "text_embeddings.npy", np.stack(vecs).astype(np.float32))
        meta["logit_scale"] = float(full.logit_scale_a.exp().item())
        meta["prompts"] = prompts
        del full
        artifacts.update({"clap": str(clap_dir), "text_embeddings": str(work / "text_embeddings.npy")})

    elif kind == "clap_head":
        head = mlflow.sklearn.load_model(f"runs:/{run_id}/head")
        joblib.dump(head, work / "head.joblib")
        artifacts.update({"clap": str(clap_dir), "head": str(work / "head.joblib")})

    (work / "meta.json").write_text(json.dumps(meta, indent=2))
    artifacts["meta"] = str(work / "meta.json")
    return artifacts


# ---------- evaluation of the packaged model ----------

def evaluate_packaged(model, m) -> dict:
    out = {}
    for split in ["val", "test"]:
        rows = m[m.split == split]
        clips = np.stack([sf.read(p, dtype="float32")[0] for p in rows.path])
        pred = model.predict(clips).to_numpy().argmax(axis=1)
        per_class = f1_score(rows.y, pred, average=None, labels=range(len(LABELS)), zero_division=0)
        out[f"{split}_macro_f1"] = float(per_class.mean())
        out[f"{split}_min_class_f1"] = float(per_class.min())
        out[f"{split}_weakest_class"] = LABELS[int(per_class.argmin())]

    one = clips[:1]
    for _ in range(2):
        model.predict(one)
    start = time.perf_counter()
    for _ in range(10):
        model.predict(one)
    out["cpu_ms_per_window"] = (time.perf_counter() - start) / 10 * 1000   # end to end, incl. features
    return out


# ---------- main ----------

def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    clap_name = yaml.safe_load(CLAP_CONFIG.read_text())["model_name"]
    name = cfg["registered_model"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_registry_uri(MLFLOW_URI)
    mlflow.set_experiment(yaml.safe_load(CLAP_CONFIG.read_text())["experiment"])
    client = mlflow.MlflowClient()
    m = load_manifest()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clap_dir = tmp / "clap_audio_encoder"
        summary = []

        for cand in cfg["candidates"]:
            run = latest_run(cand["run_name"])
            existing = already_registered(client, name, run.run_id)
            if existing:
                print(f"{cand['family']}: already registered as version {existing.version}, skipping.")
                continue

            print(f"\n=== {cand['family']} (run '{cand['run_name']}') ===")
            if cand["kind"] != "cnn" and not clap_dir.exists():
                print("Saving the CLAP audio encoder (one time)...")
                save_clap_audio_encoder(clap_name, clap_dir)

            work = tmp / cand["family"]
            work.mkdir()
            artifacts = package(cand["kind"], run, work, clap_dir, clap_name)
            size_mb = sum(folder_mb(Path(p)) if Path(p).is_dir() else Path(p).stat().st_size / 1e6
                          for p in artifacts.values())

            with mlflow.start_run(run_id=run.run_id):             # attach to the training run
                info = mlflow.pyfunc.log_model(
                    name="tagger", python_model=AudioTagger(), artifacts=artifacts,
                    code_paths=["src/tagger_pyfunc.py", "src/models"],
                    signature=SIGNATURE, pip_requirements=REQUIREMENTS,
                    registered_model_name=name)
            mv = info.registered_model_version
            print(f"Registered as {name} version {mv}. Checking the packaged model...")

            packaged = mlflow.pyfunc.load_model(f"models:/{name}/{mv}")
            scores = evaluate_packaged(packaged, m)
            logged_test = float(run["metrics.test_macro_f1"])
            check_ok = abs(scores["test_macro_f1"] - logged_test) <= cfg["packaging_tolerance"]

            tags = {"family": cand["family"], "kind": cand["kind"], "source_run_id": run.run_id,
                    "source_run_name": cand["run_name"],
                    "data_version": run.get("params.data_version", "unknown"),
                    "git_commit": run.get("tags.git_commit", "unknown"),
                    "size_mb": round(size_mb, 1),
                    "packaging_check": "passed" if check_ok else "failed",
                    "logged_test_macro_f1": round(logged_test, 4)}
            tags.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()})
            for k, v in tags.items():
                client.set_model_version_tag(name, mv, k, str(v))
            client.update_model_version(name, mv, description=(
                f"{cand['family']} packaged from run '{cand['run_name']}' ({run.run_id}). "
                f"Input: (n, 32000) float32 = 2 s clips at 16 kHz. Output: probabilities for {LABELS}."))

            summary.append({"version": mv, **tags})
            print(f"Packaging check {'PASSED' if check_ok else 'FAILED'}: packaged test macro-F1 "
                  f"{scores['test_macro_f1']:.3f} vs logged {logged_test:.3f}")
            del packaged
            shutil.rmtree(work, ignore_errors=True)

    if summary:
        print("\nNewly registered versions:")
        cols = ["version", "family", "val_macro_f1", "val_min_class_f1", "val_weakest_class",
                "test_macro_f1", "cpu_ms_per_window", "size_mb", "packaging_check"]
        import pandas as pd
        print(pd.DataFrame(summary)[cols].to_string(index=False))
    print("\nNext: python src/promote.py")


if __name__ == "__main__":
    main()