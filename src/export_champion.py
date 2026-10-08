"""Step 11a: Copy the current champion out of the registry into ./serving_model.

The Docker image is built from this folder, so each image contains exactly one,
known model version (recorded in serving_model/model_info.json).

Run from the project root:
    python src/export_champion.py
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml

from common import MLFLOW_URI

OUT = Path("serving_model")


def main() -> None:
    cfg = yaml.safe_load(Path("configs/registry.yaml").read_text())
    name, alias = cfg["registered_model"], cfg["promotion"]["alias"]
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_registry_uri(MLFLOW_URI)
    mv = mlflow.MlflowClient().get_model_version_by_alias(name, alias)

    if OUT.exists():
        shutil.rmtree(OUT)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{mv.version}", dst_path=str(OUT))

    info = {"name": name, "alias": alias, "version": int(mv.version),
            "family": mv.tags.get("family"), "source_run_id": mv.tags.get("source_run_id"),
            "data_version": mv.tags.get("data_version"), "git_commit": mv.tags.get("git_commit"),
            "val_macro_f1": float(mv.tags["val_macro_f1"]), "test_macro_f1": float(mv.tags["test_macro_f1"]),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "model_info.json").write_text(json.dumps(info, indent=2))

    # Make every file readable by any user. Some saved files (e.g. the CLAP weights)
    # come out readable by their owner only, and the container runs as a non-root user.
    for p in OUT.rglob("*"):
        p.chmod(0o755 if p.is_dir() else 0o644)

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"Exported {name} version {mv.version} ({info['family']}) to {OUT}/ ({size:.0f} MB)")


if __name__ == "__main__":
    main()