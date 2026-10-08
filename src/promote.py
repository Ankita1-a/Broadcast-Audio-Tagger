"""Step 10b: The promotion gate. Decide which registered version becomes "champion".

Rules (from configs/registry.yaml), applied to every version:
  1. Packaging check passed (the packaged model reproduces its training scores).
  2. Every class scores at least min_class_f1 on validation (no class left behind).
  3. Fast enough: CPU time per 2 s window within the latency budget.
Among eligible versions, the best validation macro-F1 wins. It replaces the current
champion only if it is better by at least min_improvement.
Test scores are shown but never used to decide.

The decision and reasons are saved to reports/promotion.json and logged to MLflow.

Run from the project root:
    python src/promote.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from common import MLFLOW_URI

CONFIG = Path("configs/registry.yaml")
REPORT = Path("reports/promotion.json")


def check(tags: dict, rules: dict) -> list[str]:
    """Return the list of rules a version fails (empty = eligible)."""
    fails = []
    if tags.get("packaging_check") != "passed":
        fails.append("packaging check failed")
    if float(tags["val_min_class_f1"]) < rules["min_class_f1"]:
        fails.append(f"{tags['val_weakest_class']} F1 {float(tags['val_min_class_f1']):.3f} "
                     f"< {rules['min_class_f1']}")
    if float(tags["cpu_ms_per_window"]) > rules["max_cpu_ms_per_window"]:
        fails.append(f"{float(tags['cpu_ms_per_window']):.0f} ms > {rules['max_cpu_ms_per_window']} ms budget")
    return fails


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    name, rules = cfg["registered_model"], cfg["promotion"]
    alias, metric = rules["alias"], rules["select_on"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_registry_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"No versions of '{name}'. Run src/register_models.py first.")

    rows = []
    for mv in versions:
        fails = check(mv.tags, rules)
        rows.append({"version": int(mv.version), "family": mv.tags.get("family"),
                     metric: float(mv.tags[metric]),
                     "val_min_class_f1": float(mv.tags["val_min_class_f1"]),
                     "test_macro_f1": float(mv.tags["test_macro_f1"]),
                     "cpu_ms": float(mv.tags["cpu_ms_per_window"]),
                     "size_mb": float(mv.tags["size_mb"]),
                     "eligible": not fails, "reasons": "; ".join(fails) or "all rules passed"})
    table = pd.DataFrame(rows).sort_values("version")

    try:
        current = client.get_model_version_by_alias(name, alias)
        current_v, current_score = int(current.version), float(current.tags[metric])
    except Exception:
        current_v, current_score = None, None

    eligible = table[table.eligible]
    if eligible.empty:
        decision, winner = "no change: no version passes the gate", current_v
    else:
        best = eligible.sort_values(metric, ascending=False).iloc[0]
        winner = int(best.version)
        if current_v is None:
            decision = f"promote v{winner}: first champion"
        elif winner == current_v:
            decision = f"keep v{current_v}: already the best eligible version"
        elif best[metric] >= current_score + rules["min_improvement"]:
            decision = (f"promote v{winner}: {metric} {best[metric]:.3f} beats champion "
                        f"v{current_v} ({current_score:.3f}) by >= {rules['min_improvement']}")
        else:
            decision = (f"keep v{current_v}: v{winner} is not better by >= {rules['min_improvement']}")
            winner = current_v

    if winner is not None and winner != current_v:
        client.set_registered_model_alias(name, alias, str(winner))

    report = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "registered_model": name, "alias": alias, "previous_champion": current_v,
              "champion": winner, "decision": decision, "rules": rules,
              "versions": table.to_dict(orient="records")}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))

    mlflow.set_experiment("audio-tagger")
    with mlflow.start_run(run_name="promotion-decision"):
        mlflow.set_tags({"decision": decision, "champion_version": str(winner)})
        mlflow.log_artifact(str(REPORT))

    pd.set_option("display.width", 160)
    print(table.round(3).to_string(index=False))
    print(f"\nDecision: {decision}")
    print(f"'{alias}' now points to: " + (f"{name} version {winner}" if winner else "nothing yet"))
    print("(Selected on validation. Test scores are shown for reporting only.)")


if __name__ == "__main__":
    main()