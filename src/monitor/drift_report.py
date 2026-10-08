"""Step 12c: Compare recent requests with the reference and report drift.

No true labels are needed. Two kinds of signals are tested:
  - input drift: does the incoming audio look different? (sample rate, loudness, silence)
  - model drift: does the model behave differently? (confidence, mix of predicted labels)
Each feature gets a statistical test from Evidently (Kolmogorov-Smirnov for numbers,
chi-square / Z-test for categories). An alert is raised if many features drift or if
mean confidence drops.

Outputs:
  reports/drift/drift_<time>.html   interactive Evidently report (open in a browser)
  reports/drift/latest.json         summary used by scripts and CI
  MLflow experiment "audio-tagger-monitoring": one run per report

    python src/monitor/drift_report.py
    python src/monitor/drift_report.py --fail-on-alert    # exit code 1 on alert (for automation)
"""
import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings("ignore", category=RuntimeWarning)   # constant columns (e.g. channels)

CONFIG = Path("configs/monitor.yaml")
REFERENCE = Path("monitoring/reference.jsonl")
CURRENT = Path("logs/requests.jsonl")
OUT_DIR = Path("reports/drift")
MLFLOW_URI = "sqlite:///mlflow.db"


def load_log(path: Path) -> pd.DataFrame:
    """Read a request log and turn the label_share dict into one column per label."""
    df = pd.read_json(path, lines=True)
    shares = pd.json_normalize(df.pop("label_share")).add_prefix("label_share_")
    return pd.concat([df.reset_index(drop=True), shares], axis=1)


def run_drift(ref: pd.DataFrame, cur: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per feature: test used, score, threshold, drifted?"""
    groups = cfg["features"]
    numerical = groups["input"]["numerical"] + groups["model"]["numerical"]
    categorical = groups["input"]["categorical"]
    cols = numerical + categorical
    ref, cur = ref[cols].copy(), cur[cols].copy()
    for c in categorical:
        ref[c], cur[c] = ref[c].astype(str), cur[c].astype(str)

    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    snapshot = Report([DataDriftPreset()]).run(Dataset.from_pandas(cur, data_definition=definition),
                                              Dataset.from_pandas(ref, data_definition=definition))
    rows = []
    for metric in snapshot.dict()["metrics"]:
        conf = metric["config"]
        if not conf.get("type", "").endswith("ValueDrift"):
            continue
        method, score, threshold = conf["method"], float(metric["value"]), float(conf["threshold"])
        drifted = score < threshold if "p_value" in method else score >= threshold
        group = "input" if conf["column"] in groups["input"]["numerical"] + categorical else "model"
        rows.append({"feature": conf["column"], "group": group, "test": method,
                     "score": score, "threshold": threshold, "drifted": drifted,
                     "reference_mean": _mean(ref[conf["column"]]),
                     "current_mean": _mean(cur[conf["column"]])})
    return pd.DataFrame(rows), snapshot


def _mean(s: pd.Series):
    try:
        return round(float(pd.to_numeric(s).mean()), 3)
    except Exception:
        return s.value_counts(normalize=True).round(2).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())

    if not REFERENCE.exists():
        raise SystemExit("No reference yet. Send normal traffic, then run make_reference.py.")
    if not CURRENT.exists():
        raise SystemExit(f"No {CURRENT}. Send some traffic first.")
    ref, cur = load_log(REFERENCE), load_log(CURRENT)
    if len(cur) < cfg["min_current"]:
        raise SystemExit(f"Only {len(cur)} new requests; need at least {cfg['min_current']}.")

    table, snapshot = run_drift(ref, cur, cfg)
    drift_share = float(table.drifted.mean())
    conf_drop = float(ref.mean_confidence.mean() - cur.mean_confidence.mean())

    reasons = []
    if drift_share >= cfg["alerts"]["drift_share"]:
        reasons.append(f"{table.drifted.sum()} of {len(table)} features drifted "
                       f"({drift_share:.0%} >= {cfg['alerts']['drift_share']:.0%})")
    if conf_drop > cfg["alerts"]["max_confidence_drop"]:
        reasons.append(f"mean confidence fell by {conf_drop:.3f} "
                       f"(> {cfg['alerts']['max_confidence_drop']})")
    status = "ALERT" if reasons else "OK"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = OUT_DIR / f"drift_{stamp}.html"
    snapshot.save_html(str(html))
    summary = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": status,
               "reasons": reasons, "n_reference": len(ref), "n_current": len(cur),
               "model_versions": sorted(int(v) for v in cur.model_version.unique()),
               "drift_share": round(drift_share, 3), "confidence_drop": round(conf_drop, 3),
               "drifted_features": table[table.drifted].feature.tolist(),
               "features": table.round(4).to_dict(orient="records"), "html_report": str(html)}
    (OUT_DIR / "latest.json").write_text(json.dumps(summary, indent=2, default=str))

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("audio-tagger-monitoring")
    with mlflow.start_run(run_name=f"drift-{stamp}"):
        mlflow.set_tags({"status": status, "reasons": "; ".join(reasons) or "none",
                         "model_versions": str(summary["model_versions"])})
        mlflow.log_metrics({"drift_share": drift_share, "n_drifted": int(table.drifted.sum()),
                            "confidence_drop": conf_drop, "n_current": len(cur),
                            "mean_confidence_current": float(cur.mean_confidence.mean()),
                            "mean_confidence_reference": float(ref.mean_confidence.mean())})
        mlflow.log_metrics({f"score_{r.feature}": r.score for r in table.itertuples()})
        mlflow.log_artifact(str(html))
        mlflow.log_artifact(str(OUT_DIR / "latest.json"))

    show = table.copy()
    show["score"] = show.score.map(lambda v: f"{v:.3g}")
    show["drifted"] = np.where(show.drifted, "YES", "-")
    pd.set_option("display.width", 160)
    print(f"Reference: {len(ref)} requests | Current: {len(cur)} requests | "
          f"model version(s) {summary['model_versions']}\n")
    print(show[["group", "feature", "test", "score", "drifted", "reference_mean", "current_mean"]]
          .sort_values(["group", "drifted"], ascending=[True, False]).to_string(index=False))
    print(f"\nStatus: {status}")
    for r in reasons:
        print(f"  - {r}")
    print(f"\nReport: {html}  (open it in a browser)")

    if args.fail_on_alert and status == "ALERT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()