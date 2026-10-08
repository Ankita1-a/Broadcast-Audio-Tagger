"""Step 12b: Freeze the current request log as the "normal" reference.

Run this right after sending normal traffic. Every later drift report compares
new requests against this reference.

    python src/monitor/make_reference.py
"""
import shutil
from pathlib import Path

import pandas as pd
import yaml

REQUEST_LOG = Path("logs/requests.jsonl")
REFERENCE = Path("monitoring/reference.jsonl")


def main() -> None:
    cfg = yaml.safe_load(Path("configs/monitor.yaml").read_text())
    if not REQUEST_LOG.exists():
        raise SystemExit(f"No {REQUEST_LOG}. Send normal traffic first.")
    log = pd.read_json(REQUEST_LOG, lines=True)
    if len(log) < cfg["min_reference"]:
        raise SystemExit(f"Only {len(log)} requests; need at least {cfg['min_reference']} for a reference.")

    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REQUEST_LOG, REFERENCE)
    print(f"Saved {len(log)} requests as the reference: {REFERENCE}")
    print(f"Model version(s): {sorted(log.model_version.unique().tolist())}")
    print(f"Mean confidence {log.mean_confidence.mean():.3f} | "
          f"mean loudness {log.mean_rms_db.mean():.1f} dB | "
          f"sample rates {sorted(log.original_sample_rate.unique().tolist())}")


if __name__ == "__main__":
    main()