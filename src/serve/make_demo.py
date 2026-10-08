"""Build a demo recording with a known answer, to check the API end to end.

Joins test-set clips (never used in training) into one file:
  speech -> music -> applause -> speech_over_music -> silence -> laughter -> noise
and writes the true timeline next to it.

Run from the project root:
    python src/serve/make_demo.py
Outputs: demo/demo.wav and demo/expected.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PLAN = [("speech", 4), ("music", 4), ("applause", 2), ("speech_over_music", 4),
        ("silence", 2), ("laughter", 2), ("noise", 3)]          # (label, number of 2 s clips)
SR = 16000


def main() -> None:
    m = pd.read_csv("data/processed/windows.csv")
    test = m[m.split == "test"]
    pieces, expected, t = [], [], 0.0
    for label, n in PLAN:
        if label == "silence":
            clips = [np.zeros(2 * SR, dtype=np.float32)] * n
        else:
            paths = test[test.label == label].sample(n, random_state=7).path
            clips = [sf.read(p, dtype="float32")[0] for p in paths]
        pieces.extend(clips)
        expected.append({"start": t, "end": t + 2.0 * n, "label": label})
        t += 2.0 * n

    out = Path("demo")
    out.mkdir(exist_ok=True)
    sf.write(out / "demo.wav", np.concatenate(pieces), SR)
    (out / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"Wrote demo/demo.wav ({t:.0f} s). True timeline:")
    for e in expected:
        print(f"  {e['start']:5.1f} - {e['end']:5.1f} s  {e['label']}")


if __name__ == "__main__":
    main()