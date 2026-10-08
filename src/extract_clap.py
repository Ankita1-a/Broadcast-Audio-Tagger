"""Step 7a: Run CLAP over every clip ONCE and save the embeddings.

An embedding is CLAP's summary of a clip: 512 numbers. Clips that sound alike get
similar numbers. Saving them means models B and C never run the heavy CLAP model
again during experiments, so each experiment takes seconds on the Mac.

Outputs:
  data/processed/clap/embeddings.npz   (window_id + 512-number embedding per clip)
  data/processed/clap/info.json        (model name, CPU speed, size)

Run from the project root (first run downloads ~600 MB):
    python src/extract_clap.py
"""
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from clap_utils import audio_inputs, embed_audio, load_clap
from common import get_device, load_manifest

CONFIG = Path("configs/clap.yaml")
OUT = Path("data/processed/clap")


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    device = get_device()
    m = load_manifest()
    processor, model = load_clap(cfg["model_name"], device)

    embs, bs = [], cfg["batch_size"]
    for i in tqdm(range(0, len(m), bs), desc=f"CLAP embeddings on {device}"):
        clips = [sf.read(p, dtype="float32")[0] for p in m.path[i:i + bs]]
        embs.append(embed_audio(model, audio_inputs(processor, clips, cfg["sample_rate"], device)))
    embs = np.concatenate(embs).astype(np.float32)

    # Speed on CPU (the serving setup): one 2 s window through the audio side of CLAP.
    model.to("cpu")
    one = audio_inputs(processor, [sf.read(m.path[0], dtype="float32")[0]], cfg["sample_rate"], "cpu")
    for _ in range(2):
        embed_audio(model, one)
    start = time.perf_counter()
    for _ in range(10):
        embed_audio(model, one)
    cpu_ms = (time.perf_counter() - start) / 10 * 1000

    audio_params = sum(p.numel() for p in model.audio_model.parameters()) + \
        sum(p.numel() for p in model.audio_projection.parameters())

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "embeddings.npz", window_id=m.window_id.to_numpy(), emb=embs)
    info = {"model_name": cfg["model_name"], "dim": int(embs.shape[1]), "n_windows": len(embs),
            "cpu_ms_per_window": round(cpu_ms, 2), "audio_params": int(audio_params),
            "audio_size_mb": round(audio_params * 4 / 1e6, 1)}
    (OUT / "info.json").write_text(json.dumps(info, indent=2))

    print(f"\nSaved {len(embs):,} embeddings of size {embs.shape[1]} to {OUT}/")
    print(f"CLAP audio encoder: {audio_params / 1e6:.1f}M parameters, "
          f"{cpu_ms:.0f} ms per 2 s window on CPU")


if __name__ == "__main__":
    main()