"""Step 11: The tagging API.

Endpoints:
  GET  /health  -> is the service up, and which model version is it serving?
  POST /tag     -> upload an audio file, get back a timeline of labelled segments
  GET  /docs    -> interactive page to try the API in a browser (made by FastAPI)

Every /tag request appends one line to logs/requests.jsonl (sizes, loudness,
label mix, confidence, speed). Step 12 uses that log to watch for drift.

Run locally from the project root (after src/export_champion.py):
    uvicorn serve.app:app --app-dir src --port 8000
"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow.pyfunc
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile

from serve.timeline import load_audio, tag

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "serving_model"))
REQUEST_LOG = Path(os.environ.get("REQUEST_LOG", "logs/requests.jsonl"))
CFG = yaml.safe_load(Path("configs/serve.yaml").read_text())
DATA_CFG = yaml.safe_load(Path("configs/data.yaml").read_text())
_log_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup (tests may put a stand-in model on app.state first).
    if getattr(app.state, "model", None) is None:
        app.state.model = mlflow.pyfunc.load_model(str(MODEL_DIR))
        app.state.model_info = json.loads((MODEL_DIR / "model_info.json").read_text())
    yield


app = FastAPI(title="Broadcast Audio Tagger",
              description="Labels speech, music, speech over music, applause, laughter, noise and silence.",
              lifespan=lifespan)


def write_log(entry: dict) -> None:
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, REQUEST_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": app.state.model_info}


@app.post("/tag")
def tag_audio(file: UploadFile = File(...)) -> dict:
    # A plain (non-async) function: FastAPI runs it in a worker thread, so a long
    # file doesn't block /health while it is being processed.
    data = file.file.read()
    if len(data) > CFG["max_upload_mb"] * 1e6:
        raise HTTPException(413, f"File larger than {CFG['max_upload_mb']} MB.")
    try:
        audio, original = load_audio(data)
    except Exception:
        raise HTTPException(400, "Could not read this file as audio (use wav, flac, ogg or mp3).")
    duration = len(audio) / 16000
    if duration < 0.5:
        raise HTTPException(400, "Audio is shorter than 0.5 seconds.")
    if duration > CFG["max_duration_s"]:
        raise HTTPException(413, f"Audio longer than {CFG['max_duration_s']} seconds.")

    start = time.perf_counter()
    model = app.state.model
    result = tag(audio, lambda x: model.predict(x).to_numpy(), CFG,
                 silence_db=DATA_CFG["silence_db"], target_db=DATA_CFG["target_rms_db"])
    latency = time.perf_counter() - start
    stats = result.pop("_stats")

    info = app.state.model_info
    write_log({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "model_version": info.get("version"), "duration_s": result["duration_s"],
               "n_windows": result["n_windows"], "latency_s": round(latency, 3),
               "ms_per_window": round(latency / result["n_windows"] * 1000, 1),
               **original, **stats})

    return {"model": {"name": info.get("name"), "version": info.get("version"),
                      "family": info.get("family")},
            "processing_s": round(latency, 2), **result}