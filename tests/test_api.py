"""API tests with a stand-in model, so they run in seconds without CLAP."""
import io
import json

import numpy as np
import pandas as pd
import soundfile as sf
from fastapi.testclient import TestClient

from serve import app as app_module
from serve.timeline import LABELS


class SpeechModel:
    def predict(self, x):
        p = np.zeros((len(x), len(LABELS)))
        p[:, 0] = 1.0
        return pd.DataFrame(p, columns=LABELS)


def client(tmp_path):
    app_module.REQUEST_LOG = tmp_path / "requests.jsonl"
    app_module.app.state.model = SpeechModel()
    app_module.app.state.model_info = {"name": "audio-tagger", "version": 99, "family": "test"}
    return TestClient(app_module.app)


def wav_bytes(seconds: float) -> bytes:
    t = np.arange(int(seconds * 16000)) / 16000
    buf = io.BytesIO()
    sf.write(buf, 0.1 * np.sin(2 * np.pi * 300 * t), 16000, format="WAV")
    return buf.getvalue()


def test_health(tmp_path):
    with client(tmp_path) as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["model"]["version"] == 99


def test_tag_returns_timeline_and_logs_request(tmp_path):
    with client(tmp_path) as c:
        r = c.post("/tag", files={"file": ("a.wav", wav_bytes(6.0), "audio/wav")})
    body = r.json()
    assert r.status_code == 200
    assert body["segments"] == [{"start": 0.0, "end": 6.0, "label": "speech", "confidence": 1.0}]
    log = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert log[0]["n_windows"] == 3 and log[0]["original_sample_rate"] == 16000


def test_rejects_non_audio(tmp_path):
    with client(tmp_path) as c:
        r = c.post("/tag", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert r.status_code == 400