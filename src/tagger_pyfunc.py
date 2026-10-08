"""One standard wrapper for every registered model.

Whatever is inside (CNN, CLAP zero-shot, CLAP + classifier), the packaged model
has the same interface:
    input : numpy array (n_clips, 32000) = 2 s clips of 16 kHz mono audio
    output: table (n_clips, 6) of class probabilities

So the serving app never needs to know which model is the champion.
This file and src/models/ are shipped inside each MLflow model (code_paths).
"""
import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import mlflow.pyfunc
import numpy as np
import pandas as pd

LABELS = ["speech", "music", "speech_over_music", "applause", "laughter", "noise"]
SAMPLE_RATE = 16000
CLAP_SAMPLE_RATE = 48000
CLAP_BATCH = 16


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class AudioTagger(mlflow.pyfunc.PythonModel):
    # We describe inputs with an explicit MLflow signature instead of Python type hints;
    # this tells MLflow not to look for type hints (and not to warn about them).
    _skip_type_hint_validation = True

    def load_context(self, context):
        import torch
        self.meta = json.loads(open(context.artifacts["meta"]).read())
        self.kind = self.meta["kind"]

        if self.kind == "cnn":
            from models.cnn import SmallCNN, log_mel
            self.log_mel = log_mel
            self.net = SmallCNN(len(LABELS), self.meta["features"]["n_mels"], **self.meta["model"])
            self.net.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
            self.net.eval()
        else:
            from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor
            self.fe = ClapFeatureExtractor.from_pretrained(context.artifacts["clap"])
            self.clap = ClapAudioModelWithProjection.from_pretrained(context.artifacts["clap"]).eval()
            if self.kind == "clap_zero_shot":
                self.text = np.load(context.artifacts["text_embeddings"])
                self.scale = float(self.meta["logit_scale"])
            else:
                import joblib
                self.head = joblib.load(context.artifacts["head"])

    def _clap_embed(self, clips: np.ndarray) -> np.ndarray:
        import librosa
        import torch
        out = []
        for i in range(0, len(clips), CLAP_BATCH):
            audio = [librosa.resample(c, orig_sr=SAMPLE_RATE, target_sr=CLAP_SAMPLE_RATE)
                     for c in clips[i:i + CLAP_BATCH]]
            inputs = self.fe(audio, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt",
                             truncation="rand_trunc", padding="repeatpad")
            with torch.no_grad():
                emb = self.clap(**inputs).audio_embeds
            out.append(torch.nn.functional.normalize(emb, dim=-1).numpy())
        return np.concatenate(out)

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        import torch
        clips = np.asarray(model_input, dtype=np.float32)
        if clips.ndim == 1:
            clips = clips[None]

        if self.kind == "cnn":
            specs = np.stack([self.log_mel(c, SAMPLE_RATE, self.meta["features"]) for c in clips])
            with torch.no_grad():
                probs = softmax(self.net(torch.from_numpy(specs)).numpy())
        elif self.kind == "clap_zero_shot":
            probs = softmax(self.scale * self._clap_embed(clips) @ self.text.T)
        else:
            probs = self.head.predict_proba(self._clap_embed(clips))
        return pd.DataFrame(probs, columns=LABELS)