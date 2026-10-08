"""Small helpers for loading CLAP and getting embeddings (used by models B and C)."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # run any op MPS lacks on CPU

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor


def load_clap(name: str, device):
    processor = ClapProcessor.from_pretrained(name)
    model = ClapModel.from_pretrained(name).to(device).eval()
    return processor, model


def _as_tensor(out) -> torch.Tensor:
    """Older transformers return a tensor; newer ones return an object with pooler_output."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def audio_inputs(processor, clips_16k: list[np.ndarray], sr: int, device) -> dict:
    """16 kHz clips -> CLAP's input features (48 kHz log-mel, 2 s clip repeated to fill 10 s)."""
    audio = [librosa.resample(x, orig_sr=16000, target_sr=sr) for x in clips_16k]
    inputs = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt",
                                         truncation="rand_trunc", padding="repeatpad")
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def embed_audio(model, inputs: dict) -> np.ndarray:
    emb = _as_tensor(model.get_audio_features(**inputs))
    return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()


@torch.no_grad()
def embed_text(processor, model, texts: list[str], device) -> np.ndarray:
    tok = processor.tokenizer(texts, padding=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    emb = _as_tensor(model.get_text_features(**tok))
    return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()