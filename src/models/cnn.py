"""Model A: a small CNN that reads log-mel spectrograms.

A spectrogram is an image of sound: time left to right, pitch bottom to top,
brightness = loudness. The CNN learns patterns in that image (e.g. the regular
harmonics of music, or the broadband bursts of applause).
"""
import librosa
import numpy as np
import torch
from torch import nn


def log_mel(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """2 s of audio -> (n_mels, frames) log-mel spectrogram in dB."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"])
    return librosa.power_to_db(mel, ref=1.0, amin=1e-10).astype(np.float32)


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, n_mels: int, channels: list[int], dropout: float):
        super().__init__()
        # Per-frequency mean/std from the training set. Stored inside the model,
        # so the served model normalises inputs exactly as in training.
        self.register_buffer("mean", torch.zeros(1, n_mels, 1))
        self.register_buffer("std", torch.ones(1, n_mels, 1))
        blocks, c_in = [], 1
        for c in channels:
            blocks.append(conv_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def classify(self, x_norm: torch.Tensor) -> torch.Tensor:
        h = self.features(x_norm.unsqueeze(1))      # (B, C, F', T')
        h = h.mean(dim=(2, 3))                      # average over frequency and time
        return self.head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_mels, frames) log-mel. Returns class scores (logits)."""
        return self.classify(self.normalise(x))