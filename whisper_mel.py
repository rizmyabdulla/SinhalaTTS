"""Whisper-compatible log-mel without openai-whisper (Python 3.12 safe).

CosyVoice3 calls ``whisper.log_mel_spectrogram`` during training
(compute_whisper_fbank). The real openai-whisper package fails to build on
Kaggle's Python 3.12 kernels, so we provide a librosa-based drop-in.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def log_mel_spectrogram(
    audio: torch.Tensor,
    n_mels: int = 80,
    padding: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Drop-in for ``whisper.log_mel_spectrogram`` (16 kHz, 128 mels for v3)."""
    import librosa

    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio, dtype=torch.float32)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))

    batched = audio.dim() == 2
    wav = audio.squeeze(0).detach().cpu().numpy() if batched else audio.detach().cpu().numpy()
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=16000,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        fmin=0,
        fmax=8000,
    )
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    out = torch.from_numpy(log_spec.astype(np.float32))
    if batched:
        out = out.unsqueeze(0)
    if device is not None:
        out = out.to(device)
    return out
