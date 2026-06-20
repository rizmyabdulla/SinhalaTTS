"""
Extract per-utterance speaker embeddings (campplus) and discrete speech
tokens (v3) for the Sinhala SFT data.

This runs ONCE before training. The CosyVoice3 training loop then reads
these from .pt files inside the parquet shards instead of recomputing
them every epoch, which would otherwise dominate training time on T4.

Outputs (written to <data_dir>/):

    <data_dir>/utt2embedding.pt   # dict {utt_id: Tensor[192]}
    <data_dir>/spk2embedding.pt   # dict {spk_id: Tensor[192]}  (utt-mean)
    <data_dir>/utt2speech_token.pt # dict {utt_id: Tensor[T]}  (int32)

Why pre-extraction (not online) for SFT on Kaggle
------------------------------------------------
The upstream CosyVoice3 recipe supports an `online_feature` mode where
the ONNX encoder runs inside the training loop. That mode is great on
A100s but on a T4 the ONNX+CUDA contention slows training by ~30%.
For SFT (where data is static), pre-extraction is strictly better.

campplus.onnx runs on CPU (it's small, ~30 MB, and CPU is fine for
batch=1 inference). speech_tokenizer_v3.onnx runs on GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Loaders (mirroring cosyvoice/utils/onnx.py but explicit so the script
# runs without the CosyVoice source tree on the import path)
# -----------------------------------------------------------------------------

def make_session(path: str, providers: List[str]) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


# -----------------------------------------------------------------------------
# Speaker embedding (campplus)
# -----------------------------------------------------------------------------

class CampPlusEmbedder:
    """Wrap campplus.onnx for 16 kHz mel -> 192-d embedding.

    The ONNX model expects (B, T, 80) log-mel features computed with the
    kaldi.fbank recipe (htk_feat, no cmvn). The Sinhala wav is 24 kHz
    (we resampled it during data prep) but campplus wants 16 kHz, so we
    resample on the fly using a single torchaudio resampler.
    """
    def __init__(self, onnx_path: str, target_sr: int = 16000):
        self.session = make_session(onnx_path, providers=["CPUExecutionProvider"])
        self.target_sr = target_sr
        self._resamplers: Dict[int, torch.nn.Module] = {}

    def _resample(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return wav
        key = sr
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr,
                resampling_method="sinc_interp_hann",
            )
        return self._resamplers[key](wav)

    @torch.inference_mode()
    def embed(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = self._resample(wav, sr)
        # Center crop to <= 10 s for reproducible speaker embeddings across runs.
        max_len = 10 * self.target_sr
        if wav.shape[1] > max_len:
            start = (wav.shape[1] - max_len) // 2
            wav = wav[:, start: start + max_len]
        feat = kaldi.fbank(
            wav,
            num_mel_bins=80,
            dither=0,
            sample_frequency=self.target_sr,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        out = self.session.run(
            None,
            {self.session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()},
        )[0]
        return torch.tensor(out).flatten()


# -----------------------------------------------------------------------------
# Discrete speech tokens (v3)
# -----------------------------------------------------------------------------

class SpeechTokenizerV3:
    """Wrap speech_tokenizer_v3.onnx for waveform -> discrete token ids.

    Matches CosyVoice3 upstream (cosyvoice/cli/frontend.py):
    16 kHz audio -> whisper.log_mel_spectrogram(n_mels=128) -> ONNX input
    shape (B, 128, T). Token vocabulary is 6561 (81^2, 2-D RVQ).
    """
    def __init__(self, onnx_path: str, device: str = "cuda"):
        providers = (["CUDAExecutionProvider"]
                     if device == "cuda" and torch.cuda.is_available()
                     else ["CPUExecutionProvider"])
        self.session = make_session(onnx_path, providers=providers)
        self.device = device
        self.target_sr = 16000

    @torch.inference_mode()
    def encode(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=sr,
                new_freq=self.target_sr,
                resampling_method="sinc_interp_hann",
            )
        # whisper expects (1, samples); output is (1, 128, T) or (128, T)
        speech = wav if wav.dim() == 2 else wav.unsqueeze(0)
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        feat_len = np.array([feat.shape[2]], dtype=np.int32)
        out, out_lens = self.session.run(
            None,
            {
                self.session.get_inputs()[0].name: feat.cpu().numpy(),
                self.session.get_inputs()[1].name: feat_len,
            },
        )
        return torch.tensor(out[0], dtype=torch.int32), int(out_lens[0][0])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True,
                    help="Folder containing wav.scp (e.g. data/train)")
    ap.add_argument("--campplus_onnx", required=True)
    ap.add_argument("--speech_tokenizer_onnx", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process this many utterances (debug)")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no-skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--save_every", type=int, default=200,
                    help="Persist .pt files every N utterances (crash safety)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    wav_scp = data_dir / "wav.scp"
    utt2spk_path = data_dir / "utt2spk"
    if not wav_scp.exists():
        sys.exit(f"!! {wav_scp} not found. Run prepare_sinhala_data.py first.")
    if not utt2spk_path.exists():
        sys.exit(f"!! {utt2spk_path} not found. Run prepare_sinhala_data.py first.")

    # Load manifest
    wav_lines = wav_scp.read_text(encoding="utf-8").strip().splitlines()
    spk_lines = utt2spk_path.read_text(encoding="utf-8").strip().splitlines()
    utt2wav = dict(line.split(maxsplit=1) for line in wav_lines)
    utt2spk = dict(line.split(maxsplit=1) for line in spk_lines)
    if args.limit:
        keys = list(utt2wav.keys())[: args.limit]
        utt2wav = {k: utt2wav[k] for k in keys}
        utt2spk = {k: utt2spk[k] for k in keys}

    # Load existing partial state for resume
    emb_path = data_dir / "utt2embedding.pt"
    spk_emb_path = data_dir / "spk2embedding.pt"
    tok_path = data_dir / "utt2speech_token.pt"
    if args.skip_existing:
        utt2emb = torch.load(emb_path, weights_only=True) if emb_path.exists() else {}
        spk2emb = torch.load(spk_emb_path, weights_only=True) if spk_emb_path.exists() else {}
        utt2tok = torch.load(tok_path, weights_only=True) if tok_path.exists() else {}
    else:
        utt2emb, spk2emb, utt2tok = {}, {}, {}

    # Init extractors
    print(f"  loading campplus ONNX on CPU")
    embedder = CampPlusEmbedder(args.campplus_onnx)
    print(f"  loading speech_tokenizer ONNX on {args.device}")
    tokenizer = SpeechTokenizerV3(args.speech_tokenizer_onnx, device=args.device)

    pending: List[str] = []
    spk2utt_acc: Dict[str, List[str]] = {}
    for utt in utt2wav.keys():
        if utt in utt2tok and utt in utt2emb:
            spk2utt_acc.setdefault(utt2spk[utt], []).append(utt)
            continue
        pending.append(utt)

    print(f"  to process: {len(pending)} | already done: {len(utt2wav) - len(pending)}")
    if not pending:
        print("  nothing to do!")
        return

    t0 = time.time()
    for i, utt in enumerate(tqdm(pending, desc="extract")):
        wav_path = utt2wav[utt]
        spk = utt2spk[utt]
        try:
            emb = embedder.embed(wav_path)
            tok, tok_len = tokenizer.encode(wav_path)
        except Exception as e:  # noqa: BLE001
            print(f"\n  ! {utt}: {e}", file=sys.stderr)
            continue
        utt2emb[utt] = emb
        utt2tok[utt] = tok[:tok_len] if tok.dim() == 1 else tok[0, :tok_len]
        spk2utt_acc.setdefault(spk, []).append(utt)

        # Periodically flush to disk
        if (i + 1) % args.save_every == 0:
            torch.save(utt2emb, emb_path)
            torch.save(utt2tok, tok_path)
            # Re-compute spk2emb as mean of utterance embeddings
            spk2emb = _aggregate_spk_emb(utt2emb, utt2spk)
            torch.save(spk2emb, spk_emb_path)
            elapsed = time.time() - t0
            print(f"  flushed @ {i+1} ({elapsed:.1f}s, {len(pending)/(i+1)*elapsed - elapsed:.0f}s left)")

    # Final flush
    torch.save(utt2emb, emb_path)
    torch.save(utt2tok, tok_path)
    spk2emb = _aggregate_spk_emb(utt2emb, utt2spk)
    torch.save(spk2emb, spk_emb_path)
    print(f"  done. utt2emb={len(utt2emb)} spk2emb={len(spk2emb)} utt2tok={len(utt2tok)}")


def _aggregate_spk_emb(utt2emb: Dict[str, torch.Tensor],
                        utt2spk: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Mean-pool utterance embeddings to speaker embeddings."""
    spk2acc: Dict[str, List[torch.Tensor]] = {}
    for utt, emb in utt2emb.items():
        spk = utt2spk.get(utt)
        if spk is None:
            continue
        spk2acc.setdefault(spk, []).append(emb)
    return {spk: torch.stack(v).mean(dim=0) for spk, v in spk2acc.items()}


if __name__ == "__main__":
    main()