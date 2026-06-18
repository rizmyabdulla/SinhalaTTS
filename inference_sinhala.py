"""
Sinhala inference for a CosyVoice3 SFT checkpoint.

This is the script you call after training to generate natural-sounding
Sinhala speech. It supports three modes:

1. SFT (use one of the speakers seen during training, e.g. the loudest one)
2. Zero-shot (provide a 10-second reference audio + its transcript to
   clone a new voice)
3. Cross-lingual (provide a reference audio only, let the model speak
   Sinhala in that voice — useful for testing prosody transfer)

The output is a 24 kHz mono WAV file. The model uses Repetition-Aware
Sampling (RAS) by default to avoid the most common failure mode of LLM-
based TTS (repeating the same syllable forever).

SFT prompt format (training distribution)
-----------------------------------------
During SFT, each utterance uses a fixed instruct prompt written to
`train/instruct`:

    You are a helpful assistant. Please read the text aloud in Sinhala.<|endofprompt|>

SFT inference conditions on `spk_id`; CosyVoice loads the speaker metadata
from the exported model. We pass `text_frontend=False` because input text
is pre-normalized with sinhala_normalize (NFC, digits, punctuation) to
match the training transcripts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

# Add the CosyVoice repo to the import path
sys.path.insert(0, os.environ.get("COSYVOICE_REPO", "/kaggle/working/CosyVoice"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinhala_normalize import normalize_sinhala  # noqa: E402


def _ensure_cosyvoice_loaded():
    """Import CosyVoice lazily so the script can be called from anywhere."""
    try:
        from cosyvoice.cli.cosyvoice import AutoModel  # noqa: F401
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"!! Could not import CosyVoice. Set COSYVOICE_REPO env var.\n   {e}"
        )


def normalize_inference_text(text: str) -> str:
    """Normalize Sinhala text; preserve '|' pause markers between segments."""
    if "|" in text:
        return "|".join(normalize_sinhala(seg) for seg in text.split("|"))
    return normalize_sinhala(text)


def pick_best_spk(spk2utt: dict, utt2dur: Optional[dict] = None) -> str:
    """Pick the speaker with the most utterances (or most total audio)."""
    if utt2dur:
        spk2dur = {}
        for spk, utts in spk2utt.items():
            spk2dur[spk] = sum(utt2dur.get(u, 0.0) for u in utts)
        return max(spk2dur, key=spk2dur.get)
    return max(spk2utt, key=lambda s: len(spk2utt[s]))


def run_sft(
    model_dir: str,
    text: str,
    spk_id: str,
    out_path: str,
    speed: float = 1.0,
):
    _ensure_cosyvoice_loaded()
    from cosyvoice.cli.cosyvoice import AutoModel

    print(f"  loading CosyVoice3 SFT model from {model_dir}")
    model = AutoModel(model_dir=model_dir, load_trt=False, load_vllm=False, fp16=True)

    print(f"  synthesizing text ({len(text)} chars) with spk={spk_id}")
    t0 = time.time()
    chunks: List[torch.Tensor] = []
    for out in model.inference_sft(
        tts_text=text, spk_id=spk_id, stream=False, speed=speed, text_frontend=False,
    ):
        chunks.append(out["tts_speech"])
    speech = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 24000)
    torchaudio.save(out_path, speech, 24000)
    print(f"  wrote {out_path}  shape={tuple(speech.shape)}  rt={time.time()-t0:.1f}s")


def run_zero_shot(
    model_dir: str,
    text: str,
    prompt_text: str,
    prompt_wav: str,
    out_path: str,
    speed: float = 1.0,
):
    _ensure_cosyvoice_loaded()
    from cosyvoice.cli.cosyvoice import AutoModel

    print(f"  loading CosyVoice3 SFT model from {model_dir}")
    model = AutoModel(model_dir=model_dir, load_trt=False, load_vllm=False, fp16=True)

    print(f"  zero-shot synthesize ({len(text)} chars), prompt_wav={prompt_wav}")
    t0 = time.time()
    chunks: List[torch.Tensor] = []
    for out in model.inference_zero_shot(
        tts_text=text,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
        stream=False,
        speed=speed,
        text_frontend=False,
    ):
        chunks.append(out["tts_speech"])
    speech = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 24000)
    torchaudio.save(out_path, speech, 24000)
    print(f"  wrote {out_path}  shape={tuple(speech.shape)}  rt={time.time()-t0:.1f}s")


def run_cross_lingual(
    model_dir: str,
    text: str,
    prompt_wav: str,
    out_path: str,
    speed: float = 1.0,
):
    _ensure_cosyvoice_loaded()
    from cosyvoice.cli.cosyvoice import AutoModel

    print(f"  loading CosyVoice3 SFT model from {model_dir}")
    model = AutoModel(model_dir=model_dir, load_trt=False, load_vllm=False, fp16=True)

    print(f"  cross-lingual synthesize ({len(text)} chars), prompt_wav={prompt_wav}")
    t0 = time.time()
    chunks: List[torch.Tensor] = []
    for out in model.inference_cross_lingual(
        tts_text=text,
        prompt_wav=prompt_wav,
        stream=False,
        speed=speed,
        text_frontend=False,
    ):
        chunks.append(out["tts_speech"])
    speech = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 24000)
    torchaudio.save(out_path, speech, 24000)
    print(f"  wrote {out_path}  shape={tuple(speech.shape)}  rt={time.time()-t0:.1f}s")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", required=True,
                    help="Folder with cosyvoice3.yaml + llm.pt/flow.pt/hift.pt "
                         "(use export_sft_model.py output after training)")
    ap.add_argument("--mode", choices=["sft", "zero_shot", "cross_lingual"], default="sft")
    ap.add_argument("--text", required=True,
                    help="Sinhala text to synthesize. For long text, split on '|' "
                         "to insert pauses between sentences.")
    ap.add_argument("--spk_id", default=None,
                    help="For SFT mode: the speaker id seen in training. "
                         "If omitted, the speaker with the most utts is used.")
    ap.add_argument("--prompt_text", default="",
                    help="For zero_shot: the transcript of the reference wav")
    ap.add_argument("--prompt_wav", default=None,
                    help="For zero_shot/cross_lingual: 24 kHz reference wav (>= 3 s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--data_dir", default=None,
                    help="If given, use this dir to pick the most-common spk_id")
    args = ap.parse_args()

    text = normalize_inference_text(args.text)
    prompt_text = normalize_inference_text(args.prompt_text) if args.prompt_text else ""

    spk_id = args.spk_id
    if args.mode == "sft" and spk_id is None:
        if not args.data_dir:
            sys.exit("!! --spk_id is required for SFT mode (or pass --data_dir)")
        spk2utt_path = Path(args.data_dir) / "train/spk2utt"
        spk2utt = {}
        for line in spk2utt_path.read_text().splitlines():
            k, *vs = line.split()
            spk2utt[k] = vs
        spk_id = pick_best_spk(spk2utt)
        print(f"  auto-picked spk_id={spk_id}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.mode == "sft":
        run_sft(args.model_dir, text, spk_id, args.out, args.speed)
    elif args.mode == "zero_shot":
        if not args.prompt_wav or not args.prompt_text:
            sys.exit("!! zero_shot requires --prompt_wav and --prompt_text")
        run_zero_shot(args.model_dir, text, prompt_text,
                      args.prompt_wav, args.out, args.speed)
    elif args.mode == "cross_lingual":
        if not args.prompt_wav:
            sys.exit("!! cross_lingual requires --prompt_wav")
        run_cross_lingual(args.model_dir, text, args.prompt_wav, args.out, args.speed)


if __name__ == "__main__":
    main()
