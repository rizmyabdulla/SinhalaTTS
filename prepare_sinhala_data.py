"""
Convert the OpenSLR30 Sinhala TTS corpus into the Kaldi-style manifest
files that CosyVoice3 expects.

Input layout (extracted from si_lk.tar.gz on openslr.org/30):

    si_lk/
        wav/                        <- 16 kHz audio files
            sin_<spk>_<utt>.wav
            ...
        si_lk.lines.txt             <- utterance id + transcript

Output layout (under --des_dir):

    train/
        wav.scp        # utt_id /abs/path/to.wav
        text           # utt_id normalized_transcript
        utt2spk        # utt_id spk_id
        spk2utt        # spk_id utt1 utt2 ...
        instruct       # utt_id "<prompt>" (constant prompt used in SFT)
    dev/
        ... (same files, 10% of speakers held out by default)

The CosyVoice3 SFT recipe expects these exact files. The split is done
*per speaker* so the dev set is held-out speakers (not just held-out
utterances) — this gives a more honest WER/SIM estimate.

Why we resample upfront
-----------------------
CosyVoice3 runs at 24 kHz. The OpenSLR30 corpus is 16 kHz. We resample
once during data prep (using torchaudio's kaiser-best resampler) and
write the 24 kHz file alongside the original. This saves hours during
training (no per-epoch on-the-fly resampling for 2000+ steps) and
roughly halves disk usage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from tqdm import tqdm

# Make `sinhala_normalize` importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinhala_normalize import is_valid_transcript, normalize_sinhala  # noqa: E402


# OpenSLR30's si_lk.lines.txt uses the format:
#   "sin_<spk>_<utt> <transcript with quotes around it>"
# e.g.
#   sin_2241_0329430812 " කොස් කම කළේ ..."
# (the file uses parens to delimit; here we show the inner contents)
def parse_openslr30_tsv(path: str) -> List[Tuple[str, str, str]]:
    """Parse si_lk.lines.txt.

    Returns a list of (utt_id, spk_id, transcript_raw) tuples.
    The TSV is space-separated with the transcript wrapped in double
    quotes (or sometimes a mix of parens — we tolerate both).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            # OpenSLR30 format:
            #   ( sin_2241_0329430812 " <text> " )
            line = raw.strip()
            if line.startswith("("):
                line = line.lstrip("(").rstrip(")").strip()
            # Split on the first whitespace.
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id = parts[0].strip()
            if not utt_id.startswith("sin_"):
                continue
            text_raw = parts[1].strip().strip('"')
            id_parts = utt_id.split("_")
            spk_id = "_".join(id_parts[:2]) if len(id_parts) >= 2 else id_parts[0]
            rows.append((utt_id, spk_id, text_raw))
    return rows


def resample_to_24k(src: str, dst: str, target_sr: int = 24000) -> bool:
    """Resample `src` to 24 kHz mono and write to `dst`. Returns success."""
    try:
        wav, sr = torchaudio.load(src)
    except Exception as e:  # noqa: BLE001
        print(f"  ! failed to load {src}: {e}", file=sys.stderr)
        return False
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav, orig_freq=sr, new_freq=target_sr,
            resampling_method="kaiser_best",
        )
    wav = wav.clamp(-1.0, 1.0)
    # torchaudio.save requires a tensor of shape (channels, samples)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torchaudio.save(dst, wav, target_sr, encoding="PCM_S", bits_per_sample=16)
    return True


def wav_duration_sec(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            frames = w.getnframes()
            sr = w.getframerate()
            return frames / float(sr) if sr > 0 else 0.0
    except Exception:
        return 0.0


def write_kaldi_files(
    des_dir: str,
    rows: List[Tuple[str, str, str, str]],  # utt_id, spk_id, transcript, wav_abs
) -> Dict[str, int]:
    """Write wav.scp / text / utt2spk / spk2utt / instruct files.

    Returns a small dict of counts for logging.
    """
    os.makedirs(des_dir, exist_ok=True)
    spk2utt: Dict[str, List[str]] = defaultdict(list)
    with open(os.path.join(des_dir, "wav.scp"), "w", encoding="utf-8") as fw, \
         open(os.path.join(des_dir, "text"), "w", encoding="utf-8") as ft, \
         open(os.path.join(des_dir, "utt2spk"), "w", encoding="utf-8") as fu, \
         open(os.path.join(des_dir, "instruct"), "w", encoding="utf-8") as fi:
        for utt, spk, transcript, wav_path in rows:
            fw.write(f"{utt} {wav_path}\n")
            ft.write(f"{utt} {transcript}\n")
            fu.write(f"{utt} {spk}\n")
            # CosyVoice3 SFT uses this fixed prompt to teach the LLM the
            # "<sft> <text> <|speech|>" format. Keep it exactly as the
            # upstream recipe does.
            fi.write(
                f"{utt} You are a helpful assistant. "
                "Please read the text aloud in Sinhala.<|endofprompt|>\n"
            )
            spk2utt[spk].append(utt)

    with open(os.path.join(des_dir, "spk2utt"), "w", encoding="utf-8") as f:
        for spk, utts in sorted(spk2utt.items()):
            f.write(f"{spk} {' '.join(sorted(utts))}\n")

    return {
        "utts": len(rows),
        "spks": len(spk2utt),
        "spk2utt": sum(len(v) for v in spk2utt.values()),
    }


def split_train_dev(
    rows: List[Tuple[str, str, str, str]],
    dev_speaker_ratio: float = 0.10,
    seed: int = 1986,
) -> Tuple[List, List]:
    """Speaker-disjoint split: hold out `dev_speaker_ratio` of speakers.

    OpenSLR30 has ~12 speakers. Holding out 10% of speakers (~1–2) gives
    a clean dev set that tests zero-shot generalization.
    """
    rng = random.Random(seed)
    spk2rows: Dict[str, List] = defaultdict(list)
    for r in rows:
        spk2rows[r[1]].append(r)
    spks = sorted(spk2rows.keys())
    rng.shuffle(spks)
    n_dev = max(1, int(round(len(spks) * dev_speaker_ratio)))
    dev_spks = set(spks[:n_dev])
    train, dev = [], []
    for spk, lst in spk2rows.items():
        (dev if spk in dev_spks else train).extend(lst)
    return train, dev


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src_dir", required=True,
                   help="Root of the OpenSLR30 si_lk folder "
                        "(must contain si_lk.lines.txt and a wav/ subdir)")
    p.add_argument("--des_dir", required=True,
                   help="Output root. Will create data/train and data/dev under it.")
    p.add_argument("--min_dur", type=float, default=0.8,
                   help="Drop utterances shorter than this many seconds")
    p.add_argument("--max_dur", type=float, default=22.0,
                   help="Drop utterances longer than this many seconds")
    p.add_argument("--dev_speaker_ratio", type=float, default=0.10)
    p.add_argument("--dev_seed", type=int, default=1986)
    p.add_argument("--resample", action="store_true", default=True,
                   help="Resample audio to 24 kHz alongside originals")
    p.add_argument("--no-resample", dest="resample", action="store_false")
    p.add_argument("--max_utts", type=int, default=0,
                   help="If >0, keep only this many utterances (debug)")
    p.add_argument("--out_wav_dir", default=None,
                   help="Where to write the resampled wavs "
                        "(default: <des_dir>/wav_24k)")
    args = p.parse_args()

    src = Path(args.src_dir)
    tsv = src / "si_lk.lines.txt"
    wav_src_dir = src / "wav"
    if not tsv.exists():
        # Search under src_dir if manifest is nested (e.g. after tar extract)
        candidates = list(src.rglob("si_lk.lines.txt"))
        if not candidates:
            sys.exit(f"!! could not find si_lk.lines.txt under {src}")
        tsv = candidates[0]
        wav_src_dir = tsv.parent / "wav"
    if not wav_src_dir.exists():
        wav_src_dir = tsv.parent  # sometimes wavs sit next to the tsv

    out_wav_dir = Path(args.out_wav_dir) if args.out_wav_dir else Path(args.des_dir) / "wav_24k"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] reading manifest: {tsv}")
    raw = parse_openslr30_tsv(str(tsv))
    print(f"      raw rows: {len(raw)}")
    if args.max_utts:
        raw = raw[: args.max_utts]

    print(f"[2/4] filtering + normalizing + resampling")
    rows = []
    skipped = {"no_wav": 0, "too_short": 0, "too_long": 0, "bad_text": 0, "resample_fail": 0}
    for utt, spk, text_raw in tqdm(raw):
        # Find the wav
        wav_src = wav_src_dir / f"{utt}.wav"
        if not wav_src.exists():
            skipped["no_wav"] += 1
            continue

        # Normalize transcript
        text_norm = normalize_sinhala(text_raw)
        if not is_valid_transcript(text_norm):
            skipped["bad_text"] += 1
            continue

        # Resample to 24 kHz
        wav_dst = out_wav_dir / f"{utt}.wav"
        if args.resample and (not wav_dst.exists() or wav_dst.stat().st_size == 0):
            if not resample_to_24k(str(wav_src), str(wav_dst)):
                skipped["resample_fail"] += 1
                continue
        elif not args.resample:
            wav_dst = wav_src  # use original
        else:
            pass  # already exists

        dur = wav_duration_sec(str(wav_dst))
        if dur < args.min_dur:
            skipped["too_short"] += 1
            continue
        if dur > args.max_dur:
            skipped["too_long"] += 1
            continue

        rows.append((utt, spk, text_norm, str(wav_dst.resolve())))

    print(f"      kept: {len(rows)} | skipped: {skipped}")

    print(f"[3/4] splitting train/dev (speaker-disjoint, "
          f"dev_speaker_ratio={args.dev_speaker_ratio})")
    train_rows, dev_rows = split_train_dev(
        rows, dev_speaker_ratio=args.dev_speaker_ratio, seed=args.dev_seed,
    )
    print(f"      train: {len(train_rows)} utts | dev: {len(dev_rows)} utts")

    print(f"[4/4] writing Kaldi-style files")
    train_dir = Path(args.des_dir) / "train"
    dev_dir = Path(args.des_dir) / "dev"
    train_counts = write_kaldi_files(str(train_dir), train_rows)
    dev_counts = write_kaldi_files(str(dev_dir), dev_rows)
    print(f"      train: {train_counts}")
    print(f"      dev:   {dev_counts}")

    # Also drop a small JSON summary for later inspection
    summary = {
        "src_tsv": str(tsv),
        "src_wav_dir": str(wav_src_dir),
        "out_wav_dir": str(out_wav_dir),
        "min_dur": args.min_dur,
        "max_dur": args.max_dur,
        "dev_speaker_ratio": args.dev_speaker_ratio,
        "skipped": skipped,
        "train": train_counts,
        "dev": dev_counts,
    }
    with open(Path(args.des_dir) / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"      summary -> {Path(args.des_dir) / 'prep_summary.json'}")


if __name__ == "__main__":
    main()