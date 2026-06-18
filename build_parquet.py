"""
Build the parquet shards the CosyVoice3 trainer reads.

For each partition (data/train and data/dev) we already have:
    wav.scp, text, utt2spk, spk2utt, instruct,
    utt2embedding.pt, spk2embedding.pt, utt2speech_token.pt

This script packs them into one or more .parquet files, each holding
a few hundred to a few thousand utterances. The trainer streams these
parquets via pyarrow and reads columns as needed.

Each row in the parquet looks like:
    utt           : str  (utt_id)
    audio_data    : bytes  (full WAV file bytes, includes header)
    wav           : str  (absolute path, kept for debugging)
    text          : str  (normalized Sinhala)
    spk           : str  (spk_id)
    utt_embedding : list[float]  (192-d campplus)
    spk_embedding : list[float]  (192-d mean-pooled campplus)
    speech_token  : list[int]  (T token ids, int32)
    instruct      : str  (the SFT prompt for this utt)

Why we embed audio as raw bytes (not a tensor)
----------------------------------------------
Reading the wav from disk in the DataLoader would multiply disk I/O by
batch_size * num_workers. By inlining the bytes, every worker reads
from local memory pages and the OS caches the bytes once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def build_shard(
    utts: List[str],
    utt2wav: dict,
    utt2text: dict,
    utt2spk: dict,
    utt2instruct: dict,
    utt2emb: dict,
    spk2emb: dict,
    utt2tok: dict,
) -> pd.DataFrame:
    df = pd.DataFrame()
    df["utt"] = utts
    df["audio_data"] = [_read_bytes(utt2wav[u]) for u in utts]
    df["wav"] = [utt2wav[u] for u in utts]
    df["text"] = [utt2text[u] for u in utts]
    df["spk"] = [utt2spk[u] for u in utts]
    df["instruct"] = [utt2instruct[u] for u in utts]
    df["utt_embedding"] = [utt2emb[u].tolist() for u in utts]
    df["spk_embedding"] = [spk2emb[utt2spk[u]].tolist() for u in utts]
    df["speech_token"] = [utt2tok[u].tolist() for u in utts]
    return df


def load_kaldi(data_dir: Path):
    wav_scp = (data_dir / "wav.scp").read_text(encoding="utf-8").strip().splitlines()
    text = (data_dir / "text").read_text(encoding="utf-8").strip().splitlines()
    utt2spk = (data_dir / "utt2spk").read_text(encoding="utf-8").strip().splitlines()
    spk2utt = (data_dir / "spk2utt").read_text(encoding="utf-8").strip().splitlines()
    instruct = (data_dir / "instruct").read_text(encoding="utf-8").strip().splitlines()

    utt2wav = {}
    for line in wav_scp:
        k, v = line.split(maxsplit=1)
        utt2wav[k] = v
    utt2text = {}
    for line in text:
        k, v = line.split(maxsplit=1)
        utt2text[k] = " ".join(v.split())  # collapse multiple spaces
    utt2spk_d = {}
    for line in utt2spk:
        k, v = line.split(maxsplit=1)
        utt2spk_d[k] = v
    spk2utt_d = {}
    for line in spk2utt:
        k, *vs = line.split()
        spk2utt_d[k] = vs
    utt2instruct = {}
    for line in instruct:
        k, *rest = line.split(maxsplit=1)
        utt2instruct[k] = rest[0] if rest else ""

    return utt2wav, utt2text, utt2spk_d, spk2utt_d, utt2instruct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--utt2emb", required=True)
    ap.add_argument("--spk2emb", required=True)
    ap.add_argument("--utt2tok", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_utts_per_parquet", type=int, default=500)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  loading Kaldi-style files from {data_dir}")
    utt2wav, utt2text, utt2spk, spk2utt, utt2instruct = load_kaldi(data_dir)
    print(f"  {len(utt2wav)} utts, {len(spk2utt)} spks")

    print(f"  loading pre-extracted features")
    utt2emb = torch.load(args.utt2emb, weights_only=True)
    spk2emb = torch.load(args.spk2emb, weights_only=True)
    utt2tok = torch.load(args.utt2tok, weights_only=True)
    print(f"  utt2emb={len(utt2emb)} spk2emb={len(spk2emb)} utt2tok={len(utt2tok)}")

    # Sanity check
    missing = [u for u in utt2wav if u not in utt2tok or u not in utt2emb]
    if missing:
        print(f"  ! {len(missing)} utts missing features, dropping")
        for u in missing:
            utt2wav.pop(u, None)
            utt2text.pop(u, None)
            utt2spk.pop(u, None)
            utt2instruct.pop(u, None)
    utts = list(utt2wav.keys())

    parquet_list = []
    utt2parquet_list = []
    spk2parquet_list = []
    t0 = time.time()
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        chunk = utts[j: j + args.num_utts_per_parquet]
        shard = build_shard(
            chunk, utt2wav, utt2text, utt2spk, utt2instruct,
            utt2emb, spk2emb, utt2tok,
        )
        # Snappy compression is the sweet spot for size/speed
        parquet_path = out_dir / f"parquet_{i:09d}.parquet"
        shard.to_parquet(parquet_path, compression="snappy", index=False)
        parquet_list.append(str(parquet_path))
        utt2parquet_path = out_dir / f"utt2parquet_{i:09d}.json"
        with open(utt2parquet_path, "w", encoding="utf-8") as f:
            json.dump({u: str(parquet_path) for u in chunk}, f, ensure_ascii=False)
        utt2parquet_list.append(str(utt2parquet_path))
        with open(out_dir / f"spk2parquet_{i:09d}.json", "w", encoding="utf-8") as f:
            json.dump(
                {s: str(parquet_path) for s in set(utt2spk[u] for u in chunk)},
                f, ensure_ascii=False,
            )
        spk2parquet_list.append(str(out_dir / f"spk2parquet_{i:09d}.json"))

        elapsed = time.time() - t0
        done = j + len(chunk)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(utts) - done) / rate if rate > 0 else 0
        print(f"  shard {i+1}: {len(chunk)} utts -> {parquet_path.name} "
              f"[{done}/{len(utts)}  rate={rate:.1f} u/s  eta={eta:.0f}s]")

    # Write the data.list files the CosyVoice3 trainer consumes
    with open(out_dir / "data.list", "w", encoding="utf-8") as f1, \
         open(out_dir / "utt2data.list", "w", encoding="utf-8") as f2, \
         open(out_dir / "spk2data.list", "w", encoding="utf-8") as f3:
        for p in parquet_list:
            f1.write(p + "\n")
        for p in utt2parquet_list:
            f2.write(p + "\n")
        for p in spk2parquet_list:
            f3.write(p + "\n")
    print(f"  wrote data.list / utt2data.list / spk2data.list -> {out_dir}")


if __name__ == "__main__":
    main()