"""
Diagnostic / smoke test utility.

Use this when something is wrong and you don't know what.

Examples:
    python diagnose.py norm          # test the Sinhala normalizer on a few lines
    python diagnose.py data data/train  # inspect a parquet shard
    python diagnose.py ckpt /path/to/llm.pt  # show checkpoint shape
    python diagnose.py gpu          # print GPU info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

def cmd_norm(_):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sinhala_normalize import normalize_sinhala, is_valid_transcript

    cases = [
        "කොස් කම කළේ මට නම් මට කියන්න",
        "කොස් කම කළේ මට නම්",
        "෴ වෙනි පටන් 1 2 3",
        "\" කොස් කම කළේ \"",
        "",
        "   ",
        "කෙසේ" + ("\u200d" * 5),  # 5 ZWJ in a row
    ]
    for c in cases:
        out = normalize_sinhala(c)
        print(f"  IN  {c!r:60s} -> OUT {out!r}  valid={is_valid_transcript(out)}")


def cmd_data(args):
    import pyarrow.parquet as pq
    data_dir = Path(args.data_dir)
    parquet_dir = data_dir / "parquet"
    if not parquet_dir.exists():
        print(f"!! {parquet_dir} does not exist")
        return
    pq_files = sorted(parquet_dir.glob("parquet_*.parquet"))
    if not pq_files:
        print(f"!! no parquet files in {data_dir}/parquet")
        return
    sample = pq_files[0]
    df = pq.read_table(sample).to_pandas()
    print(f"  parquet: {sample}")
    print(f"  rows: {len(df)}")
    print(f"  cols: {list(df.columns)}")
    print(f"  first row:")
    for c in df.columns:
        v = df.iloc[0][c]
        if isinstance(v, (bytes, bytearray)):
            v = f"<{len(v)} bytes>"
        elif isinstance(v, list):
            v = f"<list len={len(v)}>"
        print(f"    {c}: {v if not isinstance(v, str) or len(v) < 80 else v[:80] + '...'}")
    # wav.scp/text stats
    wav_scp = (data_dir / "wav.scp").read_text().splitlines()
    text = (data_dir / "text").read_text().splitlines()
    spk2utt = (data_dir / "spk2utt").read_text().splitlines()
    n_spk = len(spk2utt)
    avg_dur = 0.0
    n = 0
    import wave
    for line in wav_scp:
        utt, p = line.split(maxsplit=1)
        try:
            with wave.open(p, "rb") as w:
                avg_dur += w.getnframes() / w.getframerate()
                n += 1
        except Exception:
            pass
    print(f"\n  wav.scp: {len(wav_scp)} utts")
    print(f"  text:    {len(text)} entries")
    print(f"  speakers: {n_spk}")
    if n:
        print(f"  avg dur: {avg_dur/n:.2f}s  ({avg_dur:.0f}s total)")


def cmd_ckpt(args):
    import torch
    p = Path(args.path)
    if not p.exists():
        print(f"!! {p} not found")
        return
    print(f"  checkpoint: {p}  ({p.stat().st_size/1e6:.1f} MB)")
    try:
        state = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        # Could be a state dict
        keys = list(state.keys())[:10]
        total_params = 0
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                total_params += v.numel()
        print(f"  state_dict: {len(state)} tensors, ~{total_params/1e6:.1f}M params")
        print(f"  sample keys: {keys[:5]}")


def cmd_gpu(_):
    import torch
    print(f"  torch:    {torch.__version__}")
    print(f"  cuda:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"  device {i}: {p.name}  mem={total/1e9:.1f}GB  free={free/1e9:.1f}GB")
        print(f"  arch:     {torch.cuda.get_device_capability(0)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("norm", help="Run the Sinhala text normalizer on sample strings")
    p = sp.add_parser("data", help="Inspect a data/train or data/dev partition")
    p.add_argument("data_dir")
    p2 = sp.add_parser("ckpt", help="Print info about a model checkpoint")
    p2.add_argument("path")
    sp.add_parser("gpu", help="Print GPU info")
    args = ap.parse_args()
    {
        "norm": cmd_norm,
        "data": cmd_data,
        "ckpt": cmd_ckpt,
        "gpu": cmd_gpu,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
