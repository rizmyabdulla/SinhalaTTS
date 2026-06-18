"""
Assemble an inference-ready CosyVoice3 model directory from pretrained
assets and SFT-averaged checkpoints.

CosyVoice AutoModel expects a folder with cosyvoice3.yaml, ONNX files,
CosyVoice-BlankEN/, and llm.pt / flow.pt / hift.pt. Training only writes
the averaged .pt files under exp/<model>/deepspeed/; this script merges
them into a single deployable model_dir.

On Linux/Kaggle we symlink the large pretrained tree and copy only the
finetuned weights. On Windows (or if symlinks fail) we fall back to a
full copytree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _populate_pretrained(pretrained: Path, out: Path) -> None:
    """Mirror pretrained_dir into out_dir via symlinks or full copy."""
    out.mkdir(parents=True, exist_ok=True)
    use_symlinks = os.name != "nt"
    for item in pretrained.iterdir():
        target = out / item.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if use_symlinks:
            try:
                os.symlink(item, target, target_is_directory=item.is_dir())
                continue
            except OSError:
                use_symlinks = False
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _overlay_checkpoint(src: Path, dst: Path) -> None:
    if not src.exists():
        sys.exit(f"!! missing averaged checkpoint: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export_sft_model(pretrained_dir: Path, exp_dir: Path, out_dir: Path) -> Path:
    pretrained_dir = pretrained_dir.resolve()
    exp_dir = exp_dir.resolve()
    out_dir = out_dir.resolve()

    if not (pretrained_dir / "cosyvoice3.yaml").exists():
        sys.exit(f"!! cosyvoice3.yaml not found under {pretrained_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    _populate_pretrained(pretrained_dir, out_dir)

    overlays = {
        exp_dir / "llm" / "deepspeed" / "llm.pt": out_dir / "llm.pt",
        exp_dir / "flow" / "deepspeed" / "flow.pt": out_dir / "flow.pt",
        exp_dir / "hifigan" / "deepspeed" / "hifigan.pt": out_dir / "hifigan.pt",
    }
    for src, dst in overlays.items():
        _overlay_checkpoint(src, dst)

    # AutoModel loads hift.pt for the vocoder; training averages to hifigan.pt
    hifigan_pt = out_dir / "hifigan.pt"
    hift_pt = out_dir / "hift.pt"
    if hift_pt.exists() or hift_pt.is_symlink():
        hift_pt.unlink()
    try:
        os.symlink(hifigan_pt, hift_pt)
    except OSError:
        shutil.copy2(hifigan_pt, hift_pt)

    required = ["cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt"]
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        sys.exit(f"!! export incomplete, missing: {missing} in {out_dir}")

    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained_dir", required=True,
                    help="Fun-CosyVoice3-0.5B-2512 directory")
    ap.add_argument("--exp_dir", required=True,
                    help="Training output root (exp/cosyvoice3)")
    ap.add_argument("--out_dir", required=True,
                    help="Inference-ready model directory to create")
    args = ap.parse_args()

    out = export_sft_model(
        Path(args.pretrained_dir),
        Path(args.exp_dir),
        Path(args.out_dir),
    )
    print(f"  exported SFT model -> {out}")
    for name in ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt"):
        p = out / name
        size_mb = p.stat().st_size / 1e6 if p.is_file() else 0
        print(f"    {name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
