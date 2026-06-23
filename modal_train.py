"""
CosyVoice3 Sinhala SFT on Modal (A100-40GB + persistent Volume)
================================================================

Prerequisites
-------------
    pip install modal
    modal setup          # authenticate once

Quick start
-----------
    # Data prep + features + parquet (CPU, non-preemptible)
    modal run modal_train.py --stage prep

    # Train LLM only (~3–4 h on A100-40GB); use --detach to close laptop
    modal run --detach modal_train.py --stage llm

    # Full pipeline: prep → llm → flow → hifigan → export → inference
    modal run --detach modal_train.py --stage all

    # Resume after preemption (checkpoints live on the Volume)
    modal run --detach modal_train.py --stage llm

GPU non-preemptible
-------------------
Modal does **not** support ``nonpreemptible=True`` on GPU functions — see
https://modal.com/docs/guide/preemption

This script uses:
  * ``gpu="A100-40GB"`` (40 GB variant, not auto-upgraded to 80 GB)
  * Persistent ``modal.Volume`` for ``exp/``, data, and pretrained weights
  * ``modal.Retries(max_retries=10)`` + 24 h timeout per training chunk
  * ``save_every=99999`` by default (epoch-only checkpoints, ~7 GB each)
  * ``nonpreemptible=True`` on CPU-only prep/export steps only

Download artifacts after training::

    modal volume get sinhalatts-cosyvoice3 sft_model ./sft_model

Docs: https://modal.com/docs/guide  |  https://modal.com/docs/guide/gpu
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import modal

# ---------------------------------------------------------------------------
# Config — override via ``modal run modal_train.py --max-epoch 30 ...``
# ---------------------------------------------------------------------------

MODAL_CONFIG = {
    "volume_name": "sinhalatts-cosyvoice3",
    "app_name": "sinhalatts-cosyvoice3",
    "work_root": "/vol/workspace",  # on persistent Volume
    "repo_mount": "/root/SinhalaTTS",  # baked into image from this repo
    "gpu": "A100-40GB",
    "max_epoch": 30,
    # Epoch-only saves: step checkpoints are ~7 GB and fill small disks fast.
    "save_every": 99999,
    "num_workers": 2,
    "skip_flow": False,
    "skip_hifigan": False,
    "feature_device": "cpu",
    "extract_skip_existing": True,
    "average_num": 5,
    "keep_epoch_checkpoints": 2,
    "train_timeout_hours": 24,
    "train_max_retries": 10,
    "prep_timeout_hours": 4,
}

OPENSLR30_TARBALL = "https://www.openslr.org/resources/30/si_lk.tar.gz"
OPENSLR30_LINES = "https://openslr.trmal.net/resources/30/si_lk.lines.txt"
OPENSLR30_MIN_WAVS = 1000
COSYVOICE_GIT = "https://github.com/FunAudioLLM/CosyVoice.git"
HF_PRETRAIN = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

_REPO_ROOT = Path(__file__).resolve().parent
_VOLUME_PATH = Path("/vol")
_WORK_PATH = Path(MODAL_CONFIG["work_root"])
_REPO_PATH = Path(MODAL_CONFIG["repo_mount"])

# Pinned pip deps (match requirements.txt + Colab/Kaggle stack)
_PIP_DEPS = [
    "torch==2.4.1",
    "torchaudio==2.4.1",
    "HyperPyYAML==1.2.3",
    "conformer==0.3.2",
    "diffusers==0.29.0",
    "hydra-core==1.3.2",
    "inflect==7.3.1",
    "librosa==0.10.2",
    "lightning==2.2.4",
    "matplotlib==3.7.5",
    "modelscope==1.20.0",
    "networkx==3.1",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "omegaconf==2.3.0",
    "onnx==1.16.0",
    "onnxruntime-gpu==1.18.0",
    "protobuf==5.28.3",
    "pyarrow==18.1.0",
    "pydantic==2.7.0",
    "pyworld==0.3.4",
    "rich==13.7.1",
    "soundfile==0.12.1",
    "tensorboard==2.14.0",
    "transformers==4.51.3",
    "tiktoken",
    "x-transformers==2.11.24",
    "wetext==0.0.4",
    "wget==3.2",
    "deepspeed==0.15.1",
    "huggingface_hub",
    "hf_transfer",  # required when HF_HUB_ENABLE_HF_TRANSFER=1
]

volume = modal.Volume.from_name(MODAL_CONFIG["volume_name"], create_if_missing=True)
volumes = {_VOLUME_PATH: volume}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ffmpeg", "libsndfile1", "sox")
    .pip_install(
        *_PIP_DEPS,
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(
        str(_REPO_ROOT),
        remote_path=MODAL_CONFIG["repo_mount"],
        copy=True,
        ignore=[
            ".git",
            "__pycache__",
            "*.ipynb",
            ".cursor",
            "exp",
            "sinhala_data",
            "pretrained_models",
            "sft_model",
            "test_outputs",
            "tensorboard",
            "*.tar.gz",
        ],
    )
    .env(
        {
            "USE_TF": "0",
            "USE_FLAX": "0",
            "USE_TORCH": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

app = modal.App(MODAL_CONFIG["app_name"], image=image)

Stage = Literal["prep", "llm", "flow", "hifigan", "train", "export", "infer", "all"]


# ---------------------------------------------------------------------------
# In-container helpers (no Modal decorators — plain Python)
# ---------------------------------------------------------------------------


def _ml_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"USE_TF": "0", "USE_FLAX": "0", "USE_TORCH": "1"})
    if extra:
        env.update(extra)
    return env


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, env=env or _ml_env(), cwd=str(cwd) if cwd else None)


def _paths() -> dict[str, Path]:
    root = _WORK_PATH
    repo = root / "CosyVoice"
    scripts = _REPO_PATH
    stubs = _REPO_PATH / "stubs"
    return {
        "root": root,
        "repo": repo,
        "scripts": scripts,
        "stubs": stubs,
        "configs": _REPO_PATH,
        "pretrain": root / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512",
        "data": root / "sinhala_data",
        "exp": root / "exp" / "cosyvoice3",
        "tb": root / "tensorboard" / "cosyvoice3",
        "sft": root / "sft_model",
        "test_out": root / "test_outputs",
    }


def _ensure_layout() -> dict[str, Path]:
    p = _paths()
    for key in ("root", "exp", "tb", "test_out"):
        p[key].mkdir(parents=True, exist_ok=True)
    (p["stubs"] / "whisper").mkdir(parents=True, exist_ok=True)
    launcher = p["scripts"] / "train_sinhala_sft.sh"
    if launcher.exists():
        os.chmod(launcher, 0o755)
    return p


def _patch_torch_pca() -> None:
    import torch

    if hasattr(torch, "pca_lowrank"):
        return

    def _pca_lowrank(X, q=None, center=True, niter=2):
        if center:
            X = X - X.mean(dim=0, keepdim=True)
        return torch.svd_lowrank(X, q=q or 1, niter=niter)

    torch.pca_lowrank = _pca_lowrank  # type: ignore[attr-defined]


def _patch_constantlr_scheduler_conf(cfg: Path) -> None:
    text = cfg.read_text(encoding="utf-8")
    if "constantlr" not in text:
        return
    text = re.sub(r"^[ \t]*warmup_steps:.*(?:\n|$)", "", text, flags=re.M)
    text = re.sub(
        r"^([ \t]*scheduler_conf:\s*)\n"
        r"(?=[ \t]*(?:max_epoch|grad_clip|accum_grad|log_interval|save_per_step):)",
        r"\1 {}\n",
        text,
        flags=re.M,
    )
    cfg.write_text(text, encoding="utf-8")


def _patch_cosyvoice_pytorch_compat(repo: Path) -> None:
    path = repo / "cosyvoice" / "utils" / "train_utils.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    changed = False
    if "group_join.options._timeout" in text:
        text = text.replace(
            "group_join.options._timeout",
            "(getattr(getattr(group_join, 'options', None), '_timeout', None) "
            "or __import__('datetime').timedelta(seconds=int(os.environ.get('COSYVOICE_JOIN_TIMEOUT', '60'))))",
        )
        changed = True
    anchor = "rank = int(os.environ.get('RANK', 0))"
    body = text.split("def cosyvoice_join", 1)[1].split("def batch_forward", 1)[0]
    if anchor in text and "if world_size <= 1:" not in body:
        text = text.replace(
            anchor + '\n\n    if info_dict["batch_idx"]',
            anchor + "\n\n    if world_size <= 1:\n        return False\n\n    if info_dict[\"batch_idx\"]",
            1,
        )
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")


def _setup_training_configs(repo: Path, configs: Path) -> tuple[Path, Path]:
    conf_dir = repo / "examples" / "libritts" / "cosyvoice3" / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = conf_dir / "cosyvoice3_sinhala_sft.yaml"
    ds_path = conf_dir / "ds_stage2.json"
    shutil.copy(configs / "cosyvoice3_sinhala_sft.yaml", yaml_path)
    shutil.copy(configs / "ds_stage2.json", ds_path)
    _patch_constantlr_scheduler_conf(yaml_path)
    _patch_cosyvoice_pytorch_compat(repo)
    return yaml_path, ds_path


def _bootstrap_whisper(stubs: Path, scripts: Path) -> None:
    for path in (str(scripts), str(stubs)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import whisper  # noqa: F401


def _clone_cosyvoice(repo: Path) -> None:
    if (repo / "cosyvoice").exists():
        print(f"CosyVoice already at {repo}")
        return
    _run(
        ["git", "clone", "--depth=1", "--recursive", COSYVOICE_GIT, str(repo)],
    )


def _download_pretrained(pretrain: Path) -> None:
    if (pretrain / "cosyvoice3.yaml").exists():
        print(f"pretrained at {pretrain}")
        return
    from huggingface_hub import snapshot_download

    print(f"downloading {HF_PRETRAIN} ...")
    snapshot_download(
        repo_id=HF_PRETRAIN,
        local_dir=str(pretrain),
        allow_patterns=["*.json", "*.yaml", "*.pt", "*.onnx", "CosyVoice-BlankEN/*", "*.md"],
    )


def _count_sin_wavs(root: Path) -> int:
    return len(list(root.rglob("sin_*.wav")))


def _find_wav_dir(root: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    seen: set[Path] = set()
    for d in (root, root / "si_lk", root / "si_lk" / "wav", root / "wav"):
        if d.is_dir() and d not in seen:
            seen.add(d)
            n = len(list(d.glob("sin_*.wav")))
            if best is None or n > best[0]:
                best = (n, d)
    for d in root.rglob("*"):
        if d.is_dir() and d not in seen:
            seen.add(d)
            n = len(list(d.glob("sin_*.wav")))
            if n and (best is None or n > best[0]):
                best = (n, d)
    return best[1] if best and best[0] else None


def _ensure_openslr30(root: Path) -> tuple[Path, Path]:
    tarball = root / "si_lk.tar.gz"
    n = _count_sin_wavs(root)
    if n < OPENSLR30_MIN_WAVS:
        print("downloading OpenSLR30 (~700 MB) ...")
        if not tarball.exists():
            _run(["curl", "-L", "-o", str(tarball), OPENSLR30_TARBALL])
        _run(["tar", "-xzf", str(tarball), "-C", str(root)])
    si_lk = root / "si_lk"
    si_lk.mkdir(parents=True, exist_ok=True)
    wav_dir = si_lk / "wav"
    found = _find_wav_dir(root)
    if found and found.resolve() != wav_dir.resolve():
        wav_dir.mkdir(parents=True, exist_ok=True)
        for wav in found.glob("sin_*.wav"):
            dest = wav_dir / wav.name
            if not dest.exists():
                shutil.move(str(wav), str(dest))
    lines = si_lk / "si_lk.lines.txt"
    if not lines.exists():
        _run(["curl", "-L", "-o", str(lines), OPENSLR30_LINES])
    n = len(list(wav_dir.glob("sin_*.wav")))
    if n < OPENSLR30_MIN_WAVS:
        raise FileNotFoundError(f"expected ~1251 wavs, found {n}")
    print(f"OpenSLR30 ready: {n} wavs")
    return si_lk, wav_dir


def _build_train_env(p: dict[str, Path], yaml_cfg: Path, ds_cfg: Path, cfg: dict) -> dict[str, str]:
    return _ml_env(
        {
            "WORK_ROOT": str(p["root"]),
            "REPO_ROOT": str(p["repo"]),
            "PRETRAINED_DIR": str(p["pretrain"]),
            "DATA_DIR": str(p["data"]),
            "EXP_DIR": str(p["exp"]),
            "TB_DIR": str(p["tb"]),
            "CONFIG": str(yaml_cfg),
            "DS_CONFIG": str(ds_cfg),
            "SCRIPTS_DIR": str(p["scripts"]),
            "STUBS_DIR": str(p["stubs"]),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(p["repo"]),
                    str(p["repo"] / "third_party" / "Matcha-TTS"),
                    str(p["scripts"]),
                    str(p["stubs"]),
                ]
            ),
            "NUM_WORKERS": str(cfg["num_workers"]),
            "PREFETCH": "100",
            "SAVE_EVERY": str(cfg["save_every"]),
            "LOG_INTERVAL": "50",
            "MAX_EPOCH": str(cfg["max_epoch"]),
            "AVERAGE_NUM": str(cfg["average_num"]),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )


def _prune_checkpoints(model_dir: Path, keep_whole: int) -> None:
    """Drop old epoch_*_whole and all epoch_*_step_* folders to save Volume space."""
    if not model_dir.is_dir():
        return
    for step_dir in sorted(model_dir.glob("epoch_*_step_*")):
        print(f"prune step checkpoint: {step_dir.name}")
        shutil.rmtree(step_dir, ignore_errors=True)
    wholes = sorted(model_dir.glob("epoch_*_whole"), key=lambda x: x.name)
    while len(wholes) > keep_whole:
        victim = wholes.pop(0)
        print(f"prune old epoch: {victim.name}")
        shutil.rmtree(victim, ignore_errors=True)


def _run_prep_pipeline(cfg: dict) -> None:
    p = _ensure_layout()
    _patch_torch_pca()
    _clone_cosyvoice(p["repo"])
    _download_pretrained(p["pretrain"])
    si_lk, _ = _ensure_openslr30(p["root"])

    py = sys.executable
    _run(
        [
            py,
            str(p["scripts"] / "prepare_sinhala_data.py"),
            "--src_dir",
            str(si_lk),
            "--des_dir",
            str(p["data"]),
            "--min_dur",
            "0.8",
            "--max_dur",
            "22.0",
            "--dev_speaker_ratio",
            "0.10",
            "--out_wav_dir",
            str(p["data"] / "wav_24k"),
        ]
    )
    summary = json.loads((p["data"] / "prep_summary.json").read_text(encoding="utf-8"))
    print(f"manifests: train={summary['train']['utts']} dev={summary['dev']['utts']}")

    camp = p["pretrain"] / "campplus.onnx"
    tok = p["pretrain"] / "speech_tokenizer_v3.onnx"
    for split in ("train", "dev"):
        out = p["data"] / split
        cmd = [
            py,
            str(p["scripts"] / "extract_features.py"),
            "--data_dir",
            str(out),
            "--campplus_onnx",
            str(camp),
            "--speech_tokenizer_onnx",
            str(tok),
            "--device",
            cfg["feature_device"],
            "--save_every",
            "200",
        ]
        if cfg["extract_skip_existing"]:
            cmd.append("--skip_existing")
        t0 = time.time()
        _run(cmd)
        print(f"features {split}: {time.time() - t0:.1f}s")

    for split in ("train", "dev"):
        out = p["data"] / split
        pq_dir = out / "parquet"
        pq_dir.mkdir(exist_ok=True)
        _run(
            [
                py,
                str(p["scripts"] / "build_parquet.py"),
                "--data_dir",
                str(out),
                "--utt2emb",
                str(out / "utt2embedding.pt"),
                "--spk2emb",
                str(out / "spk2embedding.pt"),
                "--utt2tok",
                str(out / "utt2speech_token.pt"),
                "--out_dir",
                str(pq_dir),
                "--num_utts_per_parquet",
                "500",
            ]
        )
        src = pq_dir / "data.list"
        (p["data"] / f"{split}.data.list").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    import pyarrow.parquet as pq

    shards = sorted((p["data"] / "train/parquet").glob("parquet_*.parquet"))
    if not shards:
        raise FileNotFoundError("no parquet shards")
    with pq.ParquetFile(shards[0]) as pf:
        table = pf.read(columns=["text", "speech_token", "spk", "utt_embedding", "audio_data"])
    row = table.slice(0, 1)
    print(
        f"sanity: {shards[0].name} rows={table.num_rows} "
        f"text={row.column('text')[0].as_py()!r}"
    )
    volume.commit()
    print("prep done — volume committed")


def _run_train_stage(stage: str, cfg: dict) -> None:
    import torch

    p = _ensure_layout()
    _patch_torch_pca()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available — training function must use gpu=...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    _bootstrap_whisper(p["stubs"], p["scripts"])
    yaml_cfg, ds_cfg = _setup_training_configs(p["repo"], p["configs"])
    env = _build_train_env(p, yaml_cfg, ds_cfg, cfg)
    launcher = p["scripts"] / "train_sinhala_sft.sh"

    if stage == "train":
        stages = ["llm"]
        if not cfg["skip_flow"]:
            stages.append("flow")
        if not cfg["skip_hifigan"]:
            stages.append("hifigan")
    elif stage == "all":
        stages = ["llm"]
        if not cfg["skip_flow"]:
            stages.append("flow")
        if not cfg["skip_hifigan"]:
            stages.append("hifigan")
    else:
        stages = [stage]

    for st in stages:
        print(f"=== training {st} ===")
        t0 = time.time()
        _run(["bash", str(launcher), st], env=env)
        model_dir = p["exp"] / st / "deepspeed"
        _prune_checkpoints(model_dir, cfg["keep_epoch_checkpoints"])
        volume.commit()
        print(f"=== {st} done in {(time.time() - t0) / 3600:.2f} h — volume committed ===")


def _run_export_and_infer(cfg: dict) -> None:
    p = _ensure_layout()
    py = sys.executable
    exp = p["exp"]
    for m in ("llm", "flow", "hifigan"):
        ckpt = exp / m / "deepspeed" / f"{m}.pt"
        if ckpt.exists():
            print(f"{m}: {ckpt.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"{m}: MISSING")

    p["sft"].mkdir(parents=True, exist_ok=True)
    _run(
        [
            py,
            str(p["scripts"] / "export_sft_model.py"),
            "--pretrained_dir",
            str(p["pretrain"]),
            "--exp_dir",
            str(exp),
            "--out_dir",
            str(p["sft"]),
        ]
    )
    env = _build_train_env(p, Path("/dev/null"), Path("/dev/null"), cfg)
    env["COSYVOICE_REPO"] = str(p["repo"])
    p["test_out"].mkdir(parents=True, exist_ok=True)
    for text, tag in (
        ("ආයුබෝවන්, කොහොමද?", "greeting"),
        ("මම කොළඹ නගරයේ ජීවත් වෙනවා.", "medium"),
        ("ශ්‍රී ලංකාව ඉතා සුන්දර දිවයිනකි.", "long"),
    ):
        out = p["test_out"] / f"test_{tag}.wav"
        _run(
            [
                py,
                str(p["scripts"] / "inference_sinhala.py"),
                "--model_dir",
                str(p["sft"]),
                "--mode",
                "sft",
                "--text",
                text,
                "--data_dir",
                str(p["data"]),
                "--out",
                str(out),
            ],
            env=env,
        )
        print(f"wrote {out}")
    volume.commit()
    print(f"export/infer done -> {p['sft']}")


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

_train_retries = modal.Retries(
    initial_delay=0.0,
    max_retries=MODAL_CONFIG["train_max_retries"],
)


@app.function(
    volumes=volumes,
    cpu=4.0,
    memory=16384,
    timeout=int(MODAL_CONFIG["prep_timeout_hours"] * 3600),
    nonpreemptible=True,
)
def prepare_data(
    max_epoch: int = MODAL_CONFIG["max_epoch"],
    save_every: int = MODAL_CONFIG["save_every"],
    num_workers: int = MODAL_CONFIG["num_workers"],
    feature_device: str = MODAL_CONFIG["feature_device"],
    extract_skip_existing: bool = MODAL_CONFIG["extract_skip_existing"],
) -> None:
    """Download corpus, extract features, build parquet (CPU, non-preemptible)."""
    cfg = {
        **MODAL_CONFIG,
        "max_epoch": max_epoch,
        "save_every": save_every,
        "num_workers": num_workers,
        "feature_device": feature_device,
        "extract_skip_existing": extract_skip_existing,
    }
    _run_prep_pipeline(cfg)


@app.function(
    volumes=volumes,
    gpu=MODAL_CONFIG["gpu"],
    timeout=int(MODAL_CONFIG["train_timeout_hours"] * 3600),
    retries=_train_retries,
    single_use_containers=True,
    # NOTE: nonpreemptible=True is NOT supported for GPU functions (Modal docs).
)
def train_gpu(
    stage: str = "llm",
    max_epoch: int = MODAL_CONFIG["max_epoch"],
    save_every: int = MODAL_CONFIG["save_every"],
    num_workers: int = MODAL_CONFIG["num_workers"],
    skip_flow: bool = MODAL_CONFIG["skip_flow"],
    skip_hifigan: bool = MODAL_CONFIG["skip_hifigan"],
    keep_epoch_checkpoints: int = MODAL_CONFIG["keep_epoch_checkpoints"],
    average_num: int = MODAL_CONFIG["average_num"],
) -> None:
    """Run llm | flow | hifigan | train | all on A100-40GB with auto-retry on preemption."""
    cfg = {
        **MODAL_CONFIG,
        "max_epoch": max_epoch,
        "save_every": save_every,
        "num_workers": num_workers,
        "skip_flow": skip_flow,
        "skip_hifigan": skip_hifigan,
        "keep_epoch_checkpoints": keep_epoch_checkpoints,
        "average_num": average_num,
    }
    _run_train_stage(stage, cfg)


@app.function(
    volumes=volumes,
    gpu=MODAL_CONFIG["gpu"],
    cpu=2.0,
    memory=16384,
    timeout=3600,
    # nonpreemptible not supported with gpu= (Modal preemption docs)
)
def export_and_infer(
    max_epoch: int = MODAL_CONFIG["max_epoch"],
    save_every: int = MODAL_CONFIG["save_every"],
    num_workers: int = MODAL_CONFIG["num_workers"],
    average_num: int = MODAL_CONFIG["average_num"],
) -> None:
    """Export averaged weights and run three Sinhala test sentences."""
    cfg = {
        **MODAL_CONFIG,
        "max_epoch": max_epoch,
        "save_every": save_every,
        "num_workers": num_workers,
        "average_num": average_num,
    }
    _run_export_and_infer(cfg)


@app.local_entrypoint()
def main(
    stage: Stage = "llm",
    max_epoch: int = MODAL_CONFIG["max_epoch"],
    save_every: int = MODAL_CONFIG["save_every"],
    num_workers: int = MODAL_CONFIG["num_workers"],
    skip_flow: bool = MODAL_CONFIG["skip_flow"],
    skip_hifigan: bool = MODAL_CONFIG["skip_hifigan"],
    gpu: str = MODAL_CONFIG["gpu"],
) -> None:
    """
    Orchestrate prep / train / export on Modal.

    Examples::

        modal run modal_train.py --stage prep
        modal run --detach modal_train.py --stage llm
        modal run --detach modal_train.py --stage all
    """
    global MODAL_CONFIG
    MODAL_CONFIG = {
        **MODAL_CONFIG,
        "gpu": gpu,
        "max_epoch": max_epoch,
        "save_every": save_every,
        "num_workers": num_workers,
        "skip_flow": skip_flow,
        "skip_hifigan": skip_hifigan,
    }

    print(
        f"Modal SinhalaTTS | stage={stage} gpu={gpu} "
        f"max_epoch={max_epoch} save_every={save_every} "
        f"volume={MODAL_CONFIG['volume_name']}"
    )
    if gpu != "A100-40GB":
        print(f"note: using gpu={gpu!r} (request A100-40GB for 40 GB VRAM tier)")

    common = dict(
        max_epoch=max_epoch,
        save_every=save_every,
        num_workers=num_workers,
    )

    train_fn = train_gpu.with_options(gpu=gpu)
    export_fn = export_and_infer.with_options(gpu=gpu)

    if stage == "prep":
        prepare_data.spawn(**common).get()
        return

    if stage == "all":
        prepare_data.spawn(**common).get()

    if stage in ("llm", "flow", "hifigan", "train", "all"):
        train_fn.spawn(
            stage="all" if stage == "all" else stage,
            skip_flow=skip_flow,
            skip_hifigan=skip_hifigan,
            **common,
        ).get()
        if stage not in ("all", "export", "infer"):
            return

    if stage in ("export", "infer", "all"):
        export_fn.spawn(**common).get()
