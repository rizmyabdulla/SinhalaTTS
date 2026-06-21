# %% [markdown]
# # CosyVoice3 Sinhala SFT — Google Colab
#
# Runtime → **GPU**. Upload SinhalaTTS to `/content` or Drive (`MyDrive/SinhalaTTS`).
# Split on `# %%` markers into cells; run in order.
#
# **Drive persistence:** Cell 1 → `mount_drive=True`, `drive_workdir="/content/drive/MyDrive/SinhalaTTS"`.

# %% CELL 1 — Setup
from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_DRIVE_DEFAULT = "/content/drive/MyDrive/SinhalaTTS"
COLAB_CONFIG = {
    "workdir": os.environ.get("SINHALATTS_WORKDIR", "/content"),
    "mount_drive": False,
    "drive_workdir": _DRIVE_DEFAULT,
    "drive_repo_path": _DRIVE_DEFAULT,
    "sinhalatts_git_url": os.environ.get("SINHALATTS_GIT_URL", ""),
    "max_epoch": 30,
    "save_every": 500,
    "num_workers": 4,
    "skip_flow": False,
    "skip_hifigan": False,
    "feature_device": "cpu",
    "extract_skip_existing": True,
}

WORKDIR = Path(COLAB_CONFIG["workdir"])
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)

try:
    SCRIPT_ROOT = Path(__file__).resolve().parent
except NameError:
    SCRIPT_ROOT = WORKDIR

SCRIPTS_DIR = WORKDIR / "scripts"
CONFIGS_DIR = WORKDIR / "configs"
STUBS_DIR = WORKDIR / "stubs"

_REPO_SCRIPTS = (
    "sinhala_normalize.py", "prepare_sinhala_data.py", "extract_features.py",
    "build_parquet.py", "export_sft_model.py", "inference_sinhala.py",
    "train_sinhala_sft.sh", "whisper_mel.py",
)
_REPO_STUBS = ("stubs/whisper/__init__.py", "stubs/whisper/tokenizer.py")
_REPO_CONFIGS = ("cosyvoice3_sinhala_sft.yaml", "ds_stage2.json")

OPENSLR30_TARBALL = "https://www.openslr.org/resources/30/si_lk.tar.gz"
OPENSLR30_LINES = "https://openslr.trmal.net/resources/30/si_lk.lines.txt"
OPENSLR30_MIN_WAVS = 1000

REQS = [
    "HyperPyYAML==1.2.3", "conformer==0.3.2", "diffusers==0.29.0", "hydra-core==1.3.2",
    "inflect==7.3.1", "librosa==0.10.2", "lightning==2.2.4", "matplotlib==3.7.5",
    "modelscope==1.20.0", "networkx==3.1", "numpy==1.26.4", "pandas==2.2.2",
    "omegaconf==2.3.0", "onnx==1.16.0", "onnxruntime-gpu==1.18.0",
    "pyarrow==18.1.0", "pydantic==2.7.0", "pyworld==0.3.4", "rich==13.7.1",
    "soundfile==0.12.1", "tensorboard==2.14.0", "transformers==4.51.3", "tiktoken",
    "x-transformers==2.11.24", "wetext==0.0.4", "wget==3.2", "deepspeed==0.15.1",
    "huggingface_hub",
]

PYTHON = sys.executable


def _mount_drive_if_needed() -> None:
    if not COLAB_CONFIG["mount_drive"]:
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    if not COLAB_CONFIG.get("drive_repo_path"):
        COLAB_CONFIG["drive_repo_path"] = COLAB_CONFIG["drive_workdir"]
    root = Path(COLAB_CONFIG["drive_workdir"])
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(COLAB_CONFIG["drive_repo_path"])
    if not repo.exists():
        raise FileNotFoundError(f"Upload SinhalaTTS to Drive first: {repo}")
    print(f"[1] Drive: {root}")


def _repo_roots() -> list[Path]:
    roots = [WORKDIR, SCRIPT_ROOT]
    if COLAB_CONFIG.get("drive_repo_path"):
        roots.append(Path(COLAB_CONFIG["drive_repo_path"]))
    return list(dict.fromkeys(roots))


def _find_repo_file(name: str) -> Path | None:
    for root in _repo_roots():
        p = root / name
        if p.exists():
            return p
    return None


def _stage_file(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = _find_repo_file(name)
    if src is None:
        if dest.exists():
            return dest
        raise FileNotFoundError(
            f"Missing {name}. Upload repo to /content or set drive_repo_path."
        )
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def storage_root() -> Path:
    if COLAB_CONFIG["mount_drive"]:
        return Path(COLAB_CONFIG["drive_workdir"])
    return WORKDIR


def artifact_dir(name: str) -> Path:
    p = storage_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_repo_layout() -> None:
    url = COLAB_CONFIG.get("sinhalatts_git_url") or ""
    if url and not _find_repo_file("train_sinhala_sft.sh"):
        dest = WORKDIR / "SinhalaTTS_repo"
        if not dest.exists():
            subprocess.check_call(
                ["git", "clone", "--depth=1", url, str(dest)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        global SCRIPT_ROOT
        SCRIPT_ROOT = dest
    for name in _REPO_SCRIPTS:
        _stage_file(name, SCRIPTS_DIR / name)
    for name in _REPO_STUBS:
        _stage_file(name, WORKDIR / name)
    for name in _REPO_CONFIGS:
        _stage_file(name, CONFIGS_DIR / name)
    os.chmod(SCRIPTS_DIR / "train_sinhala_sft.sh", 0o755)


def bootstrap_whisper_stub() -> None:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (STUBS_DIR / "whisper").mkdir(parents=True, exist_ok=True)
    for src, dst in (
        ("whisper_mel.py", SCRIPTS_DIR / "whisper_mel.py"),
        ("stubs/whisper/__init__.py", STUBS_DIR / "whisper" / "__init__.py"),
        ("stubs/whisper/tokenizer.py", STUBS_DIR / "whisper" / "tokenizer.py"),
    ):
        if _find_repo_file(src):
            _stage_file(src, dst)
    for path in (str(SCRIPTS_DIR), str(STUBS_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


_PROTOBUF_VERIFY = (
    "import numpy as np\n"
    "assert np.__version__.startswith('1.26.'), f'numpy {np.__version__} (need 1.26.x)'\n"
    "import google.protobuf as pb\n"
    "from google.protobuf import runtime_version  # noqa: F401\n"
    "import transformers.models.qwen2.modeling_qwen2  # noqa: F401\n"
    "print(pb.__version__)"
)
# runtime_version exists only in protobuf >= 5.24 (not 4.25.x).
_PROTOBUF_SPECS = ("protobuf==5.28.3", "protobuf==6.30.2")


def _ml_import_env() -> dict[str, str]:
    """Keep transformers off TensorFlow/JAX (Colab preinstalls both)."""
    env = os.environ.copy()
    env.update({"USE_TF": "0", "USE_FLAX": "0", "USE_TORCH": "1"})
    return env


def _pin_numpy_stack() -> None:
    """Protobuf 5.x pip often upgrades numpy to 2.x; our stack needs 1.26.x wheels."""
    subprocess.check_call([
        PYTHON, "-m", "pip", "install", "-q", "--no-cache-dir", "--force-reinstall",
        "numpy==1.26.4",
    ])
    subprocess.check_call([
        PYTHON, "-m", "pip", "install", "-q", "--no-cache-dir", "--force-reinstall",
        "pandas==2.2.2", "pyarrow==18.1.0", "matplotlib==3.7.5",
    ])


def repair_protobuf() -> str:
    """Install protobuf 5.x+ for transformers/qwen2 (needs runtime_version).

    Colab often ships protobuf 3.x/4.x and/or the PyPI ``google`` metapackage.
    Verify in a subprocess — the notebook kernel may cache stale google.protobuf.
    """
    for pkg in ("google", "protobuf"):
        subprocess.run(
            [PYTHON, "-m", "pip", "uninstall", "-y", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    last_err = ""
    verify_env = _ml_import_env()
    for spec in _PROTOBUF_SPECS:
        subprocess.check_call([
            PYTHON, "-m", "pip", "install", "-q", "--no-cache-dir", spec,
        ])
        _pin_numpy_stack()
        r = subprocess.run(
            [PYTHON, "-c", _PROTOBUF_VERIFY],
            capture_output=True, text=True, env=verify_env,
        )
        if r.returncode == 0:
            lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
            return lines[-1]
        last_err = (r.stderr or r.stdout or "").strip()
        subprocess.run(
            [PYTHON, "-m", "pip", "uninstall", "-y", "protobuf"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    raise RuntimeError(f"protobuf repair failed:\n{last_err}")


def verify_protobuf_subprocess() -> str:
    verify_env = _ml_import_env()
    try:
        out = subprocess.check_output(
            [PYTHON, "-c", _PROTOBUF_VERIFY],
            text=True, env=verify_env,
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return lines[-1]
    except subprocess.CalledProcessError:
        return repair_protobuf()


def patch_constantlr_scheduler_conf(cfg: Path) -> None:
    text = cfg.read_text(encoding="utf-8")
    if "constantlr" not in text:
        return
    text = re.sub(r"^[ \t]*warmup_steps:.*(?:\n|$)", "", text, flags=re.M)
    text = re.sub(
        r"^([ \t]*scheduler_conf:\s*)\n"
        r"(?=[ \t]*(?:max_epoch|grad_clip|accum_grad|log_interval|save_per_step):)",
        r"\1 {}\n", text, flags=re.M,
    )
    cfg.write_text(text, encoding="utf-8")


def patch_cosyvoice_pytorch_compat(repo: Path) -> None:
    path = repo / "cosyvoice" / "utils" / "train_utils.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    changed = False
    if "group_join.options._timeout" in text:
        text = text.replace(
            "group_join.options._timeout",
            "(getattr(getattr(group_join, 'options', None), '_timeout', None) "
            "or datetime.timedelta(seconds=int(os.environ.get('COSYVOICE_JOIN_TIMEOUT', '60'))))",
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


def ensure_openslr30_corpus(root: Path) -> tuple[Path, Path]:
    tarball = root / "si_lk.tar.gz"
    n = _count_sin_wavs(root)
    if n < OPENSLR30_MIN_WAVS:
        print("[2] downloading OpenSLR30 (~700 MB) ...")
        if not tarball.exists():
            subprocess.check_call(["curl", "-L", "-o", str(tarball), OPENSLR30_TARBALL])
        subprocess.check_call(["tar", "-xzf", str(tarball), "-C", str(root)])
        n = _count_sin_wavs(root)
    else:
        print(f"[2] OpenSLR30 present ({n} wavs)")

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
        subprocess.check_call(["curl", "-L", "-o", str(lines), OPENSLR30_LINES])

    n = len(list(wav_dir.glob("sin_*.wav")))
    if n < OPENSLR30_MIN_WAVS:
        raise FileNotFoundError(f"expected ~1251 wavs, found {n}")
    print(f"[2] corpus ready: {n} wavs")
    return si_lk, wav_dir


def build_train_env(repo: Path, pretrain: Path, data: Path, config: Path, ds: Path) -> dict[str, str]:
    env = _ml_import_env()
    env.update({
        "REPO_ROOT": str(repo),
        "PRETRAINED_DIR": str(pretrain),
        "DATA_DIR": str(data),
        "EXP_DIR": str(artifact_dir("exp/cosyvoice3")),
        "TB_DIR": str(artifact_dir("tensorboard/cosyvoice3")),
        "CONFIG": str(config),
        "DS_CONFIG": str(ds),
        "SCRIPTS_DIR": str(SCRIPTS_DIR),
        "STUBS_DIR": str(STUBS_DIR),
        "PYTHONPATH": os.pathsep.join([
            str(repo), str(repo / "third_party" / "Matcha-TTS"),
            str(SCRIPTS_DIR), str(STUBS_DIR),
        ]),
        "NUM_WORKERS": str(COLAB_CONFIG["num_workers"]),
        "PREFETCH": "100",
        "SAVE_EVERY": str(COLAB_CONFIG["save_every"]),
        "LOG_INTERVAL": "50",
        "MAX_EPOCH": str(COLAB_CONFIG["max_epoch"]),
        "AVERAGE_NUM": "5",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return env


def setup_training_configs(repo: Path) -> tuple[Path, Path]:
    conf_dir = repo / "examples" / "libritts" / "cosyvoice3" / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = conf_dir / "cosyvoice3_sinhala_sft.yaml"
    ds_path = conf_dir / "ds_stage2.json"
    shutil.copy(CONFIGS_DIR / "cosyvoice3_sinhala_sft.yaml", yaml_path)
    shutil.copy(CONFIGS_DIR / "ds_stage2.json", ds_path)
    patch_constantlr_scheduler_conf(yaml_path)
    patch_cosyvoice_pytorch_compat(repo)
    return yaml_path, ds_path


def gc_cuda() -> None:
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_train(launcher: Path, stage: str, env: dict[str, str]) -> None:
    subprocess.check_call(["bash", str(launcher), stage], env=env)
    gc_cuda()


_mount_drive_if_needed()

print(f"[1] Python {sys.version.split()[0]} | installing deps ...")
t0 = time.time()
subprocess.check_call([PYTHON, "-m", "pip", "install", "-q", "--no-cache-dir"] + REQS)
_pin_numpy_stack()
print(f"[1] protobuf {repair_protobuf()} (subprocess ok; restart runtime if in-kernel import fails)")
print(f"[1] pip {time.time() - t0:.1f}s")

ensure_repo_layout()
bootstrap_whisper_stub()

import torch  # noqa: E402

if not hasattr(torch, "pca_lowrank"):
    def _pca_lowrank(X, q=None, center=True, niter=2):
        if center:
            X = X - X.mean(dim=0, keepdim=True)
        return torch.svd_lowrank(X, q=q or 1, niter=niter)
    torch.pca_lowrank = _pca_lowrank

COSYVOICE_DIR = artifact_dir("CosyVoice")
if not (COSYVOICE_DIR / "cosyvoice").exists():
    subprocess.check_call([
        "git", "clone", "--depth=1", "--recursive",
        "https://github.com/FunAudioLLM/CosyVoice.git", str(COSYVOICE_DIR),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[1] cloned CosyVoice -> {COSYVOICE_DIR}")
else:
    print(f"[1] CosyVoice at {COSYVOICE_DIR}")

for p in (COSYVOICE_DIR, COSYVOICE_DIR / "third_party" / "Matcha-TTS"):
    sys.path.insert(0, str(p))

import whisper  # noqa: E402, F401

if torch.cuda.is_available():
    print(f"[1] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[1] WARNING: no GPU — Runtime → Change runtime type → GPU")


# %% CELL 2 — Pretrained model + OpenSLR30
PRETRAIN_DIR = artifact_dir("pretrained_models/Fun-CosyVoice3-0.5B-2512")

if not (PRETRAIN_DIR / "cosyvoice3.yaml").exists():
    print("[2] downloading Fun-CosyVoice3-0.5B-2512 ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local_dir=str(PRETRAIN_DIR),
        allow_patterns=["*.json", "*.yaml", "*.pt", "*.onnx", "CosyVoice-BlankEN/*", "*.md"],
    )
else:
    print(f"[2] pretrained at {PRETRAIN_DIR}")

SINHALA_SRC, _ = ensure_openslr30_corpus(storage_root())


# %% CELL 3 — Manifests
ensure_repo_layout()
sys.path.insert(0, str(SCRIPTS_DIR))
DATA_OUT = artifact_dir("sinhala_data")

subprocess.check_call([
    PYTHON, str(SCRIPTS_DIR / "prepare_sinhala_data.py"),
    "--src_dir", str(SINHALA_SRC), "--des_dir", str(DATA_OUT),
    "--min_dur", "0.8", "--max_dur", "22.0", "--dev_speaker_ratio", "0.10",
    "--out_wav_dir", str(DATA_OUT / "wav_24k"),
])
s = json.loads((DATA_OUT / "prep_summary.json").read_text())
print(f"[3] train {s['train']['utts']} | dev {s['dev']['utts']} utts")


# %% CELL 4 — Features
ensure_repo_layout()
camp = PRETRAIN_DIR / "campplus.onnx"
tok = PRETRAIN_DIR / "speech_tokenizer_v3.onnx"
assert camp.exists() and tok.exists()

for split in ("train", "dev"):
    out = DATA_OUT / split
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "extract_features.py"),
        "--data_dir", str(out), "--campplus_onnx", str(camp),
        "--speech_tokenizer_onnx", str(tok),
        "--device", COLAB_CONFIG["feature_device"], "--save_every", "200",
    ]
    if COLAB_CONFIG["extract_skip_existing"]:
        cmd.append("--skip_existing")
    t0 = time.time()
    subprocess.check_call(cmd)
    print(f"[4] {split} {time.time() - t0:.1f}s")


# %% CELL 5 — Parquet
ensure_repo_layout()

for split in ("train", "dev"):
    out = DATA_OUT / split
    pq_dir = out / "parquet"
    pq_dir.mkdir(exist_ok=True)
    subprocess.check_call([
        PYTHON, str(SCRIPTS_DIR / "build_parquet.py"),
        "--data_dir", str(out),
        "--utt2emb", str(out / "utt2embedding.pt"),
        "--spk2emb", str(out / "spk2embedding.pt"),
        "--utt2tok", str(out / "utt2speech_token.pt"),
        "--out_dir", str(pq_dir), "--num_utts_per_parquet", "500",
    ])

for name in ("train", "dev"):
    src = DATA_OUT / name / "parquet" / "data.list"
    (DATA_OUT / f"{name}.data.list").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
print("[5] parquet ready")


# %% CELL 6 — Sanity check
import pyarrow.parquet as pq

shards = sorted((DATA_OUT / "train/parquet").glob("parquet_*.parquet"))
if not shards:
    raise FileNotFoundError("no parquet shards")
sample_pq = shards[0]
# ParquetFile avoids pq.read_table() → pyarrow.dataset → pandas (ABI mismatch on Colab).
with pq.ParquetFile(sample_pq) as pf:
    table = pf.read(columns=["text", "speech_token", "spk", "utt_embedding", "audio_data"])
row = table.slice(0, 1)
text = row.column("text")[0].as_py()
speech_tok = row.column("speech_token")[0].as_py()
print(f"[6] {sample_pq.name}: {table.num_rows} utts, audio bytes={len(row.column('audio_data')[0].as_py())}")
print(f"[6] text={text!r} spk={row.column('spk')[0].as_py()} utt_emb={len(row.column('utt_embedding')[0].as_py())}")
print(f"[6] speech_token len={len(speech_tok)}")
gc_cuda()
print("[6] ok")


# %% CELL 7 — Train LLM
ensure_repo_layout()
bootstrap_whisper_stub()

yaml_cfg, ds_cfg = setup_training_configs(COSYVOICE_DIR)
launcher = SCRIPTS_DIR / "train_sinhala_sft.sh"
train_env = build_train_env(COSYVOICE_DIR, PRETRAIN_DIR, DATA_OUT, yaml_cfg, ds_cfg)

print(f"[7] protobuf {verify_protobuf_subprocess()} ok")
print("[7] LLM training ...")
run_train(launcher, "llm", train_env)
print("[7] done")


# %% CELL 8 — Train Flow
if COLAB_CONFIG["skip_flow"]:
    print("[8] skipped")
else:
    print("[8] Flow ...")
    run_train(launcher, "flow", train_env)
    print("[8] done")


# %% CELL 9 — Train HiFi-GAN
if COLAB_CONFIG["skip_hifigan"]:
    print("[9] skipped")
else:
    print("[9] HiFi-GAN ...")
    run_train(launcher, "hifigan", train_env)
    print("[9] done")


# %% CELL 10 — Export
ensure_repo_layout()
EXP_ROOT = Path(train_env["EXP_DIR"])
for m in ("llm", "flow", "hifigan"):
    ckpt = EXP_ROOT / m / "deepspeed" / f"{m}.pt"
    print(f"[10] {m}: {ckpt.stat().st_size / 1e6:.1f} MB" if ckpt.exists() else f"[10] {m}: MISSING")

SFT_MODEL_DIR = artifact_dir("sft_model")
subprocess.check_call([
    PYTHON, str(SCRIPTS_DIR / "export_sft_model.py"),
    "--pretrained_dir", str(PRETRAIN_DIR),
    "--exp_dir", str(EXP_ROOT), "--out_dir", str(SFT_MODEL_DIR),
])
print(f"[10] exported -> {SFT_MODEL_DIR}")


# %% CELL 11 — Inference
ensure_repo_layout()
TEST_OUT = WORKDIR / "test_outputs"
TEST_OUT.mkdir(exist_ok=True)
train_env["COSYVOICE_REPO"] = str(COSYVOICE_DIR)

for text, tag in (
    ("ආයුබෝවන්, කොහොමද?", "greeting"),
    ("මම කොළඹ නගරයේ ජීවත් වෙනවා.", "medium"),
    ("ශ්‍රී ලංකාව ඉතා සුන්දර දිවයිනකි.", "long"),
):
    out = TEST_OUT / f"test_{tag}.wav"
    subprocess.check_call([
        PYTHON, str(SCRIPTS_DIR / "inference_sinhala.py"),
        "--model_dir", str(SFT_MODEL_DIR), "--mode", "sft",
        "--text", text, "--data_dir", str(DATA_OUT), "--out", str(out),
    ], env=train_env)
    print(f"[11] {out.name}")

try:
    from IPython.display import Audio, display  # type: ignore
    for wav in sorted(TEST_OUT.glob("*.wav")):
        display(Audio(str(wav), autoplay=False))
except ImportError:
    pass


# %% CELL 12 — Download
bundle = WORKDIR / "sinhala_cosyvoice3_sft.tar.gz"
subprocess.check_call(["tar", "czf", str(bundle), "-C", str(SFT_MODEL_DIR.parent), SFT_MODEL_DIR.name])

try:
    from google.colab import files  # type: ignore
    files.download(str(bundle))
except ImportError:
    pass

if COLAB_CONFIG["mount_drive"]:
    drive_copy = Path(COLAB_CONFIG["drive_workdir"]) / bundle.name
    shutil.copy2(bundle, drive_copy)
    print(f"[12] Drive copy: {drive_copy}")

print(f"\n=== DONE ===\n  model: {SFT_MODEL_DIR}\n  bundle: {bundle}")
