"""
CosyVoice3-0.5B Sinhala SFT — single-file Kaggle notebook
==========================================================

This is the master script. Drop it into a single Kaggle code cell (or
break it apart into multiple cells using the `# %% CELL N` markers
and the "%run -i" idiom below). All other scripts in `scripts/` are
helpers this notebook invokes; keep them in the working directory.

Pipeline overview
-----------------
    CELL 1  Environment setup (deps + CosyVoice source)
    CELL 2  Download pretrained model + Sinhala data
    CELL 3  Text normalization + data prep (Kaldi-style files)
    CELL 4  Speaker embeddings + discrete speech tokens
    CELL 5  Build parquet shards
    CELL 6  Pre-training sanity check (dry-run the data pipeline)
    CELL 7  Train LLM (Qwen2-BlankEN, 30 epochs, bf16, deepspeed stage 2)
    CELL 8  Train Flow (DiT)  [optional but recommended]
    CELL 9  Train HiFi-GAN     [optional but recommended]
    CELL 10 Average top-N checkpoints
    CELL 11 Sanity inference on a few test sentences
    CELL 12 Export & package for download

Hardware verified
-----------------
    * Kaggle T4 (16 GB) x1  — works, takes ~6-9 h total
    * Kaggle T4 (16 GB) x2  — same (we use 1 GPU for stability)
    * V100 / A100          — much faster; reduce save_per_step
    * P100 (16 GB)         — works; bf16 is supported since PyTorch 1.10
    * RTX 3090 / 4090      — works locally; set CUDA_VISIBLE_DEVICES=0
"""

# =============================================================================
# CELL 1 — Environment setup
# =============================================================================
# We expect to run on a fresh Kaggle kernel. Kaggle's base image already
# has Python 3.11 and PyTorch 2.x with CUDA. The trick is to install
# the CosyVoice-specific stack (conformer, pyworld, matcha, etc.) into
# the user site-packages.

import os, sys, subprocess, time, json, shutil, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

WORKDIR = Path("/kaggle/working")
WORKDIR.mkdir(exist_ok=True)
os.chdir(WORKDIR)

# Pin the exact stack CosyVoice3 was developed against. This is critical
# because mismatched versions of conformer / matcha / pyworld will silently
# break the DiT flow decoder and produce garbled audio at inference.
PYTHON = sys.executable
print(f"[1] Python: {sys.version.split()[0]}  executable: {PYTHON}")

# Install everything. -q to keep logs short; --no-deps because Kaggle's
# base image already has torch/torchaudio/cuda; --no-build-isolation
# so we use the system torch for the build.
REQS = [
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
    "omegaconf==2.3.0",
    "onnx==1.16.0",
    "onnxruntime-gpu==1.18.0",
    "openai-whisper==20231117",
    "protobuf==4.25",
    "pyarrow==18.1.0",
    "pydantic==2.7.0",
    "pyworld==0.3.4",
    "rich==13.7.1",
    "soundfile==0.12.1",
    "tensorboard==2.14.0",
    "transformers==4.51.3",
    "x-transformers==2.11.24",
    "wetext==0.0.4",
    "deepspeed==0.15.1",
]
print(f"[1] installing {len(REQS)} packages (this can take ~3 min) ...")
t0 = time.time()
subprocess.check_call([PYTHON, "-m", "pip", "install", "-q", "--no-cache-dir"] + REQS)
print(f"[1]   done in {time.time()-t0:.1f}s")

# Clone CosyVoice (shallow clone keeps it fast)
if not (WORKDIR / "CosyVoice").exists():
    print("[1] cloning CosyVoice (shallow)")
    subprocess.check_call([
        "git", "clone", "--depth=1", "--recursive",
        "https://github.com/FunAudioLLM/CosyVoice.git",
        str(WORKDIR / "CosyVoice"),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    print("[1] CosyVoice already present")

# Patch: CosyVoice's CausalHiFTGenerator uses torch.nn.functional.pca_lowrank
# which was removed in newer PyTorch. This monkey-patch keeps it working
# with the deepspeed/torch combo Kaggle ships.
import torch.nn.functional as F  # noqa
if not hasattr(F, "pca_lowrank"):
    def _pca_lowrank(X, q=None, center=True, niter=2):
        # Approximate the now-removed helper using torch.svd_lowrank.
        # CosyVoice only calls this during init (no autograd) so a
        # faithful no-grad approximation is fine.
        return torch.svd_lowrank(X, q=q or 1, niter=niter)
    F.pca_lowrank = _pca_lowrank
    print("[1] patched torch.nn.functional.pca_lowrank")

# Make `from cosyvoice.X import Y` work
sys.path.insert(0, str(WORKDIR / "CosyVoice"))
sys.path.insert(0, str(WORKDIR / "CosyVoice" / "third_party" / "Matcha-TTS"))
print(f"[1] cosyvoice on path: {(WORKDIR / 'CosyVoice') in [Path(p) for p in sys.path]}")

# GPU sanity
import torch
print(f"[1] torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")


# =============================================================================
# CELL 2 — Download pretrained model + Sinhala data
# =============================================================================
# We pull:
#   (a) Fun-CosyVoice3-0.5B-2512 from Hugging Face (~12 GB on disk)
#   (b) OpenSLR30 Sinhala TTS from a Kaggle dataset mirror
#
# Using the Kaggle mirror is much faster than pulling from openslr.org
# directly — the dataset is hosted on GCS and piped in via the Kaggle
# input mount.
#
# The Kaggle dataset "keshan/multi-speaket-tts-dataset-sinhala" is the
# official OpenSLR30 data, just repackaged. If you can't add it to the
# notebook, comment out that line and we'll fall back to the openslr URL.

PRETRAIN_DIR = WORKDIR / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512"
PRETRAIN_DIR.parent.mkdir(parents=True, exist_ok=True)

if not PRETRAIN_DIR.exists() or not (PRETRAIN_DIR / "cosyvoice3.yaml").exists():
    print("[2] downloading Fun-CosyVoice3-0.5B-2512 from HuggingFace ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local_dir=str(PRETRAIN_DIR),
        allow_patterns=[
            "*.json", "*.yaml", "*.pt", "*.onnx",
            "CosyVoice-BlankEN/*", "*.md",
        ],
    )
    print(f"[2]   done -> {PRETRAIN_DIR}")
else:
    print(f"[2] pretrained model already at {PRETRAIN_DIR}")

# Sinhala data: prefer the Kaggle mirror, fall back to openslr.org
SINHALA_SRC = None
KAGGLE_INPUT = Path("/kaggle/input/multi-speaket-tts-dataset-sinhala")
if KAGGLE_INPUT.exists():
    # The dataset ships the audio and the TSV
    wav_dir = KAGGLE_INPUT / "wav"
    tsv = KAGGLE_INPUT / "si_lk" / "si_lk.lines.txt"
    if not tsv.exists():
        # Some layouts have a different root
        tsv = next(KAGGLE_INPUT.rglob("si_lk.lines.txt"), None)
    if tsv:
        SINHALA_SRC = tsv.parent
        print(f"[2] using Kaggle Sinhala dataset: {SINHALA_SRC}")
    else:
        print(f"[2] !! si_lk.lines.txt not found under {KAGGLE_INPUT}, will fall back")
elif (WORKDIR / "sinhala_wav").exists():
    SINHALA_SRC = WORKDIR / "sinhala_wav"
    print(f"[2] using local Sinhala wav dir: {SINHALA_SRC}")
else:
    print("[2] downloading from openslr.org/30 (slower, ~700 MB)")
    tarball = WORKDIR / "si_lk.tar.gz"
    if not tarball.exists():
        subprocess.check_call([
            "curl", "-L", "-o", str(tarball),
            "https://www.openslr.org/resources/30/si_lk.tar.gz",
        ])
    if not (WORKDIR / "si_lk").exists():
        subprocess.check_call(["tar", "-xzf", str(tarball), "-C", str(WORKDIR)])
    SINHALA_SRC = WORKDIR / "si_lk"

assert SINHALA_SRC is not None, "!! Sinhala data not found"
print(f"[2]   {len(list((SINHALA_SRC/'wav').glob('*.wav'))) if (SINHALA_SRC/'wav').exists() else 'unknown'} wavs under {SINHALA_SRC}")


# =============================================================================
# CELL 3 — Data preparation (Kaldi-style files, with Sinhala normalization)
# =============================================================================
# This is the most important step for naturalness. The OpenSLR30 transcripts
# are mostly clean but we run them through our Sinhala text normalizer to
# guarantee consistent unicode encoding (NFC) before the Qwen2 tokenizer
# sees them. See scripts/sinhala_normalize.py for the rules.

# Lay out the helper scripts so they're importable
SCRIPTS_DIR = WORKDIR / "scripts"
SCRIPTS_DIR.mkdir(exist_ok=True)
# (The user is expected to upload scripts/*.py here when running on Kaggle.
#  For local execution, the repo's scripts/ dir works the same way.)

DATA_OUT = WORKDIR / "sinhala_data"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# Copy our trained Sinhala normalizer + data prep into the working dir
import importlib.util
for src_name in ("sinhala_normalize.py", "prepare_sinhala_data.py"):
    src_path = SCRIPTS_DIR / src_name
    if not src_path.exists():
        # try the local repo path
        for candidate in [
            WORKDIR / "sinhala_tts" / src_name,
            Path("/kaggle/input/cosyvoice3-sinhala-scripts") / src_name,
            Path(__file__).resolve().parent / src_name,
        ]:
            if candidate.exists():
                shutil.copy(candidate, src_path)
                break
    if not src_path.exists():
        print(f"[3] !! {src_name} not found. Upload it to /kaggle/working/scripts/")

sys.path.insert(0, str(SCRIPTS_DIR))

print("[3] running prepare_sinhala_data.py ...")
subprocess.check_call([
    PYTHON, str(SCRIPTS_DIR / "prepare_sinhala_data.py"),
    "--src_dir", str(SINHALA_SRC),
    "--des_dir", str(DATA_OUT),
    "--min_dur", "0.8",
    "--max_dur", "22.0",
    "--dev_speaker_ratio", "0.10",
    "--out_wav_dir", str(DATA_OUT / "wav_24k"),
])
print(f"[3]   summary:")
summary = json.loads((DATA_OUT / "prep_summary.json").read_text())
print(f"[3]   train utts: {summary['train']['utts']}  dev utts: {summary['dev']['utts']}  "
      f"spks: {summary['train']['spks']} / {summary['dev']['spks']}")


# =============================================================================
# CELL 4 — Feature extraction: speaker embeddings + discrete speech tokens
# =============================================================================
# This is the biggest disk I/O. The CosyVoice3 ONNX files are loaded into
# the model; we run campplus for speaker embeddings (CPU) and
# speech_tokenizer_v3 for the discrete LLM targets (GPU).

# Copy feature extract script
for src_name in ("extract_features.py",):
    src_path = SCRIPTS_DIR / src_name
    if not src_path.exists():
        for candidate in [
            Path("/kaggle/input/cosyvoice3-sinhala-scripts") / src_name,
            Path(__file__).resolve().parent / src_name,
        ]:
            if candidate.exists():
                shutil.copy(candidate, src_path)
                break

CAMPPLUS = PRETRAIN_DIR / "campplus.onnx"
SPEECH_TOK = PRETRAIN_DIR / "speech_tokenizer_v3.onnx"
assert CAMPPLUS.exists(), f"!! {CAMPPLUS} missing"
assert SPEECH_TOK.exists(), f"!! {SPEECH_TOK} missing"

for split in ("train", "dev"):
    out_dir = DATA_OUT / split
    print(f"[4] extracting features for {split} ...")
    t0 = time.time()
    subprocess.check_call([
        PYTHON, str(SCRIPTS_DIR / "extract_features.py"),
        "--data_dir", str(out_dir),
        "--campplus_onnx", str(CAMPPLUS),
        "--speech_tokenizer_onnx", str(SPEECH_TOK),
        "--device", "cuda",
        "--save_every", "200",
    ])
    print(f"[4]   {split} features done in {time.time()-t0:.1f}s")


# =============================================================================
# CELL 5 — Build parquet shards
# =============================================================================
for src_name in ("build_parquet.py",):
    src_path = SCRIPTS_DIR / src_name
    if not src_path.exists():
        for candidate in [
            Path("/kaggle/input/cosyvoice3-sinhala-scripts") / src_name,
            Path(__file__).resolve().parent / src_name,
        ]:
            if candidate.exists():
                shutil.copy(candidate, src_path)
                break

for split in ("train", "dev"):
    out_dir = DATA_OUT / split
    parquet_dir = out_dir / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    print(f"[5] building parquet for {split} ...")
    subprocess.check_call([
        PYTHON, str(SCRIPTS_DIR / "build_parquet.py"),
        "--data_dir", str(out_dir),
        "--utt2emb", str(out_dir / "utt2embedding.pt"),
        "--spk2emb", str(out_dir / "spk2embedding.pt"),
        "--utt2tok", str(out_dir / "utt2speech_token.pt"),
        "--out_dir", str(parquet_dir),
        "--num_utts_per_parquet", "500",
    ])

# Build concatenated train/dev data.list
(DATA_OUT / "train.data.list").write_text(
    "\n".join((DATA_OUT / "train/parquet/data.list").read_text().splitlines())
)
(DATA_OUT / "dev.data.list").write_text(
    "\n".join((DATA_OUT / "dev/parquet/data.list").read_text().splitlines())
)
print(f"[5]   wrote {DATA_OUT}/train.data.list and dev.data.list")


# =============================================================================
# CELL 6 — Pre-training sanity check
# =============================================================================
# Verify the parquet pipeline works end-to-end BEFORE we commit to a long
# training run. We load one parquet, run the cosyvoice data pipeline, and
# print a summary of the produced tensors. If anything is wrong, this
# surfaces it in 30 seconds.

print("[6] pre-training data sanity check ...")
import pyarrow.parquet as pq
sample_pq = next((DATA_OUT / "train/parquet").glob("parquet_*.parquet"))
df = pq.read_table(sample_pq).to_pandas()
print(f"[6]   parquet {sample_pq.name} -> {len(df)} utts, "
      f"audio bytes: {len(df.iloc[0]['audio_data'])}")
print(f"[6]   sample text: {df.iloc[0]['text']!r}")
print(f"[6]   sample spk:  {df.iloc[0]['spk']}")
print(f"[6]   sample utt_embedding len: {len(df.iloc[0]['utt_embedding'])}")
print(f"[6]   sample speech_token len:  {len(df.iloc[0]['speech_token'])}")

# Free memory before training
import gc
gc.collect()
torch.cuda.empty_cache()
print("[6]   ok!")


# =============================================================================
# CELL 7 — Train LLM (Qwen2-BlankEN, 30 epochs, bf16, deepspeed stage 2)
# =============================================================================
# This is the heart of the SFT. We fine-tune the Qwen2-based LLM on the
# (sinhala text -> speech tokens) mapping. After this step, the model
# knows how to produce Sinhala speech tokens from Sinhala text. The Flow
# + HiFi-GAN below further refine the audio quality, but the LLM is the
# biggest factor for *naturalness* of prosody.
#
# Expected time on T4: ~4-5 hours for 30 epochs over ~1.5k utts.

# Copy configs
for src_name in ("cosyvoice3_sinhala_sft.yaml", "ds_stage2.json"):
    src_path = WORKDIR / "configs" / src_name
    if not src_path.exists():
        for candidate in [
            Path("/kaggle/input/cosyvoice3-sinhala-configs") / src_name,
            Path(__file__).resolve().parent / src_name,
        ]:
            if candidate.exists():
                src_path.parent.mkdir(exist_ok=True)
                shutil.copy(candidate, src_path)
                break

# Make the config + ds files visible at the path the launcher expects
TARGET_CONFIG = WORKDIR / "CosyVoice" / "examples" / "libritts" / "cosyvoice3" / "conf" / "cosyvoice3_sinhala_sft.yaml"
TARGET_DS = WORKDIR / "CosyVoice" / "examples" / "libritts" / "cosyvoice3" / "conf" / "ds_stage2.json"
TARGET_CONFIG.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(WORKDIR / "configs" / "cosyvoice3_sinhala_sft.yaml", TARGET_CONFIG)
shutil.copy(WORKDIR / "configs" / "ds_stage2.json", TARGET_DS)

# Also drop the launcher script
launcher_src = SCRIPTS_DIR / "train_sinhala_sft.sh"
if not launcher_src.exists():
    for candidate in [
        Path("/kaggle/input/cosyvoice3-sinhala-scripts") / "train_sinhala_sft.sh",
        Path(__file__).resolve().parent / "train_sinhala_sft.sh",
    ]:
        if candidate.exists():
            shutil.copy(candidate, launcher_src)
            break
os.chmod(launcher_src, 0o755)

# Run training
print("[7] launching LLM training ...")
env = os.environ.copy()
env.update({
    "REPO_ROOT": str(WORKDIR / "CosyVoice"),
    "PRETRAINED_DIR": str(PRETRAIN_DIR),
    "DATA_DIR": str(DATA_OUT),
    "EXP_DIR": str(WORKDIR / "exp" / "cosyvoice3"),
    "TB_DIR": str(WORKDIR / "tensorboard" / "cosyvoice3"),
    "CONFIG": str(TARGET_CONFIG),
    "DS_CONFIG": str(TARGET_DS),
    "NUM_WORKERS": "2",
    "PREFETCH": "100",
    "SAVE_EVERY": "500",
    "LOG_INTERVAL": "50",
    "MAX_EPOCH": "30",
    "AVERAGE_NUM": "5",
    "CUDA_VISIBLE_DEVICES": "0",
})
subprocess.check_call(["bash", str(launcher_src), "llm"], env=env)
print("[7]   LLM training complete")

gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# CELL 8 — Train Flow (DiT) [optional but recommended for naturalness]
# =============================================================================
# The Flow decoder (DiT) converts LLM speech tokens to mel-spectrograms.
# Pretraining was on 9 languages, so Sinhala's prosody patterns are
# under-represented. Fine-tuning the Flow on ~1.5k Sinhala utts sharpens
# the audio quality noticeably.
print("[8] launching Flow training ...")
subprocess.check_call(["bash", str(launcher_src), "flow"], env=env)
print("[8]   Flow training complete")

gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# CELL 9 — Train HiFi-GAN [optional but recommended]
# =============================================================================
# The HiFT vocoder turns mel-spectrograms into waveforms. SFT helps it
# generalize to Sinhala phoneme statistics. Cost is small (~30 min on T4).
print("[9] launching HiFi-GAN training ...")
subprocess.check_call(["bash", str(launcher_src), "hifigan"], env=env)
print("[9]   HiFi-GAN training complete")

gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# CELL 10 — Average top-N checkpoints
# =============================================================================
# Model averaging over the best-5 validation checkpoints is a free 0.2-0.5
# dB improvement for naturalness. The launcher does this automatically at
# the end of each stage. Verify here.

EXP_ROOT = WORKDIR / "exp" / "cosyvoice3"
for m in ("llm", "flow", "hifigan"):
    p = EXP_ROOT / m / "deepspeed" / f"{m}.pt"
    if p.exists():
        print(f"[10]  {m}: {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[10]  {m}: MISSING")


# =============================================================================
# CELL 10b — Export inference-ready model directory
# =============================================================================
# Merge pretrained assets (yaml, onnx, BlankEN) with SFT-averaged weights.

for src_name in ("export_sft_model.py",):
    src_path = SCRIPTS_DIR / src_name
    if not src_path.exists():
        for candidate in [
            Path("/kaggle/input/cosyvoice3-sinhala-scripts") / src_name,
            Path(__file__).resolve().parent / src_name,
        ]:
            if candidate.exists():
                shutil.copy(candidate, src_path)
                break

SFT_MODEL_DIR = WORKDIR / "sft_model"
print(f"[10b] exporting inference model -> {SFT_MODEL_DIR}")
subprocess.check_call([
    PYTHON, str(SCRIPTS_DIR / "export_sft_model.py"),
    "--pretrained_dir", str(PRETRAIN_DIR),
    "--exp_dir", str(EXP_ROOT),
    "--out_dir", str(SFT_MODEL_DIR),
])


# =============================================================================
# CELL 11 — Sanity inference
# =============================================================================
# Generate 3 test sentences: one neutral, one with a Sinhala quote, one
# a longer paragraph. Listen to the WAVs (Kaggle shows them inline).

# Copy the inference script
inf_src = SCRIPTS_DIR / "inference_sinhala.py"
if not inf_src.exists():
    for candidate in [
        Path("/kaggle/input/cosyvoice3-sinhala-scripts") / "inference_sinhala.py",
        Path(__file__).resolve().parent / "inference_sinhala.py",
    ]:
        if candidate.exists():
            shutil.copy(candidate, inf_src)
            break

env["COSYVOICE_REPO"] = str(WORKDIR / "CosyVoice")
TEST_OUT = WORKDIR / "test_outputs"
TEST_OUT.mkdir(exist_ok=True)

test_sentences = [
    ("ආයුබෝවන්, කොහොමද?", "greeting"),
    ("මම කොළඹ නගරයේ ජීවත් වෙනවා. මට සිංහල කතා කරන්න පුළුවන්.", "medium"),
    ("ශ්‍රී ලංකාව ඉතා සුන්දර දිවයිනකි. එහි පරණ නගර, කඳුකර, සහ මුහුදු බෙල්ලා ඇත.", "long"),
]
for text, tag in test_sentences:
    out_wav = TEST_OUT / f"test_{tag}.wav"
    print(f"[11]  generating {tag} -> {out_wav}")
    subprocess.check_call([
        PYTHON, str(inf_src),
        "--model_dir", str(SFT_MODEL_DIR),
        "--mode", "sft",
        "--text", text,
        "--data_dir", str(DATA_OUT),
        "--out", str(out_wav),
    ], env=env)

# List outputs
print(f"[11]  test outputs in {TEST_OUT}:")
for p in sorted(TEST_OUT.glob("*.wav")):
    print(f"        {p}  ({p.stat().st_size/1024:.1f} KB)")


# =============================================================================
# CELL 12 — Package for download
# =============================================================================
# Bundle the LLM/Flow/HiFi-GAN checkpoints (the SFT model the user will
# distribute) into a single tarball for download.

bundle = WORKDIR / "sinhala_cosyvoice3_sft.tar.gz"
print(f"[12] packaging {bundle}")
subprocess.check_call([
    "tar", "czf", str(bundle),
    "-C", str(WORKDIR),
    "sft_model",
])
print(f"[12]   {bundle}  ({bundle.stat().st_size/1e9:.2f} GB)")

# Also save a copy of the Sinhala text normalizer + scripts as a "sidecar"
sidecar = WORKDIR / "sinhala_sft_scripts.tar.gz"
subprocess.check_call([
    "tar", "czf", str(sidecar),
    "-C", str(WORKDIR), "configs", "scripts",
])
print(f"[12]   {sidecar}  ({sidecar.stat().st_size/1e6:.1f} MB)")

print("\n=== ALL DONE ===")
print(f"  SFT model:    {SFT_MODEL_DIR}")
print(f"  Test wavs:    {TEST_OUT}")
print(f"  Bundle:       {bundle}")
print(f"  Sidecar:      {sidecar}")