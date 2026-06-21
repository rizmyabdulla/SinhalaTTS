#!/bin/bash
# =============================================================================
# CosyVoice3 Sinhala SFT launcher (Kaggle T4-friendly)
# =============================================================================
#
# Usage:
#   bash train_sinhala_sft.sh llm                 # train the LLM only
#   bash train_sinhala_sft.sh flow                # train the Flow only
#   bash train_sinhala_sft.sh hifigan             # train the HiFi-GAN only
#   bash train_sinhala_sft.sh all                 # LLM, then Flow, then HiFi-GAN
#
# What this does:
#   1. Loads the Fun-CosyVoice3-0.5B-2512 pretrained weights.
#   2. Runs torchrun with deepspeed stage 2 + bf16 (T4-safe).
#   3. Writes checkpoints to <repo>/exp/cosyvoice3/<model>/.
#   4. After all stages, averages the top-N checkpoints (more stable
#      for natural-sounding output).
#
# Why we default to one GPU (not torch DDP multi-GPU) on Kaggle:
#   Kaggle T4x2 has NVLink off, and the LLM's all-gather in DDP becomes
#   the bottleneck. For a 0.5B model on a small corpus, single-GPU with
#   grad accumulation is actually faster than 2x DDP.
#
# Adjust the env block below for your run.
# =============================================================================

set -euo pipefail

# --- env (auto-detect Colab /content vs Kaggle /kaggle/working) -------------
if [ -z "${WORK_ROOT:-}" ]; then
    if [ -d "/content/CosyVoice/cosyvoice" ]; then
        WORK_ROOT="/content"
    elif [ -d "/kaggle/working/CosyVoice/cosyvoice" ]; then
        WORK_ROOT="/kaggle/working"
    else
        WORK_ROOT="/content"
    fi
fi

SCRIPTS_DIR=${SCRIPTS_DIR:-"${WORK_ROOT}/scripts"}
STUBS_DIR=${STUBS_DIR:-"${WORK_ROOT}/stubs"}
REPO_ROOT=${REPO_ROOT:-"${WORK_ROOT}/CosyVoice"}
MATCHA_ROOT="${REPO_ROOT}/third_party/Matcha-TTS"
export PYTHONPATH="${REPO_ROOT}:${MATCHA_ROOT}:${SCRIPTS_DIR}:${STUBS_DIR}:${PYTHONPATH:-}"

PRETRAINED_DIR=${PRETRAINED_DIR:-"${WORK_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B-2512"}
DATA_DIR=${DATA_DIR:-"${WORK_ROOT}/sinhala_data"}
EXP_DIR=${EXP_DIR:-"${WORK_ROOT}/exp/cosyvoice3"}
TB_DIR=${TB_DIR:-"${WORK_ROOT}/tensorboard/cosyvoice3"}
CONFIG=${CONFIG:-"${REPO_ROOT}/examples/libritts/cosyvoice3/conf/cosyvoice3_sinhala_sft.yaml"}
DS_CONFIG=${DS_CONFIG:-"${REPO_ROOT}/examples/libritts/cosyvoice3/conf/ds_stage2.json"}
NUM_WORKERS=${NUM_WORKERS:-2}
PREFETCH=${PREFETCH:-100}
SAVE_EVERY=${SAVE_EVERY:-500}
LOG_INTERVAL=${LOG_INTERVAL:-50}
MAX_EPOCH=${MAX_EPOCH:-30}
AVERAGE_NUM=${AVERAGE_NUM:-5}
STAGE=${1:-llm}

# Kaggle T4x2: only one GPU to avoid DDP bottleneck
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_GPUS=1
else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

cd "${REPO_ROOT}"

# --- sanity checks ----------------------------------------------------------
if [ ! -d "${PRETRAINED_DIR}" ]; then
    echo "!! Pretrained dir not found: ${PRETRAINED_DIR}" >&2
    echo "   Download with: huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ${PRETRAINED_DIR}" >&2
    exit 1
fi
if [ ! -f "${DATA_DIR}/train/parquet/data.list" ] || [ ! -f "${DATA_DIR}/dev/parquet/data.list" ]; then
    echo "!! Data not prepared. Run prepare_sinhala_data.py and build_parquet.py first." >&2
    exit 1
fi
if [ ! -f "${PRETRAINED_DIR}/CosyVoice-BlankEN/config.json" ]; then
    echo "!! Qwen2 base (CosyVoice-BlankEN) missing in pretrained dir" >&2
    exit 1
fi

mkdir -p "${EXP_DIR}" "${TB_DIR}"

# --- build train/dev data lists --------------------------------------------
cat "${DATA_DIR}"/train/parquet/data.list > "${DATA_DIR}/train.data.list"
cat "${DATA_DIR}"/dev/parquet/data.list > "${DATA_DIR}/dev.data.list"

# Apply shell env overrides to yaml (CosyVoice train.py reads train_conf from config)
python - "${CONFIG}" "${SAVE_EVERY}" "${MAX_EPOCH}" "${LOG_INTERVAL}" <<'PY'
import re, sys
from pathlib import Path

cfg = Path(sys.argv[1])
save_every, max_epoch, log_interval = sys.argv[2:5]
text = cfg.read_text(encoding="utf-8")
text = re.sub(r"(save_per_step:\s*)\d+", rf"\g<1>{save_every}", text)
text = re.sub(r"(max_epoch:\s*)\d+", rf"\g<1>{max_epoch}", text)
text = re.sub(r"(log_interval:\s*)\d+", rf"\g<1>{log_interval}", text)
if "constantlr" in text:
    text = re.sub(r"^[ \t]*warmup_steps:.*(?:\n|$)", "", text, flags=re.M)
    text = re.sub(
        r"^([ \t]*scheduler_conf:\s*)\n"
        r"(?=[ \t]*(?:max_epoch|grad_clip|accum_grad|log_interval|save_per_step):)",
        r"\1 {}\n",
        text,
        flags=re.M,
    )
cfg.write_text(text, encoding="utf-8")
PY

# PyTorch 2.x: ProcessGroup.options removed; single-GPU Kaggle skips monitored_barrier
python - "${REPO_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "cosyvoice" / "utils" / "train_utils.py"
if not path.exists():
    sys.exit(0)
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
if anchor in text and "if world_size <= 1:" not in text.split("def cosyvoice_join", 1)[1].split("def batch_forward", 1)[0]:
    text = text.replace(
        anchor + '\n\n    if info_dict["batch_idx"]',
        anchor + "\n\n    if world_size <= 1:\n        return False\n\n    if info_dict[\"batch_idx\"]",
        1,
    )
    changed = True
if changed:
    path.write_text(text, encoding="utf-8")
PY

# --- launch -----------------------------------------------------------------
train_engine=deepspeed
job_id=$((RANDOM % 9999))
dist_backend=nccl

train_model() {
    local model=$1
    local ckpt="${PRETRAINED_DIR}/${model}.pt"
    local model_dir="${EXP_DIR}/${model}/${train_engine}"
    local tb_dir="${TB_DIR}/${model}/${train_engine}"
    mkdir -p "${model_dir}" "${tb_dir}"
    echo "=========================================================="
    echo "  TRAINING MODEL: ${model}"
    echo "  pretrained:     ${ckpt}"
    echo "  output:         ${model_dir}"
    echo "  tensorboard:    ${tb_dir}"
    echo "=========================================================="

    local free_port
    free_port=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

    # The LLM uses Qwen2-BlankEN; flow/hift are pure DiT/HiFT.
    torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
        --rdzv_id=${job_id} --rdzv_backend="c10d" --rdzv_endpoint="localhost:${free_port}" \
        "${REPO_ROOT}/cosyvoice/bin/train.py" \
        --train_engine "${train_engine}" \
        --config "${CONFIG}" \
        --train_data "${DATA_DIR}/train.data.list" \
        --cv_data "${DATA_DIR}/dev.data.list" \
        --qwen_pretrain_path "${PRETRAINED_DIR}/CosyVoice-BlankEN" \
        --onnx_path "${PRETRAINED_DIR}" \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --model_dir "${model_dir}" \
        --tensorboard_dir "${tb_dir}" \
        --ddp.dist_backend "${dist_backend}" \
        --num_workers "${NUM_WORKERS}" \
        --prefetch "${PREFETCH}" \
        --pin_memory \
        --use_amp \
        --deepspeed_config "${DS_CONFIG}" \
        --deepspeed.save_states "model+optimizer"
}

average_ckpt() {
    local model=$1
    local model_dir="${EXP_DIR}/${model}/${train_engine}"
    local dst="${model_dir}/${model}.pt"
    python "${REPO_ROOT}/cosyvoice/bin/average_model.py" \
        --dst_model "${dst}" \
        --src_path "${model_dir}" \
        --num "${AVERAGE_NUM}" \
        --val_best
}

case "${STAGE}" in
    llm)
        train_model llm
        average_ckpt llm
        ;;
    flow)
        train_model flow
        average_ckpt flow
        ;;
    hifigan)
        train_model hifigan
        average_ckpt hifigan
        ;;
    all)
        train_model llm;        average_ckpt llm
        train_model flow;       average_ckpt flow
        train_model hifigan;    average_ckpt hifigan
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use llm | flow | hifigan | all." >&2
        exit 2
        ;;
esac

echo "Done. Next step: run inference_sinhala.py to test."