# SinhalaTTS — CosyVoice3 Sinhala SFT

Fine-tune [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) on the OpenSLR30 Sinhala TTS corpus for Sinhala text-to-speech.

## Requirements

- Python 3.11+
- CUDA GPU (tested on Kaggle T4, V100, A100, RTX 3090/4090)
- Dependencies in [`requirements.txt`](requirements.txt)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) cloned at runtime (Kaggle notebook) or manually for local runs

## Pipeline

| Step                | Script                    | Output                                                          |
| ------------------- | ------------------------- | --------------------------------------------------------------- |
| 1. Prepare data     | `prepare_sinhala_data.py` | Kaldi-style `wav.scp`, `text`, `utt2spk`, `spk2utt`, `instruct` |
| 2. Extract features | `extract_features.py`     | `utt2embedding.pt`, `spk2embedding.pt`, `utt2speech_token.pt`   |
| 3. Build parquet    | `build_parquet.py`        | `parquet/` shards + `data.list`                                 |
| 4. Train            | `train_sinhala_sft.sh`    | Checkpoints under `exp/cosyvoice3/`                             |
| 5. Export           | `export_sft_model.py`     | `sft_model/` inference bundle                                   |
| 6. Infer            | `inference_sinhala.py`    | WAV files                                                       |

Text normalization lives in [`sinhala_normalize.py`](sinhala_normalize.py) and is applied during data prep and inference.

## Kaggle (recommended)

1. Create a Kaggle notebook with GPU enabled.
2. Copy all SinhalaTTS repo files into `/kaggle/working/` (Kaggle: **Add Data → Upload**, or add this repo as a notebook dataset).
3. Run [`cosyvoice3_sinhala_kaggle.py`](cosyvoice3_sinhala_kaggle.py) (single cell or split at `# CELL N` markers). Run CELL 1 first if using split cells.

The notebook installs deps, clones CosyVoice, downloads [OpenSLR30](https://openslr.org/30/) and pretrained weights, runs the full pipeline, and exports `sft_model/` for download. No separate Kaggle dataset attachment is required.

## Dataset

Training uses the official [OpenSLR SLR30](https://openslr.org/30/) Sinhala multi-speaker TTS corpus (Google, CC BY-SA 4.0). Download manually:

```bash
curl -L -o si_lk.tar.gz https://www.openslr.org/resources/30/si_lk.tar.gz
tar -xzf si_lk.tar.gz
curl -L -o si_lk/si_lk.lines.txt https://openslr.trmal.net/resources/30/si_lk.lines.txt
```

The transcript manifest (`si_lk.lines.txt`) is **not** included in the tarball — download it separately as shown above. Then point `--src_dir` at the extracted `si_lk/` folder.

## Local run

```bash
pip install -r requirements.txt
git clone --depth=1 https://github.com/FunAudioLLM/CosyVoice.git
huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir pretrained_models/Fun-CosyVoice3-0.5B-2512

# Prepare OpenSLR30 data (point --src_dir at extracted si_lk corpus)
python prepare_sinhala_data.py --src_dir /path/to/si_lk --des_dir sinhala_data --out_wav_dir sinhala_data/wav_24k

python extract_features.py \
    --data_dir sinhala_data/train \
    --campplus_onnx pretrained_models/Fun-CosyVoice3-0.5B-2512/campplus.onnx \
    --speech_tokenizer_onnx pretrained_models/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx

python build_parquet.py --data_dir sinhala_data/train \
    --utt2emb sinhala_data/train/utt2embedding.pt \
    --spk2emb sinhala_data/train/spk2embedding.pt \
    --utt2tok sinhala_data/train/utt2speech_token.pt \
    --out_dir sinhala_data/train/parquet

export REPO_ROOT="$(pwd)/CosyVoice"
export PRETRAINED_DIR="$(pwd)/pretrained_models/Fun-CosyVoice3-0.5B-2512"
export DATA_DIR="$(pwd)/sinhala_data"
bash train_sinhala_sft.sh llm   # or: flow | hifigan | all
```

Copy [`cosyvoice3_sinhala_sft.yaml`](cosyvoice3_sinhala_sft.yaml) and [`ds_stage2.json`](ds_stage2.json) into `CosyVoice/examples/libritts/cosyvoice3/conf/` before training.

## Diagnostics

```bash
python diagnose.py norm              # test Sinhala normalizer
python diagnose.py data sinhala_data/train   # inspect parquet shards
python diagnose.py ckpt path/to/llm.pt       # checkpoint info
python diagnose.py gpu               # GPU / CUDA info
```

## Key outputs

- `sinhala_data/` — prepared training data
- `exp/cosyvoice3/` — training checkpoints (LLM, Flow, HiFi-GAN)
- `sft_model/` — merged model ready for inference

## Training config

Training hyperparameters are in [`cosyvoice3_sinhala_sft.yaml`](cosyvoice3_sinhala_sft.yaml). The launcher [`train_sinhala_sft.sh`](train_sinhala_sft.sh) overrides `save_per_step`, `max_epoch`, and `log_interval` from environment variables (`SAVE_EVERY`, `MAX_EPOCH`, `LOG_INTERVAL`).
