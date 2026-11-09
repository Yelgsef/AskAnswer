#!/usr/bin/env bash
# Retrain the three LoRA adapters from the packaged SFT data, in order (each continues from the previous one). See README.md.
#   bash /code/training_code/train.sh            # inside the image (training venv /opt/venv-train, base model /code/models/Qwen3.5-9B)
# One GPU with >= 24 GB. Measured on 1x RTX PRO 4000 Blackwell 24 GB: v1 3.8 h, v2 6.9 h, v3 8.2 h.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-/opt/venv-train/bin/python}
BASE=${BASE_MODEL:-/code/models/Qwen3.5-9B}
OUT=${OUT_DIR:-/tmp/lora}
COMMON=(--model "$BASE" --r 16 --alpha 32 --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking)
mkdir -p "$OUT"
# The fla kernels (Gated DeltaNet layers) must match the PyTorch reference before they are trusted for training
$PY src/check_fla.py --T 4096
$PY src/train_lora.py data/sft/sql_rft_v1.jsonl --out "$OUT/sql_rft_v1" --max-len 16384 --lr 1e-4 "${COMMON[@]}"
$PY src/train_lora.py data/sft/sql_rft_v2.jsonl --out "$OUT/sql_rft_v2" --init-adapter "$OUT/sql_rft_v1/adapter" --max-len 20000 --lr 5e-5 "${COMMON[@]}"
$PY src/train_lora.py data/sft/sql_rft_v3_notes.jsonl --out "$OUT/sql_rft_v3" --init-adapter "$OUT/sql_rft_v2/adapter" --max-len 32768 --lr 5e-5 \
    "${COMMON[@]}" --completion-logits
echo "adapters in $OUT/sql_rft_v{1,2,3}/adapter; the inference adapter is $OUT/sql_rft_v3/adapter (packaged one: /code/models/lora)"
