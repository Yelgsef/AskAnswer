#!/usr/bin/env bash
# The submitted setup on the public test, outside Docker, one process per GPU. The file to upload is OUT/submission.jsonl.
#   bash scripts/run_public.sh [OUT]            reuses the views and debt layers of the 0.683 run (configs/public_v9)
#   FRESH=1 bash scripts/run_public.sh [OUT]    drafts every layer again, as the private-test run in Docker does
# PYTHON (default .venv/bin/python), NUM_GPUS, MODEL_DIR, LORA_DIR override the defaults. About 3.5 h on one 24 GB GPU.
set -eu
cd "$(dirname "$0")/.."
source scripts/flags.sh
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
OUT=${1:-runs/public}
NG=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
LAYERS=(--views-from configs/public_v9); [ "${FRESH:-0}" = 1 ] && LAYERS=()
mkdir -p "$OUT"
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -m datamate.predict --data data/public_test --output "$OUT/shard$i" --shard "$i/$NG" \
    --model "${MODEL_DIR:-models/Qwen3.5-9B}" --lora "${LORA_DIR:-adapters/sql_rft_v3}" --fewshot assets/fewshot_bank.jsonl \
    "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" --save-prompts "${LAYERS[@]}" > "$OUT/shard$i.log" 2>&1 &
done
wait
"$PY" -m datamate.submission merge "$OUT" --questions data/public_test/questions.jsonl
"$PY" -m datamate.submission check "$OUT/submission.jsonl" --questions data/public_test/questions.jsonl
