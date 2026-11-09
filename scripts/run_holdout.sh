#!/usr/bin/env bash
# The submitted setup on the 632-question shop holdout of train (splits/holdout600.txt), one process per GPU, then scored.
#   bash scripts/run_holdout.sh OUT [extra predict flags]      views and debt layers are drafted fresh unless --views-from is given
# A smoke test, not a gate: LoRA v3 was trained on rows of these questions (docs/leaderboard.md).
set -eu
cd "$(dirname "$0")/.."
source scripts/flags.sh
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
OUT=$1; shift
NG=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
mkdir -p "$OUT"
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -m datamate.predict --data data/train --ids splits/holdout600.txt --output "$OUT/shard$i" --shard "$i/$NG" \
    --model "${MODEL_DIR:-models/Qwen3.5-9B}" --lora "${LORA_DIR:-adapters/sql_rft_v3}" --fewshot assets/fewshot_bank.jsonl \
    "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" --save-prompts "$@" > "$OUT/shard$i.log" 2>&1 &
done
wait
"$PY" -m datamate.submission merge "$OUT" --questions data/train/questions.jsonl
grep -F -f splits/holdout600.txt data/train/questions.jsonl > "$OUT/questions.jsonl"   # the question records of the holdout ids
"$PY" -m datamate.scoring --data data/train --pred "$OUT/submission.jsonl" --questions "$OUT/questions.jsonl" --output "$OUT/score.json"
