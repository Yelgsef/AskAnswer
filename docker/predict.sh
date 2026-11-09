#!/usr/bin/env bash
# Entry point of the image: bash /code/predict.sh, no arguments (organizers' Docker guide, section 5). All settings are fixed in
# /code/flags.sh. Reads /data (read-only); writes into /result:
#   submission.jsonl  every question id; rewritten after each shop by every process, so answers saved before a stop still count
#   logs/<id>.json    one log record per question on one line: id, answer, elapsed_ms, thinking, code, output (+ the full trace)
#   work/shard<i>/    per-GPU working files: the shop layers (views, debt macros) the SQL reads, trace.jsonl, process log
# One datamate.predict process per GPU (data parallel over shops, no tensor parallelism, so no NCCL traffic between the GPUs).
# Each process stops itself at TIMEOUT_SECONDS (5 h 50 min), below the 6 h limit that includes model loading.
set -uo pipefail
cd /code
source ./flags.sh
DATA="${DATA_DIR:-/data}" OUT="${RESULT_DIR:-/result}"
NG="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
T="${TIMEOUT_SECONDS:-21000}"
mkdir -p "$OUT/logs" "$OUT/work"
start=$(date +%s)
echo "datamate: $NG GPU(s), $(grep -c . "$DATA/questions.jsonl") questions, timeout ${T}s"
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i timeout --signal=TERM --kill-after=15s "${T}s" \
    python3 -m datamate.predict --data "$DATA" --output "$OUT/work/shard$i" --shard "$i/$NG" --deadline "$(( T - 900 ))" \
    --log-dir "$OUT/logs" --merge-into "$OUT" \
    --model /code/models/Qwen3.5-9B --lora /code/models/lora --fewshot /code/assets/fewshot_bank.jsonl \
    "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" "$@" > "$OUT/work/shard$i.log" 2>&1 &
done
wait
python3 -m datamate.submission merge "$OUT" --shards "$OUT/work" --no-copy-logs --questions "$DATA/questions.jsonl"
python3 -m datamate.submission fill-logs "$OUT/logs" --questions "$DATA/questions.jsonl"
python3 -m datamate.submission check "$OUT/submission.jsonl" --questions "$DATA/questions.jsonl" || true
grep -h -E "model ready|loaded [0-9]+ shops|view drafts|debt drafts" "$OUT"/work/shard*.log || true
echo "datamate: done in $(( $(date +%s) - start ))s, $(ls "$OUT/logs" | wc -l) question logs"
