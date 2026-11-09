# The submitted setup (public 0.683, 2026-11-06), sourced by scripts/run_public.sh, scripts/run_holdout.sh and docker/predict.sh
# so the three always run the same flags. Paths of the model, the adapter and the example bank are set by each caller.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
DATAMATE_FLAGS=(--quant "${QUANT:-fp8}" --k 8 --think-k 4 --context 49152 --think-budget 16384 --notes --max-tokens 2048
                --lora-thinking --debt-views --debt-route --canon-names --answer-unanswerable --gpu-mem "${GPU_MEM:-0.92}")
DATAMATE_ENGINE="${ENGINE:-{\"kv_cache_dtype\":\"fp8\",\"max_num_seqs\":128,\"max_num_batched_tokens\":8192,\"attention_backend\":\"TRITON_ATTN\"\}}"
