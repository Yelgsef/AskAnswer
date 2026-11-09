# The DataMate private-test image, laid out as the organizers' guide asks (zac2026_DataMate_docker_guideline_final.pdf):
#   /code/predict.sh  /code/README.md  /code/requirements.txt  /code/src (inference code)  /code/models (Qwen3.5-9B + LoRA v3)
#   /code/assets (few-shot bank)  /code/training_code (training code, data, logs, checkpoints, data_manifest.json)
# Build from the repo root, after `git lfs pull`, the base model in models/Qwen3.5-9B and `python3 scripts/pack_training_code.py`:
#   docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_<team_name>:final .
# Scoring command (guide, section 10):
#   docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
#     -v /path/to/test:/data:ro -v /path/to/output:/result zaic2026_datamate_<team_name>:final /code/predict.sh
#
# Base image: the guide recommends vllm/vllm-openai:v0.10.1.1, which predates Qwen3.5 (Gated DeltaNet layers) and cannot load the
# model. v0.32.0 runs it and supports the RTX 5090 (compute capability 12.0), which the guide allows for an alternative image.
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
# A plain `docker run` starts predict.sh, not the vLLM server of the base image
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_CACHE_DIR=1 NCCL_P2P_DISABLE=1 VLLM_NO_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONPATH=/code/src
WORKDIR /code
COPY docker/requirements.txt /code/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /code/requirements.txt
# Training pins its own torch/transformers: a separate offline-ready venv, so the vLLM environment stays untouched
COPY training/requirements.txt /code/training_code/requirements.txt
RUN python3 -m venv /opt/venv-train \
    && /opt/venv-train/bin/pip install --no-cache-dir -r /code/training_code/requirements.txt \
    && /opt/venv-train/bin/pip install --no-cache-dir fla-core==0.6.0 --no-deps
# Weights first, so a code change does not invalidate their layers
COPY models/Qwen3.5-9B /code/models/Qwen3.5-9B
COPY adapters/sql_rft_v3 /code/models/lora
COPY build/training_code/ /code/training_code/
COPY assets/fewshot_bank.jsonl /code/assets/fewshot_bank.jsonl
COPY docker/README.md /code/README.md
COPY docker/predict.sh scripts/flags.sh docker/model_manifest.json /code/
COPY datamate /code/src/datamate
RUN chmod +x /code/predict.sh
CMD ["bash", "/code/predict.sh"]
