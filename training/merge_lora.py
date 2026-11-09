"""Merge a PEFT LoRA adapter into the bf16 base checkpoint, one tensor at a time (the dev box has 30 GB RAM
for a 24 GB model). Writes a sharded safetensors checkpoint that vLLM loads like the original. Serving the
merged weights avoids vLLM's LoRA path and keeps the parameter count of the base model.

    .venv-train/bin/python training/merge_lora.py adapters/sql_rft_v3 models/Qwen3.5-9B-sql_rft_v3
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 4 * 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--base", default=None, help="base model id or path (default: from adapter_config.json)")
    args = ap.parse_args()

    cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    base = Path(snapshot_download(args.base or cfg["base_model_name_or_path"]))
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])

    lora = {}
    with safe_open(args.adapter / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            lora[k] = f.get_tensor(k)
    # base_model.model.<module path>.lora_A.weight -> <module path>.weight
    deltas = {}
    for k in lora:
        if k.endswith("lora_A.weight"):
            mod = k.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
            deltas[mod + ".weight"] = (lora[k], lora[k.replace("lora_A", "lora_B")])

    files = sorted(base.glob("*.safetensors"))
    args.out.mkdir(parents=True, exist_ok=True)
    index, shard, size, n_shard, merged = {}, {}, 0, 0, 0

    def flush():
        nonlocal shard, size, n_shard
        if shard:
            name = f"model-{n_shard:05d}.safetensors"
            save_file(shard, args.out / name, metadata={"format": "pt"})
            index.update({k: name for k in shard})
            n_shard += 1
            shard, size = {}, 0

    for fp in files:
        with safe_open(fp, "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k in deltas:
                    a, b = deltas.pop(k)
                    t = (t.float() + (b.float() @ a.float()) * scale).to(t.dtype)
                    merged += 1
                shard[k] = t.contiguous()
                size += t.numel() * t.element_size()
                if size >= SHARD_BYTES:
                    flush()
    flush()
    if deltas:
        raise SystemExit(f"{len(deltas)} adapter modules have no base tensor, e.g. {next(iter(deltas))}")
    for name in [p.name for p in base.iterdir() if p.suffix != ".safetensors" and not p.name.endswith(".index.json")]:
        shutil.copy(base / name, args.out / name)
    (args.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    print(f"merged {merged} LoRA modules (scale {scale}) into {n_shard} shards -> {args.out}")


if __name__ == "__main__":
    main()
