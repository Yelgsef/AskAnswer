"""Assemble build/training_code/ for the Docker image (organizers' guide, sections 4 and 8), then docker build copies it to
/code/training_code/. Run from the repo root after the training runs exist locally:

    python3 scripts/pack_training_code.py [--out build/training_code] [--skip-model-hashes]

Layout written:
    README.md, requirements.txt, train.sh                  how to rebuild the data and retrain LoRA v1 -> v2 -> v3
    configs/       inference flags of the generation runs, training settings of each adapter, the train splits
    src/           training source (train_lora.py, check_fla.py, merge_lora.py, the no_fla fallback)
    data_generation/  SFT row builders, the notes pass, the few-shot bank builder, and the datamate package that sampled the rows
    data/          competition train split (as provided), SFT data v1/v2/v3 (prompts and replies of every row), few-shot bank
    logs/          training logs and trainer_state.json of the three adapters
    checkpoints/   final adapters v1 and v2 (v3 is /code/models/lora), to verify the continued-training chain
    data_manifest.json   every file: path, size, SHA-256; dataset counts; base and generator model sources and checksums
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1]

ADAPTERS = {
    "sql_rft_v1": {"data": "data/sft/sql_rft_v1.jsonl", "init": None, "max_len": 16384, "lr": 1e-4, "examples": 1636, "dropped": 164, "steps": 205,
                   "runtime_s": 13850, "train_loss": 0.1176},
    "sql_rft_v2": {"data": "data/sft/sql_rft_v2.jsonl", "init": "sql_rft_v1", "max_len": 20000, "lr": 5e-5, "examples": 2398, "dropped": 2, "steps": 300,
                   "runtime_s": 24830, "train_loss": 0.1762},
    "sql_rft_v3": {"data": "data/sft/sql_rft_v3_notes.jsonl", "init": "sql_rft_v2", "max_len": 32768, "lr": 5e-5, "examples": 2909, "dropped": 1,
                   "steps": 364, "runtime_s": 29440, "train_loss": 0.1809},
}
COMMON = {"r": 16, "alpha": 32, "dropout": 0, "epochs": 1, "batch": 1, "grad_accum": 8, "warmup_steps": 10, "scheduler": "cosine",
          "optimizer": "adamw_8bit", "weight_decay": 0.0, "seed": 3407, "precision": "bf16", "base_quantization": "4-bit (Unsloth QLoRA)",
          "no_thinking_default": True, "end": "<|im_end|>"}
FINAL_CKPT = {"sql_rft_v1": "checkpoint-205", "sql_rft_v2": "checkpoint-300", "sql_rft_v3": "checkpoint-364"}

BASE = {"repo": "Qwen/Qwen3.5-9B", "url": "https://huggingface.co/Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "created": "2026-02-27", "license": "Apache-2.0", "parameters": 9653104368, "local": "models/Qwen3.5-9B",
        "role": "base model of every adapter and the inference model; also packaged for inference at /code/models/Qwen3.5-9B"}
TEACHER = {"repo": "avyukth/Qwen3.8-27B-AWQ-INT4", "url": "https://huggingface.co/avyukth/Qwen3.8-27B-AWQ-INT4",
           "revision": "5a2ee524cbc9f3ac5df12729ecf0c4b57f311192", "created": "2026-08-15", "license": "Apache-2.0", "base_model": "Qwen/Qwen3.8-27B",
           "local": "models/Qwen3.8-27B-int4",
           "role": "data generation only (never at inference): solved train questions the 9B could not, through the same pipeline; "
                   "replies kept only when they matched train/answers.jsonl (26 questions in the v3 data)"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "optimizer.pt", "rng_state.pth", "scheduler.pt"))
    else:
        shutil.copy2(src, dst)


def sft_counts(p: Path) -> dict:
    rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    c = {"rows": len(rows), "questions": len({r["id"] for r in rows}),
         "thinking_rows": sum(bool(r.get("thinking")) for r in rows), "plain_rows": sum(not r.get("thinking") for r in rows)}
    if any("route" in r for r in rows):
        c["routes"] = dict(Counter(r.get("route") for r in rows))
    shops = Counter(r["user"].split("# Dữ liệu ", 1)[1].split("\n", 1)[0].strip() for r in rows if "# Dữ liệu " in r["user"])
    c["shops"] = len(shops)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=R / "build/training_code")
    ap.add_argument("--skip-model-hashes", action="store_true", help="leave out the SHA-256 of the base/teacher weights (slow: ~37 GB)")
    a = ap.parse_args()
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # code
    copy(R / "docker/training_code_README.md", out / "README.md")
    copy(R / "training/requirements.txt", out / "requirements.txt")
    copy(R / "docker/train.sh", out / "train.sh")
    for f in ("train_lora.py", "check_fla.py", "merge_lora.py"):
        copy(R / "training" / f, out / "src" / f)
    if (R / "training/no_fla").exists():
        copy(R / "training/no_fla", out / "src/no_fla")
    for f in ("build_sft.py", "add_notes.py", "build_fewshot_bank.py", "build_pool.py"):
        copy(R / "training" / f, out / "data_generation" / f)
    copy(R / "datamate", out / "data_generation/datamate")
    # configs
    copy(R / "scripts/flags.sh", out / "configs/flags.sh")
    for f in ("rft_train24.txt", "holdout600.txt"):
        if (R / "splits" / f).exists():
            copy(R / "splits" / f, out / "configs/splits" / f)
    for name, cfg in ADAPTERS.items():
        (out / "configs").mkdir(exist_ok=True)
        (out / f"configs/{name}.json").write_text(json.dumps({**COMMON, **cfg, "base_model": BASE["repo"], "base_revision": BASE["revision"]}, indent=2) + "\n")
    # data
    copy(R / "data/train", out / "data/competition_train")
    for f in ("sql_rft_v1.jsonl", "sql_rft_v2.jsonl", "sql_rft_v3.jsonl", "sql_rft_v3_notes.jsonl"):
        copy(R / "data/sft" / f, out / "data/sft" / f)
    copy(R / "assets/fewshot_bank.jsonl", out / "data/fewshot_bank.jsonl")
    # logs and checkpoints
    for name, ck in FINAL_CKPT.items():
        copy(R / f"runs/lora/{name}.log", out / f"logs/{name}.log")
        copy(R / f"runs/lora/{name}/{ck}/trainer_state.json", out / f"logs/{name}_trainer_state.json")
    for name in ("sql_rft_v1", "sql_rft_v2"):
        copy(R / f"runs/lora/{name}/adapter", out / f"checkpoints/{name}")

    # manifest
    files = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            files.append({"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    datasets = {f"data/sft/{n}": sft_counts(out / "data/sft" / n) for n in ("sql_rft_v1.jsonl", "sql_rft_v2.jsonl", "sql_rft_v3.jsonl", "sql_rft_v3_notes.jsonl")}
    train_q = sum(1 for line in (out / "data/competition_train/questions.jsonl").open(encoding="utf-8-sig") if line.strip())
    datasets["data/competition_train"] = {"questions": train_q, "businesses": len([d for d in (out / "data/competition_train/data").iterdir() if d.is_dir()]),
                                          "source": "Zalo AI Challenge 2026 DataMate train split, as provided (questions, answers, data, rulebook, calendar)"}
    models = {"base": dict(BASE), "teacher": dict(TEACHER)}
    for m in models.values():
        local = R / m.pop("local")
        if local.exists() and not a.skip_model_hashes:
            m["files"] = [{"name": str(p.relative_to(local)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(local.rglob("*")) if p.is_file()]
    manifest = {"description": "Training code, data and evidence of the DataMate LoRA adapters (v1 -> v2 -> v3; v3 is the inference adapter at /code/models/lora)",
                "adapters": {n: {**COMMON, **c, "final_checkpoint": FINAL_CKPT[n]} for n, c in ADAPTERS.items()},
                "inference_adapter": {"path": "/code/models/lora", "sha256": sha256(R / "adapters/sql_rft_v3/adapter_model.safetensors"), "parameters": 43278336},
                "models": models, "datasets": datasets, "files": files}
    (out / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    total = sum(f["bytes"] for f in files)
    print(f"{out}: {len(files)} files, {total / 1e9:.2f} GB; datasets: " + ", ".join(f"{k} {v.get('rows', v.get('questions'))}" for k, v in datasets.items()))


if __name__ == "__main__":
    main()
