"""QLoRA on Qwen3.5-9B with Unsloth (4-bit base, LoRA r16 on attention and MLP). Loss on the reply only.

    .venv-train/bin/python training/train_lora.py data/sft/sql_rft_v3_notes.jsonl --out runs/lora/sql_rft_v4 --model models/Qwen3.5-9B \
        --init-adapter adapters/sql_rft_v3 --max-len 32768 --lr 5e-5 --no-thinking --completion-logits [--resume] [--limit-rows 200]

Each line of the data files: {"system", "user", "reply", "thinking"} (training/build_sft.py). The prompt is rendered with the
model's own chat template, as datamate.predict sends it to vLLM, so the adapter learns the continuation inference asks for.
A row's own "thinking" flag picks the thinking template. --no-thinking sets the default for rows without one.
Checkpoints go to --out every --save-steps steps. --resume continues from the newest one (sessions with a time limit, e.g. Kaggle).
bf16 where the GPU has it (Ampere and newer), else fp16. notebooks/05_train_lora.ipynb drives this script.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from unsloth import FastModel  # noqa: I001  (unsloth patches transformers on import: keep it first)
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

END = "<|im_end|>"   # Qwen closes the model turn with it


def load_rows(paths: list[Path], max_len: int, tokenizer, thinking: bool = True, end: str = END) -> list[dict]:
    rows, dropped = [], 0
    for p in paths:
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=r.get("thinking", thinking))
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(r["reply"] + end, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) > max_len:
                dropped += 1
                continue
            rows.append({"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids})
    print(f"{len(rows)} examples, {dropped} longer than {max_len} tokens dropped")
    return rows


class CompletionLogitsTrainer(Trainer):
    """The same loss, with the LM head run only on the positions that predict reply tokens (logits_to_keep and the matching label
    slice, the model shifts labels inside). The prompt is ~97% of each row, and Unsloth's fused CE otherwise runs the 248k-vocab head
    and a float32 softmax on every prompt position before masking them."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        first = (labels != -100).any(0).nonzero()
        if len(first):
            m = labels.shape[1] - int(first[0]) + 1   # from the last prompt position, which predicts the first reply token
            inputs = {**inputs, "labels": labels[:, -m:].contiguous(), "logits_to_keep": m}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--max-len", type=int, default=20480)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap for a quick throughput / memory check")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--init-adapter", type=Path, help="continue from this adapter's weights (same r and target modules)")
    ap.add_argument("--end", default=END, help="end-of-turn text appended to every reply")
    ap.add_argument("--no-thinking", action="store_true", help="render prompts with enable_thinking=False (non-thinking replies)")
    ap.add_argument("--completion-logits", action="store_true", help="LM head only on reply positions (same loss, less compute)")
    ap.add_argument("--limit-rows", type=int, help="train on the first N rows after the shuffle (a short run to learn or check the setup)")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out, if any")
    args = ap.parse_args()

    model, tokenizer = FastModel.from_pretrained(model_name=args.model, max_seq_length=args.max_len,
                                                 load_in_4bit=True, full_finetuning=False)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.r, lora_alpha=args.alpha, lora_dropout=0, bias="none",
        random_state=args.seed, use_gradient_checkpointing="unsloth")
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(args.init_adapter / "adapter_model.safetensors"))
        print(f"continuing from {args.init_adapter}: {len(res.unexpected_keys)} unexpected keys")
    tok = getattr(tokenizer, "tokenizer", tokenizer)   # the processor wraps the text tokenizer
    rows = load_rows(args.data, args.max_len, tok, thinking=not args.no_thinking, end=args.end)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit_rows] if args.limit_rows else rows
    if not rows:
        raise SystemExit(f"no row fits --max-len {args.max_len}: rows of this pipeline are about 9k-19k tokens (the prompt holds the rulebook and the shop schema); "
                         "16384 keeps ~92%, 24576 keeps all")
    n_tok = sum(len(r["input_ids"]) for r in rows)
    n_lab = sum(sum(x != -100 for x in r["labels"]) for r in rows)
    print(f"tokens: {n_tok:,} total, {n_lab:,} trained ({n_lab / max(n_tok, 1):.1%})")

    bf16 = torch.cuda.is_bf16_supported()
    print(f"precision: {'bf16' if bf16 else 'fp16'}")
    trainer = (CompletionLogitsTrainer if args.completion_logits else Trainer)(
        model=model,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=str(args.out), per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr, warmup_steps=max(1, int(0.05 * args.max_steps)) if args.max_steps > 0 else 10,
            lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.0, bf16=bf16, fp16=not bf16, logging_steps=1,
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=2, report_to="none", seed=args.seed,
            remove_unused_columns=False),
    )
    t = time.time()
    last = max(args.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]), default=None) if args.resume else None
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    dt = time.time() - t
    steps = trainer.state.global_step
    print(f"{steps} steps in {dt:.0f}s; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; "
          f"~{n_tok * (steps * args.grad_accum / max(len(rows), 1)) / dt:,.0f} tokens/s")
    model.save_pretrained(str(args.out / "adapter"))
    tok.save_pretrained(str(args.out / "adapter"))


if __name__ == "__main__":
    main()
