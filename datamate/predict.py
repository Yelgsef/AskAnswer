"""DataMate inference CLI: K sampled DuckDB queries per question, executed, majority-voted, abstaining on low agreement.

    python -m datamate.predict --data data/public_test --output runs/public --model models/Qwen3.5-9B --lora adapters/sql_rft_v3 ...

scripts/run_public.sh and docker/predict.sh hold the flags of the submitted setup. The stages live in pipeline.py.
"""
import argparse
import json
import time
from pathlib import Path

from .llm import Engine
from .pipeline import Pipeline, load_questions, run_in_background
from .submission import merge_shards, write_submission


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, default=16, help="samples per question, split over the two prompts")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--threshold", type=float, default=0.5, help="min vote share to answer")
    p.add_argument("--ids", type=Path, help="optional JSONL/TXT of question ids to run")
    p.add_argument("--limit", type=int)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--quant")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--hide-raw", action="store_true", help="leave the raw sales tables the views map out of the schema, naming only their columns")
    p.add_argument("--notes", action="store_true", help="plain samples must start with short notes before the SQL: on hard holdout questions, right samples 24.9%% -> 44.2%%, votes right/wrong 42/20 -> 77/7")
    p.add_argument("--canon-names", action="store_true", help="write every name of a shop one way when loading it (ingest.canon_spellings)")
    p.add_argument("--views", choices=["llm", "none"], default="llm", help="canonical sales tables written by the LLM per shop (views.py), or none")
    p.add_argument("--view-think", type=int, default=6144, help="thinking-mode token budget for the per-shop view stage (0 = no thinking)")
    p.add_argument("--view-sample-rows", type=int, default=0, help="representative rows per selected source table for view drafts (0 disables)")
    p.add_argument("--views-from", type=Path, help="reuse the view SELECTs an earlier run chose (its logs/_views_<shop>.json) instead of drafting")
    p.add_argument("--repairs", type=int, default=2, help="greedy repair turns for a failed sample (thinking samples included)")
    p.add_argument("--lora-thinking", action="store_true", help="also use --lora for the thinking samples")
    p.add_argument("--lora", type=Path, help="LoRA adapter for the non-thinking SQL samples, TYPE fills and every repair turn (view drafts stay on the base model)")
    p.add_argument("--answer-unanswerable", action="store_true", help="answer unanswerable when that is the voted group")
    p.add_argument("--debt-route", action="store_true", help="debt questions on a shop with the debt layer: only samples calling a debt macro vote")
    p.add_argument("--fewshot", type=Path, help="example bank (training/build_fewshot_bank.py): the second prompt gets similar train examples")
    p.add_argument("--fewshot-k", type=int, default=4)
    p.add_argument("--debt-views", action="store_true", help="canonical debt tables + debt_at(day) written by the LLM per shop (debts.py)")
    p.add_argument("--save-prompts", action="store_true", help="keep each question's user message in the trace (SFT data from train runs)")
    p.add_argument("--shard", default="0/1", help="i/n: run the i-th of n business shards (one process per GPU)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--think-k", type=int, default=0, help="extra thinking-mode samples for questions whose vote share is below --think-below")
    p.add_argument("--think-below", type=float, default=0.75)
    p.add_argument("--think-budget", type=int, default=8192)
    p.add_argument("--deadline", type=float, default=0, help="seconds for this process; thinking is skipped for a shop once the run is behind a pro-rata pace")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=Path, help="one log record per question goes here (Docker: /result/logs); default OUTPUT/logs")
    p.add_argument("--merge-into", type=Path, help="after every shop, merge the sibling shard dirs of --output into DIR/submission.jsonl (Docker: /result)")
    p.add_argument("--engine", type=json.loads, default={}, help='extra vLLM engine kwargs as JSON, e.g. {"kv_cache_dtype": "fp8"}')
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    qs, shops = load_questions(a)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "logs").mkdir(exist_ok=True)
    if a.log_dir:
        a.log_dir.mkdir(parents=True, exist_ok=True)
    write_submission(a.output / "submission.jsonl", qs, {})  # valid all-abstain file in case the run is killed
    if a.merge_into:
        merge_shards(a.merge_into, a.data / "questions.jsonl", shards=a.output.parent, copy_logs=False, quiet=True)
    pipe = Pipeline(a)
    loading = run_in_background(pipe.load_shops, shops)  # DuckDB loading overlaps the model start (the engine is a separate process)
    pipe.engine = Engine(a.model, tp=a.tp, gpu_mem=a.gpu_mem, context=a.context, seed=a.seed, quant=a.quant, lora=a.lora, engine_kwargs=a.engine)
    print(f"model ready in {time.time() - t0:.0f}s; shard {a.shard}: {len(shops)} shops, {len(qs)} questions", flush=True)
    loaded = loading.result()
    print(f"loaded {len(loaded)} shops in {time.time() - t0:.0f}s", flush=True)
    results, n_repair = pipe.run(qs, shops, loaded, t0)
    run = {"args": {k: str(v) for k, v in vars(a).items()}, "questions": len(qs), "seconds": round(time.time() - t0),
           "answered": sum(r["type"] != "abstain" for r in results.values()), "repairs": n_repair}
    (a.output / "run.json").write_text(json.dumps(run, indent=2))
    print(run, flush=True)


if __name__ == "__main__":
    main()
