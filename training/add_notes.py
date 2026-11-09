"""Adds short notes before the answer of the plain rows of an SFT file, so plain samples learn to reason a little before the SQL. Greedy, base weights, a reply that breaks the format keeps the row as it was.
python training/add_notes.py IN.jsonl OUT.jsonl [--model models/Qwen3.5-9B] [--rulebook data/train/rulebook.md] [--quant fp8|none]"""
import argparse
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from vllm import LLM, SamplingParams

ASK = """Below are a question and the SQL that answers it correctly. Write the notes an analyst makes before writing that SQL: at most 5 short lines, each starting with "- ": which rulebook rules apply, which tables or columns hold the data, the date window, the grain (rows or distinct orders), the status filter and units. Keep what does not apply out. No SQL, no code block, no final answer.

QUESTION: {q}

SQL:
{sql}"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path); ap.add_argument("dst", type=Path)
    ap.add_argument("--model", default=str(R / "models/Qwen3.5-9B")); ap.add_argument("--rulebook", type=Path, default=R / "data/train/rulebook.md")
    ap.add_argument("--quant", default="fp8", help="fp8 (Ada and newer) or none (Ampere)")
    a = ap.parse_args()
    src, dst = a.src, a.dst
    rows = [json.loads(x) for x in src.open(encoding="utf-8")]
    plain = [r for r in rows if not r.get("thinking") and not r["reply"].startswith("Notes:")]
    fp8 = {"quantization": "fp8", "kv_cache_dtype": "fp8"} if a.quant == "fp8" else {}
    llm = LLM(model=a.model, gpu_memory_utilization=0.92, max_model_len=16384, enable_prefix_caching=True, seed=0, language_model_only=True, **fp8,
              max_num_seqs=128, max_num_batched_tokens=8192, attention_backend="TRITON_ATTN")
    rulebook = a.rulebook.read_text(encoding="utf-8-sig")
    chats = []
    for r in plain:
        q = r["user"].rsplit("QUESTION: ", 1)[-1].strip()[:800]
        sql = r["reply"].split("```sql")[-1].split("```")[0].strip()
        chats.append([{"role": "system", "content": "RULEBOOK:\n" + rulebook}, {"role": "user", "content": ASK.format(q=q, sql=sql)}])
    outs = llm.chat(chats, SamplingParams(n=1, temperature=0, max_tokens=220), chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    ok = 0
    for r, o in zip(plain, outs):
        text = o.outputs[0].text.strip()
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if o.outputs[0].finish_reason == "stop" and 1 <= len(lines) <= 6 and all(x.startswith("- ") for x in lines) and "```" not in text and "TYPE:" not in text:
            r["reply"] = "Notes:\n" + "\n".join(lines) + "\n" + r["reply"]
            ok += 1
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"notes on {ok} of {len(plain)} plain rows -> {dst}")


if __name__ == "__main__":
    main()
