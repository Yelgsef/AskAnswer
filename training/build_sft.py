"""SFT rows for LoRA v3 from labeled runs with --save-prompts: python training/build_sft.py RUN [RUN ...] --out data/sft/sql_rft_v3.jsonl [--fewshot BANK] [--max 2400] [--easy 300]
What changes from v2: a question keeps only its solutions on the best route that reached the gold answer (debt macros, then canonical views, then raw tables), so raw SQL
is taught only where nothing else worked. A thinking sample's final SQL also serves as a plain row when plain samples missed that route (plain samples never learned the macros),
debt questions get up to 3 rows and survive the --max cut. --teacher runs (a bigger model on the questions the 9B never solved, 2 plain samples then thinking samples) feed rows the same way, also safe from the cut. Scorer: datamate.scoring.score_one, the one that matches the leaderboard's alias handling.
Plain rows: questions with 0-5 of 8 right samples give up to 2 solutions, questions with 6-8 right give --easy rows in all. Thinking rows: 1 per question, gold answer without a repair turn."""
import os
import argparse, json, random, re, sys
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATAMATE_DATA") or R / "data")   # the notebooks set it (Kaggle and Colab data live elsewhere)
sys.path.insert(0, str(R))
from datamate import prompts
from datamate import fewshot
from datamate.scoring import score
from datamate.ingest import norm
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", type=Path); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--teacher", nargs="*", type=Path, default=[]); ap.add_argument("--fewshot", type=Path); ap.add_argument("--fewshot-k", type=int, default=4); ap.add_argument("--max", type=int, default=2400); ap.add_argument("--easy", type=int, default=300)
a = ap.parse_args()
rng = random.Random(0)
gold = {x["id"]: x for x in map(json.loads, open(DATA / "train/answers.jsonl"))}
rulebook = (DATA / "train/rulebook.md").read_text(encoding="utf-8-sig")
head, *sections = re.split(r"(?m)^(?=## )", rulebook)
VIEWS, DEBT = re.compile(r"(?i)\bv_(sales|orders)(_all)?\b"), re.compile(r"debt_(at|new|repaid)\s*\(")
DEBT_Q = re.compile(r"(^|_)no(_|$)|cong_no|mua_chiu|ghi_no")
RANK = {"macro": 0, "view": 1, "raw": 2}
route = lambda sql: "macro" if DEBT.search(sql) else "view" if VIEWS.search(sql) else "raw"
def book():
    if rng.random() < 0.5:
        return rulebook
    s = sections[:]; rng.shuffle(s)
    return head + "".join(x.rstrip() + "\n\n" for x in s)
bank = fewshot.Bank(R / a.fewshot) if a.fewshot else None
def row(t, reply, thinking=False):
    user = t["prompt_user"]
    if bank and "CANONICAL VIEWS" in user and rng.random() < 0.5:  # as at inference: half of the samples get examples from other shops
        shots = fewshot.block(bank.retrieve(t["question"], t["business_id"], a.fewshot_k, debt="DEBTS (" in user))
        user = user.replace("\n\nQUESTION: ", "\n\n" + shots + "\n\nQUESTION: ", 1)
    return {"system": prompts.SYSTEM + "\n\nRULEBOOK:\n" + book(), "user": user, "reply": reply, "id": t["id"], "thinking": thinking, "route": route(reply.split("```sql")[-1])}
right = lambda t, s: bool(s.get("pred")) and score(s["pred"], gold[t["id"]]) == 1
hard, easy, think, debt_rows, stats = [], [], [], [], Counter()
for run, plain_k in [(x, 8) for x in a.runs] + [(x, 2) for x in a.teacher]:
    for t in map(json.loads, open(R / run / "trace.jsonl", encoding="utf-8")):
        if not t.get("prompt_user"):
            stats["no saved prompt"] += 1
            continue
        plain, finals, thoughts, seen = [], [], [], set()
        n = 0
        for j, s in enumerate(t["samples"]):
            if not right(t, s):
                continue
            if j < plain_k:
                n += 1
                parts = s["text"].split("\n#REPAIR\n")
                reply = parts[-1].strip()
                if len(parts) > 1 and parts[0].startswith("Notes:") and "Notes:" not in reply:  # a repair turn fixes the SQL. Keep the notes of the first turn
                    reply = parts[0].split("TYPE:")[0].strip() + "\n" + reply
                key = fewshot.skeleton(reply.split("```sql")[-1])
                if key not in seen:
                    seen.add(key); plain.append(reply)
            elif "\n#REPAIR\n" not in s["text"] and "</think>" in s["text"]:
                text = re.sub(r"^\s*<think>\s*", "", s["text"]).strip()
                final = text.split("</think>")[-1].strip()
                if "```sql" in final:
                    thoughts.append(text); finals.append(final)
        stats[f"plain {'0' if n == 0 else '1-5' if n <= 5 else '6-8'}/8"] += 1
        everything = plain + finals
        if not everything:
            stats["no right sample"] += 1
            continue
        top = min((route(x.split("```sql")[-1]) for x in everything), key=RANK.get)
        stats[f"best route {top}"] += 1
        on_top = lambda xs: [x for x in xs if route(x.split("```sql")[-1]) == top]
        good = on_top(plain)
        if not good:  # plain samples missed this route: the thinking sample's final SQL teaches it
            good = on_top(finals)[:2]
            stats["plain row from a thinking sample"] += bool(good)
        rng.shuffle(good)
        is_debt = bool(DEBT_Q.search(norm(t["question"])))
        if plain_k != 8:
            stats["teacher solved"] += 1
        rows = [row(t, r) for r in good[:3 if is_debt else 2 if n <= 5 else 1]]
        keep = on_top(thoughts)
        tr = [row(t, rng.choice(keep), thinking=True)] if keep else []
        if is_debt or plain_k != 8:
            debt_rows += rows + tr
        else:
            (hard if n <= 5 else easy).extend(rows)
            think += tr
rng.shuffle(easy); rest = think + hard + easy[:a.easy]; rng.shuffle(rest)
rows = (debt_rows + rest)[:a.max]; rng.shuffle(rows)
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(dict(stats))
print(f"available: debt and teacher {len(debt_rows)}, thinking {len(think)}, hard {len(hard)}, easy {len(easy)} (keep {a.easy}) -> {len(rows)} rows in {a.out}")
print("rows by route", Counter(r["route"] for r in rows).most_common(), "| thinking rows", sum(r["thinking"] for r in rows))
