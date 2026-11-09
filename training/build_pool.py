"""Training pool and teacher work list after the view fixes of 2026-11-05 (train data only, labels from train answers).
python training/build_pool.py --fresh runs/sql_v8_gen7 --old runs/sql_v5_gen24 runs/sql_v6_holdout632 --teacher-keep runs/sql_v5_teacher_ids_00
- runs/sql_v8_pool/trace.jsonl: old runs without the shops of --fresh, plus --fresh
- runs/sql_v8_teacher_keep/trace.jsonl: teacher questions on shops whose views did not change
- runs/analysis/v8_teacher_ids_NN.txt: pool questions no 9B sample solved and no kept teacher question covers, 40 per chunk, shops mixed
- runs/views_v8_teacher/logs: the views and debt layers each pool shop was sampled with (a rejected view as no view, as at k=2 it would pass)"""
import os
import argparse, json, random, shutil, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATAMATE_DATA") or R / "data")   # the notebooks set it (Kaggle and Colab data live elsewhere)
sys.path.insert(0, str(R))
from datamate.scoring import score

ap = argparse.ArgumentParser()
ap.add_argument("--fresh", type=Path, required=True); ap.add_argument("--old", type=Path, nargs="+", required=True); ap.add_argument("--teacher-keep", type=Path, nargs="*", default=[])
a = ap.parse_args()
gold = {x["id"]: x for x in map(json.loads, open(DATA / "train/answers.jsonl"))}
fresh = [json.loads(l) for l in open(R / a.fresh / "trace.jsonl", encoding="utf-8")]
redone = {t["business_id"] for t in fresh}
pool, src_of = [], {}
for run in a.old:
    for line in open(R / run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line)
        if t["business_id"] not in redone:
            pool.append(line if line.endswith("\n") else line + "\n"); src_of[t["business_id"]] = run
for t in fresh:
    src_of[t["business_id"]] = a.fresh
pool += [json.dumps(t, ensure_ascii=False) + "\n" for t in fresh]
out = R / "runs/sql_v8_pool"; out.mkdir(parents=True, exist_ok=True)
(out / "trace.jsonl").write_text("".join(pool), encoding="utf-8")
keep, kept_ids = [], set()
for run in a.teacher_keep:
    for line in open(R / run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line)
        if t["business_id"] not in redone:
            keep.append(line if line.endswith("\n") else line + "\n"); kept_ids.add(t["id"])
tk = R / "runs/sql_v8_teacher_keep"; tk.mkdir(parents=True, exist_ok=True)
(tk / "trace.jsonl").write_text("".join(keep), encoding="utf-8")
hard = []
for line in pool:
    t = json.loads(line)
    if t["id"] not in kept_ids and not any(s.get("pred") and score(s["pred"], gold[t["id"]]) == 1 for s in t["samples"]):
        hard.append(t["id"])
random.Random(0).shuffle(hard)
(R / "runs/analysis").mkdir(parents=True, exist_ok=True)
for f in (R / "runs/analysis").glob("v8_teacher_ids_*.txt"):
    f.unlink()
for i in range(0, len(hard), 40):
    (R / f"runs/analysis/v8_teacher_ids_{i // 40:02d}.txt").write_text("\n".join(hard[i:i + 40]) + "\n")
dst = R / "runs/views_v8_teacher/logs"; dst.mkdir(parents=True, exist_ok=True)
for f in dst.glob("*.json"):
    f.unlink()
for b, run in src_of.items():
    logs = R / run / "logs"
    if run in a.old and not (logs / f"_views_{b}.json").exists():
        logs = R / "runs/views_qty_holdout/logs"  # the holdout run reused these views and debts
    for f in [logs / f"_views_{b}.json", logs / f"_debt_{b}.json"]:
        if not f.exists():
            continue
        if f.name.startswith("_views_"):
            v = json.loads(f.read_text(encoding="utf-8"))
            if not v.get("used"):
                v["sql"] = None
            v.setdefault("drafts", [])
            (dst / f.name).write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            shutil.copy(f, dst / f.name)
print(f"pool {len(pool)} questions ({len(redone)} shops fresh), teacher kept {len(kept_ids)}, hard {len(hard)} in {len(list((R / 'runs/analysis').glob('v8_teacher_ids_*.txt')))} chunks, "
      f"teacher views {len(list(dst.glob('_views_*.json')))} / debts {len(list(dst.glob('_debt_*.json')))}")
