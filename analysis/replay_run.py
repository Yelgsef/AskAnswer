"""Replay a saved run without the model: rebuild every question's user message from the data and the run's saved views/debt
layers, and re-vote its samples. Report where either differs from the trace. A code change that should not change behaviour
must replay with 0 differences.

    python analysis/replay_run.py RUN [RUN ...] --data data/public_test --layers DIR [DIR ...] [predict flags]

RUN dirs hold trace.jsonl (with prompt_user, from --save-prompts). --layers: dirs whose logs/_views_<shop>.json and
_debt_<shop>.json are the layers the run used (the first dir that has a shop wins). Remaining flags are predict.py flags.
"""
import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.hints import DEBT_WORDS  # noqa: E402
from datamate.ingest import norm  # noqa: E402
from datamate.pipeline import Pipeline  # noqa: E402
from datamate.predict import parse_args  # noqa: E402
from datamate.voting import debt_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--layers", nargs="+", type=Path, required=True)
    args, rest = ap.parse_known_args()
    trace = {}
    for run in args.runs:
        for line in (run / "trace.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            trace[r["id"]] = r
    tmp = Path(tempfile.mkdtemp())
    (tmp / "logs").mkdir()
    for d in reversed(args.layers):  # the first dir wins
        for f in (d / "logs").glob("_*.json"):
            shutil.copy(f, tmp / "logs" / f.name)
    a = parse_args(["--model", "none", "--output", str(tmp / "out"), "--views-from", str(tmp)] + rest)
    (a.output / "logs").mkdir(parents=True)
    pipe = Pipeline(a)
    qs = [json.loads(l) for l in (a.data / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    qs = [q for q in qs if q["id"] in trace]
    shops = sorted({q["business_id"] for q in qs})
    loaded = pipe.load_shops(shops)
    pipe.reuse_layers(loaded)
    diff = Counter()
    for b, (db, meta) in loaded.items():
        shop = pipe.setup_shop(b, db, meta)
        todo = [q for q in qs if q["business_id"] == b]
        chats = pipe.build_chats(shop, todo)
        for q, cs in zip(todo, chats):
            t = trace[q["id"]]
            diff["questions"] += 1
            if cs[0][1]["content"] != t.get("prompt_user"):
                diff["prompt differs"] += 1
                if diff["prompt differs"] <= 3:
                    old, new = t.get("prompt_user") or "", cs[0][1]["content"]
                    k = next((i for i, (x, y) in enumerate(zip(old, new)) if x != y), min(len(old), len(new)))
                    print(f"{q['id']} prompt differs at char {k}:\n  trace: {old[max(0, k - 80):k + 120]!r}\n  now:   {new[max(0, k - 80):k + 120]!r}")
            texts = [s["text"] for s in t["samples"]]
            preds = [s["pred"] for s in t["samples"]]
            dq = bool(shop.debt) and a.debt_route and bool(DEBT_WORDS.search(norm(q["question"])))
            pool = debt_pool(texts, 2 * pipe.half, dq) if a.debt_route else list(range(len(texts)))
            (final, vote, conf), = pipe.decide([q], [preds], [pool])
            if final != t["final"]:
                diff["final differs"] += 1
                if diff["final differs"] <= 3:
                    print(f"{q['id']} final: trace {t['final']} now {final}")
            if vote != t["vote"] or abs(conf - t["conf"]) > 1e-9:
                diff["vote differs"] += 1
        db.close()
    print(dict(diff))
    shutil.rmtree(tmp)
    sys.exit(1 if diff["prompt differs"] or diff["final differs"] or diff["vote differs"] else 0)


if __name__ == "__main__":
    main()
