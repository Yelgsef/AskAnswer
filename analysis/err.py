"""Error analysis of holdout runs (train gold only, never at inference).
Usage: python analysis/err.py RUN [RUN2]   (RUN = folder with shard*/trace.jsonl or trace.jsonl)
Prints PA / correct / wrong / abstain / coverage, Oracle@K, where each lost question was lost (no sample right,
vote picked another answer, right vote abstained), breakdowns by gold type and topic, latency, and with RUN2 a paired diff."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.scoring import score  # noqa: E402
from datamate.ingest import norm  # noqa: E402

GOLD = {r["id"]: r for r in map(json.loads, (Path(__file__).resolve().parents[1] / "data/train/answers.jsonl").read_text(encoding="utf-8").splitlines())}
TOPIC = [("debt", r"(?:^|_)(no|cong_no|ghi_no|tra_no|thu_no|con_no|tuoi_no)(?:_|$)"), ("stock", r"ton|het_hang|nhap"),
         ("status", r"huy|tra_hang|hoan|doi_tra"), ("customer", r"khach"), ("product", r"san_pham|mat_hang|ban_chay|mon|thuoc|sp(?:_|$)")]


def load(run):
    run = Path(run)
    files = sorted(run.glob("shard*/trace.jsonl")) or [run / "trace.jsonl"]
    return {r["id"]: r for f in files for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())}


def topic(q):
    t = norm(q)
    return next((k for k, p in TOPIC if re.search(p, t)), "other")


def where(r, g):
    """correct | wrong/abstain split by cause."""
    s = score(r["final"], g)
    if s == 1:
        return "correct"
    preds = [x["pred"] for x in r["samples"] if x["pred"]]
    hit = any(score(p, g) == 1 for p in preds)
    vote_ok = score(r["vote"], g) == 1 if r["vote"].get("type") != "abstain" else False
    tag = "wrong" if s == -1 else "abstain"
    if vote_ok:
        return f"{tag}:gate"  # the vote was right, the decision rule dropped or changed it
    if hit:
        return f"{tag}:vote"  # some sample was right, the vote picked another answer
    if not preds:
        return f"{tag}:no_exec"  # no sample produced an answer
    return f"{tag}:no_sample"  # samples answered, none right


def report(name, rs):
    ids = [i for i in rs if i in GOLD]
    sc = Counter(score(rs[i]["final"], GOLD[i]) for i in ids)
    n = len(ids)
    print(f"== {name}: N={n} PA={(sc[1] - sc[-1]) / n:.4f} correct={sc[1]} wrong={sc[-1]} abstain={sc[0]} coverage={(sc[1] + sc[-1]) / n:.3f}"
          f" precision={sc[1] / max(1, sc[1] + sc[-1]):.3f}")
    orc = sum(any(x["pred"] and score(x["pred"], GOLD[i]) == 1 for x in rs[i]["samples"]) for i in ids)
    orc8 = sum(any(x["pred"] and score(x["pred"], GOLD[i]) == 1 for x in rs[i]["samples"][:8]) for i in ids)
    print(f"Oracle@all={orc / n:.3f} ({orc})  Oracle@8={orc8 / n:.3f}  mean samples={sum(len(rs[i]['samples']) for i in ids) / n:.1f}"
          f"  latency mean={sum(rs[i]['elapsed_seconds'] for i in ids) / n:.1f}s")
    w = Counter(where(rs[i], GOLD[i]) for i in ids)
    print("causes:", ", ".join(f"{k}={v}" for k, v in sorted(w.items())))
    for key, f in (("gold type", lambda i: GOLD[i]["type"]), ("topic", lambda i: topic(rs[i]["question"])), ("shop", lambda i: rs[i]["business_id"])):
        tab = defaultdict(Counter)
        for i in ids:
            tab[f(i)][score(rs[i]["final"], GOLD[i])] += 1
        print(f"by {key}: " + "; ".join(f"{k} {c[1]}/{c[-1]}/{c[0]} ({(c[1] - c[-1]) / sum(c.values()):+.2f})" for k, c in sorted(tab.items(), key=lambda x: -sum(x[1].values()))))
    return ids


def main():
    runs = [load(r) for r in sys.argv[1:]]
    ids = [report(r, rs) for r, rs in zip(sys.argv[1:], runs)]
    if len(runs) == 2:
        a, b = runs
        both = [i for i in ids[0] if i in b]
        d = Counter((score(a[i]["final"], GOLD[i]), score(b[i]["final"], GOLD[i])) for i in both)
        gain = sum((y - x) * n for (x, y), n in d.items())
        print(f"paired on {len(both)}: PA {sum(x * n for (x, _), n in d.items()) / len(both):.4f} -> {sum(y * n for (_, y), n in d.items()) / len(both):.4f} "
              f"(net {gain:+d}); transitions (old,new): " + ", ".join(f"{k}={v}" for k, v in sorted(d.items()) if k[0] != k[1]))


if __name__ == "__main__":
    main()
