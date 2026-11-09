"""Error matrix over labeled runs (train questions only): python analysis/error_matrix.py RUN [RUN ...] [--name NAME] [--plain-k 8] [--dump 80]
Per sample and per question, against gold: question kind (keyword rules) x error class, gold type x answer type, route (macro / view / raw) x kind,
per-shop accuracy, and the questions no sample solved split into systematic (one wrong answer on 75%+ of the answered samples: a view, rule or gold
problem more than a reasoning one) and scattered. Writes the per-question classes to runs/debug/errors_NAME.jsonl and the report to stdout."""
import argparse, json, math, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from datamate.scoring import norm, score

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", type=Path); ap.add_argument("--name", default="runs"); ap.add_argument("--plain-k", type=int, default=8)
ap.add_argument("--dump", type=int, default=80)
a = ap.parse_args()
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
asof = {x["id"]: x.get("as_of_date") for x in map(json.loads, open(R / "data/train/questions.jsonl"))}

def plain(s):
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d").replace("đ", "d")
    return " ".join("".join(c for c in s if unicodedata.category(c) != "Mn").split())

KINDS = [  # first match wins
    ("debt", r"\b(cong no|ghi no|mua chiu|tra no|con no|du no|dang no|no cu|no moi|thu no|khach no|no bao nhieu|no tien|no nhieu|tien no|so no|chua tra)\b"),
    ("purchase", r"\b(nhap hang|phieu nhap|nha cung cap|ncc|nhap kho|nhap ve|lan nhap|don nhap|gia nhap|nhap)\b"),
    ("inventory", r"\b(ton kho|hang ton|ton cuoi|ton dau|con ton|ton)\b"),
    ("cancel/return", r"\b(huy|hoan|tra hang|tra lai|bi tra|doi tra)\b"),
    ("ratio/percent", r"(\bti le\b|\bty le\b|phan tram|%|\bchiem\b)"),
    ("average", r"\b(trung binh|binh quan)\b"),
    ("compare/growth", r"\b(so voi|tang truong|tang bao nhieu|giam bao nhieu|chenh lech|tang hay giam|cao hon|thap hon)\b"),
    ("top customer", r"\bkhach\b.*\b(nhieu nhat|it nhat|top|lon nhat|cao nhat|nhat)\b|\b(top|nhieu nhat|lon nhat)\b.*\bkhach\b"),
    ("top item", r"\b(ban chay|hot|top|nhieu nhat|it nhat|cham nhat|e nhat|lon nhat|cao nhat|thap nhat|nhat)\b"),
    ("customer", r"\b(khach|khach hang|khach quen|khach moi)\b"),
    ("staff/channel/pay", r"\b(nhan vien|thu ngan|kenh|shopee|grab|lazada|tiktok|online|thanh toan|chuyen khoan|tien mat|the|vi)\b"),
    ("order count", r"\b(so don|bao nhieu don|may don|dem|hoa don|don hang|so hoa don|luot)\b"),
    ("quantity", r"\b(so luong|bao nhieu (cai|ly|hop|chai|kg|goi|lon|thung|cay|bao|cuon|vien|chiec|phan|suat|to|dia))\b"),
    ("revenue/amount", r"\b(doanh thu|doanh so|tien|gia tri|thu duoc|ban duoc|tong)\b"),
    ("time pattern", r"\b(gio nao|ngay nao|thu may|khung gio|ca sang|ca chieu|ca toi|buoi)\b"),
]
KINDS = [(k, re.compile(p)) for k, p in KINDS]
kind_of = lambda q: next((k for k, p in KINDS if p.search(plain(q))), "other")
VIEWS, DEBT = re.compile(r"(?i)\bv_(sales|orders)(_all)?\b"), re.compile(r"debt_(at|new|repaid)\s*\(")
route = lambda text: (lambda sql: "macro" if DEBT.search(sql) else "view" if VIEWS.search(sql) else "raw")(text.split("```sql")[-1])
num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def err_class(p, g):
    """Why answer p misses gold g."""
    if p is None:
        return "no answer (SQL failed)"
    pt, gt = p.get("type"), g["type"]
    if gt == "unanswerable":
        return "answered an unanswerable"
    if pt in ("unanswerable", "abstain"):
        return "said unanswerable"
    if pt != gt:
        if {pt, gt} == {"number", "percent"}:
            return "number/percent type"
        return f"type {gt}->{pt}"
    pv, gv = p.get("answer"), g["answer"]
    if gt in ("number", "percent"):
        if not (num(pv) and num(gv)):
            return "number: not a number"
        if gv == 0:
            return "number: gold 0"
        if pv == 0:
            return "number: answered 0"
        r = pv / gv
        if r < 0:
            return "number: sign"
        if abs(r - 1) < 0.001:
            return "number: rounding (<0.1%)"
        if abs(r - 1) < 0.05:
            return "number: near (<5%)"
        for k in (100, 1000, 2, 3, 4, 5, 6, 10, 12, 24, 7, 8, 20, 30, 50):
            if abs(r - k) / k < 0.01:
                return f"number: x{k}" if k not in (100, 1000) else f"number: x{k} (scale)"
            if abs(1 / r - k) / k < 0.01:
                return f"number: /{k}" if k not in (100, 1000) else f"number: /{k} (scale)"
        return "number: far, larger" if r > 1 else "number: far, smaller"
    if gt == "list":
        if not isinstance(pv, list):
            return "list: not a list"
        P = [norm(x) for x in pv]; G = [norm(x) for x in gv]
        if sorted(P) == sorted(G):
            return "list: order"
        if len(P) != len(G):
            return "list: length"
        ov = len(set(P) & set(G)) / max(1, len(G))
        return "list: one item off" if ov >= (len(G) - 1) / len(G) else "list: partial" if ov >= 0.5 else "list: mostly wrong"
    if gt == "string":
        sp = plain(str(pv)); sg = plain(str(gv))
        return "string: spelling/accent" if re.sub(r"\W", "", sp) == re.sub(r"\W", "", sg) else "string: other entity"
    if gt == "date":
        return "date: wrong day"
    if gt == "table":
        return "table: mismatch"
    return f"{gt}: wrong"

key = lambda p: json.dumps(p, ensure_ascii=False, sort_keys=True) if p else None
rows = []
for run in a.runs:
    pk = a.plain_k
    for line in open(R / run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line); g = gold[t["id"]]
        ok = lambda p: bool(p) and score(p, g) == 1
        S = []
        for j, s in enumerate(t["samples"]):
            p = s.get("pred")
            S.append({"think": j >= pk, "right": ok(p), "cls": "right" if ok(p) else err_class(p, g), "route": route(s["text"]), "key": key(p),
                      "repair": "\n#REPAIR\n" in s["text"], "p": p})
        wrong = Counter(x["key"] for x in S if not x["right"] and x["key"])
        nans = sum(1 for x in S if x["key"])
        top, topn = (wrong.most_common(1)[0] if wrong else (None, 0))
        fin = t["final"]; fin_ok = fin.get("type") != "abstain" and ok(fin)
        nr = sum(x["right"] for x in S)
        if fin_ok:
            outcome = "final right"
        elif nr:
            outcome = "solved by a sample, vote missed" if fin.get("type") != "abstain" else "solved by a sample, abstained"
        elif not nans:
            outcome = "no sample answered"
        elif topn >= 0.75 * len(S) or (nans >= 2 and topn >= 0.75 * nans):
            outcome = "systematic wrong"
        else:
            outcome = "scattered wrong"
        modal = next((x for x in S if x["key"] == top), None)
        rows.append({"id": t["id"], "shop": t["business_id"], "q": t["question"], "kind": kind_of(t["question"]), "gtype": g["type"], "outcome": outcome,
                     "final_type": fin.get("type"), "final_ok": fin_ok, "n": len(S), "n_right": nr, "samples": S, "run": str(run),
                     "modal_cls": modal["cls"] if modal else None, "modal": modal["p"] if modal else None, "gold": g, "asof": asof.get(t["id"])})

def table(title, rk, ck, items, pct_of=None, min_row=1):
    cnt = defaultdict(Counter)
    for it in items:
        cnt[rk(it)][ck(it)] += 1
    cols = Counter(); [cols.update(c) for c in cnt.values()]
    cols = [c for c, _ in cols.most_common()]
    print(f"\n## {title}")
    w = max([len(str(r)) for r in cnt] + [8])
    print(" " * w, "  total", " ".join(f"{c[:22]:>22}" for c in cols))
    for r in sorted(cnt, key=lambda r: -sum(cnt[r].values())):
        tot = sum(cnt[r].values())
        if tot < min_row:
            continue
        print(f"{str(r):<{w}} {tot:>7}", " ".join(f"{(str(cnt[r][c]) + (f' {100*cnt[r][c]/tot:.0f}%' if cnt[r][c] else '')) if cnt[r][c] else '.':>22}" for c in cols))

print(f"# Error matrix: {len(rows)} questions from {', '.join(map(str, a.runs))}")
fr = sum(r["final_ok"] for r in rows); ab = sum(r["final_type"] == "abstain" for r in rows)
print(f"final right {fr}, abstain {ab}, wrong {len(rows)-fr-ab}; PA {(fr-(len(rows)-fr-ab))/len(rows):.3f}; any sample right {sum(r['n_right']>0 for r in rows)}")
table("question outcome by kind", lambda r: r["kind"], lambda r: r["outcome"], rows)
table("question outcome by gold type", lambda r: r["gtype"], lambda r: r["outcome"], rows)
table("final answer: gold type x answer type", lambda r: r["gtype"], lambda r: r["final_type"] if not r["final_ok"] else "right", rows)
samples = [dict(s, kind=r["kind"], shop=r["shop"], gtype=r["gtype"]) for r in rows for s in r["samples"]]
table("samples: kind x error class", lambda s: s["kind"], lambda s: s["cls"], samples)
table("samples: route x result", lambda s: f"{s['route']}{' think' if s['think'] else ''}", lambda s: "right" if s["right"] else "wrong" if s["key"] else "no answer", samples)
table("samples: kind x route (right share in the cell)", lambda s: s["kind"], lambda s: f"{s['route']} {'ok' if s['right'] else 'x'}", samples)
table("questions by shop", lambda r: r["shop"], lambda r: r["outcome"], rows)
bad = [r for r in rows if r["n_right"] == 0]
table("unsolved questions: modal error class by kind", lambda r: r["kind"], lambda r: r["modal_cls"] or "no answer", bad)
table("unsolved questions: systematic vs scattered by shop", lambda r: r["shop"], lambda r: r["outcome"], bad)
print(f"\n## systematic wrong ({sum(r['outcome']=='systematic wrong' for r in rows)}), first {a.dump}: shop | kind | question | gold | modal answer | class")
for r in sorted([r for r in rows if r["outcome"] == "systematic wrong"], key=lambda r: (r["shop"], r["kind"]))[:a.dump]:
    g = r["gold"]
    print(f"{r['shop']} | {r['kind']} | {r['q'][:90]} | {g['type']}:{json.dumps(g['answer'], ensure_ascii=False)[:80]} | {json.dumps((r['modal'] or {}).get('answer'), ensure_ascii=False)[:80]} | {r['modal_cls']}")
out = R / "runs/debug" / f"errors_{a.name}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({k: v for k, v in r.items() if k != "samples"} | {"classes": [s["cls"] for s in r["samples"]], "routes": [s["route"] for s in r["samples"]]}, ensure_ascii=False) + "\n")
print(f"\nper-question rows -> {out}")
