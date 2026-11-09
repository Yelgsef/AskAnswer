"""Flags at each checkpoint of a run: shop (views, units, status, duplicates), question (hints, values, agreement), sample (query, period, filters, source). With labels, each question gets the first
stage that broke it. Writes runs/debug/<name>.jsonl and prints matrices. Public runs get the same flags without labels, as rates only.
python analysis/flags.py NAME RUN VIEWS --data data/train [--labels]"""
import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from datamate import fewshot
from datamate import values, views
from datamate import sqlexec, voting  # noqa: E402
from datamate.hints import period_hints  # noqa: E402
from datamate.scoring import score
from datamate.ingest import load_business, norm

INTENTS = [("delivery", r"dot_hang|dot_giao|giao_hang|xe_cho|xe_da_giao|xe_giao"), ("debt", r"(^|_)no(_|$)|cong_no|mua_chiu|ghi_no"), ("lot/expiry", r"(^|_)lo(_|$)|het_han|han_dung"),
           ("stock", r"(^|_)ton(_|$)|ton_kho|het_hang|khong_ban_duoc"), ("price change", r"tang_gia|len_gia|gia_ban|doi_gia"), ("cost/profit", r"(^|_)lai(_|$)|loi_nhuan|gia_von|nhap_hang|tien_nhap|nha_cung_cap|chi_phi"),
           ("ranking", r"ban_chay|ban_nhieu|hot_nhat|(^|_)top(_|$)|nhieu_nhat|cao_nhat|lon_nhat|it_nhat|hang_nao|mat_hang_nao"), ("percent", r"phan_tram|ti_le|ty_le|ti_trong|ty_trong|tang_truong|tang_hay_giam|(^|_)%"),
           ("quantity", r"so_luong|bao_nhieu_(cai|ly|lon|vien|hop|chai|bao|goi|chiec|dia|phan|kg|thung)|(bn|bnhiu)_(cai|ly|lon|vien|hop|chai|bao)|so_ly|so_cai"),
           ("order count", r"so_don|bao_nhieu_don|(bn|bnhiu)_don|so_bill|so_hoa_don|may_don|bao_nhieu_bill"), ("customer", r"khach"), ("revenue", r"doanh_thu|dthu|(^|_)tien(_|$)|thu_ve|ban_duoc_bao_nhieu")]
intent = lambda q: next((n for n, p in INTENTS if re.search(p, norm(q))), "other")
DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
last = lambda s: s["text"].split("\n#REPAIR\n")[-1]


def shop_flags(db, meta, v, d, as_of):
    f = {"tables": len(meta["tables"]), "view_sql": bool(v.get("sql")), "view_offered": bool(v.get("sql")) and v.get("agree", 0) * 4 >= v.get("k", 8), "agree": v.get("agree"), "debt_layer": bool(d.get("used"))}
    if not v.get("sql"):
        return f
    spread, items, _ = views.unit_spread(db, v["sql"])
    f.update(unit_spread=spread, multi_unit_items=items)
    views.build(db, v["sql"])
    r = db.execute("""SELECT COUNT(*), AVG(CAST(customer_key IS NULL AS INT)), AVG(CAST(product_name IS NULL AS INT)), AVG(CAST(product_group IS NULL AS INT)), AVG(CAST(order_ts IS NULL AS INT)), AVG(CAST(qty IS NULL AS INT)),
        AVG(CAST(order_date > ? AS INT)), AVG(CAST(line_kind = 'service' AS INT)), AVG(CAST(line_kind = 'adjust' AS INT)) FROM v_sales_all""", [as_of]).fetchone()
    f.update(lines=r[0], null_customer=r[1], null_product=r[2], null_group=r[3], null_time=r[4], null_qty=r[5], lines_after_today=r[6], service_lines=r[7], adjust_lines=r[8])
    r = db.execute("SELECT COUNT(*), AVG(CAST(is_cancelled AS INT)), AVG(CAST(is_returned AS INT)) FROM v_orders_all").fetchone()
    f.update(orders=r[0], cancelled=r[1], returned=r[2])
    f["duplicate_lines"] = db.execute("SELECT 1 - COUNT(*) * 1.0 / NULLIF((SELECT COUNT(*) FROM v_sales_all), 0) FROM (SELECT DISTINCT * FROM v_sales_all)").fetchone()[0]
    return f


def sample_flags(s, question, as_of, hints, known, view_offered):
    t = last(s)
    if "<think>" in s["text"] and "</think>" not in s["text"]:
        return {"state": "thought cut off"}
    sql = sqlexec.parse(t)[1] or ""
    if not sql:
        return {"state": "no query"}
    tb = fewshot.tables_of(sql)
    out = {"state": "answer" if s["pred"] else "error", "repaired": "#REPAIR" in s["text"], "source": "views" if tb and tb <= fewshot.CANON else "raw" if tb and not tb & fewshot.CANON else "mixed", "macro": bool(sqlexec.DEBT_CALL.search(sql))}
    a = (s["pred"] or {}).get("answer")
    out["nil"] = bool(s["pred"]) and (a in (0, [], None) or (s["pred"]["type"] == "table" and not (a or {}).get("rows")))
    days = DATE.findall(sql)
    d = date.fromisoformat(as_of)
    out["date_after_today"] = any(date.fromisoformat(x) > d + timedelta(days=1) for x in days if _ok(x))
    want = [m for h in hints for m in re.findall(r"\[(\d{4}-\d\d-\d\d), (\d{4}-\d\d-\d\d)\)", h)[:1]]
    if want:
        out["hint_followed"] = all(a_ in sql and (b_ in sql or _prev(b_) in sql) for a_, b_ in want)
    miss = values.missing_filters(sql, known)
    out["missing_value"] = bool(miss)
    out["uses_qty"] = bool(re.search(r"(?i)\bqty\b", sql))
    return out


def _ok(x):
    try:
        date.fromisoformat(x)
        return True
    except ValueError:
        return False


def _prev(x):
    return str(date.fromisoformat(x) - timedelta(days=1))


def root(i, t, sf, shop, gold, res, voters, intent_):
    """The first stage that broke a question with a label."""
    g = gold[i]
    S = lambda p: score(p, g)
    if res == 1:
        return "OK"
    if t["final"].get("reason"):
        return "R0 rule says outside the data period"
    right = [s for s in t["samples"] if s["pred"] and S(s["pred"]) == 1]
    if right:
        a, c = voting.vote([s["pred"] for s in voters], len(voters))
        top = a.get("type") != "abstain" and S(a) == 1
        if top:
            return "D1 right answer dropped by a rule (0, empty, unanswerable)" if c >= .5 else "V1 right group leads under .5"
        return "V2 wrong group outvotes a right sample"
    if g["type"] == "unanswerable":
        return "G0 gold unanswerable, no sample says so"
    n = max(1, len(sf))
    share = lambda k, v=True: sum(x.get(k) == v for x in sf) / n
    if sum(x["state"] in ("no query", "thought cut off", "error") for x in sf) / n >= .5:
        return "G1 most samples give no answer"
    if share("hint_followed", False) >= .5:
        return "G2 period differs from the hint"
    if share("date_after_today") >= .5:
        return "G3 window runs past today"
    if share("missing_value") >= .5:
        return "G4 filter value not in the data"
    if intent_ == "debt" and shop.get("debt_layer") and share("macro") < .5:
        return "G5 debt macros not used"
    if (shop.get("unit_spread") or 0) > 2 and share("uses_qty") >= .5:
        return "G6 qty not in base units"
    if share("source", "raw") >= .5:
        return "G7 raw tables, logic wrong"
    return "G8 views, logic wrong"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("run")
    p.add_argument("views")
    p.add_argument("--data", default="data/train")
    p.add_argument("--labels", action="store_true")
    p.add_argument("--canon", action="store_true", help="one spelling per name when loading the shops")
    p.add_argument("--start", type=int, default=8)
    a = p.parse_args()
    data, layers = R / a.data, R / a.views / "logs"
    Q = {q["id"]: q for q in map(json.loads, (x for x in open(data / "questions.jsonl", encoding="utf-8-sig") if x.strip()))}
    tr = defaultdict(list)
    for line in open(R / a.run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line)
        tr[t["business_id"]].append(t)
    gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))} if a.labels else {}
    (R / "runs/debug").mkdir(exist_ok=True)
    out = open(R / "runs/debug" / f"{a.name}.jsonl", "w", encoding="utf-8")
    shops, rows = {}, []
    for b, ts in sorted(tr.items()):
        db, meta = load_business(data, b, canon=a.canon)
        vf, df = layers / f"_views_{b}.json", layers / f"_debt_{b}.json"
        v = json.loads(vf.read_text(encoding="utf-8")) if vf.exists() else {}
        d = json.loads(df.read_text(encoding="utf-8")) if df.exists() else {}
        as_of = max(Q[t["id"]]["as_of_date"] for t in ts)
        shops[b] = shop_flags(db, meta, v, d, as_of)
        known = values.known_values(db)
        for t in ts:
            i = t["id"]
            q = Q[i]
            hints = period_hints(t["question"], q["as_of_date"], db)
            sf = [sample_flags(s, t["question"], q["as_of_date"], hints, known, shops[b]["view_offered"]) for s in t["samples"]]
            idx = voting.debt_pool([s["text"] for s in t["samples"]], a.start)
            voters = [t["samples"][j] for j in idx]
            ans, conf = voting.vote([s["pred"] for s in voters], len(voters))
            final = t["final"] if t["final"].get("reason") else ans if voting.decide(ans, conf, .5) else {"type": "abstain"}
            r = {"id": i, "shop": b, "intent": intent(t["question"]), "hints": len(hints), "share": conf, "groups": len({voting.answer_key(s["pred"]) for s in voters if s["pred"]}), "final_type": final["type"], "top_type": ans.get("type"), "samples": sf}
            if a.labels:
                res = score(final, gold[i])
                r["result"] = res
                r["root"] = root(i, t, sf, shops[b], gold, res, voters, r["intent"])
            rows.append(r)
            out.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        db.close()
    out.close()
    (R / "runs/debug" / f"{a.name}_shops.json").write_text(json.dumps(shops, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pc = lambda x: "  -  " if x is None else f"{x:5.1%}"
    print(f"## {a.name}: shop flags")
    print("  shop      views agree spread items | no cust  no grp no time no qty after-today | cancel return dup lines | service adjust | debt")
    for b, f in sorted(shops.items()):
        sp = f.get("unit_spread")
        print(f"  {b:8s} {'yes' if f['view_offered'] else 'no ':3s} {str(f.get('agree')):>5s} {('%.1f' % sp) if sp else '  -':>6s} {str(f.get('multi_unit_items', '-')):>5s} | {pc(f.get('null_customer'))} {pc(f.get('null_group'))} {pc(f.get('null_time'))} {pc(f.get('null_qty'))} {pc(f.get('lines_after_today'))} | {pc(f.get('cancelled'))} {pc(f.get('returned'))} {pc(f.get('duplicate_lines'))} | {pc(f.get('service_lines'))} {pc(f.get('adjust_lines'))} | {'yes' if f['debt_layer'] else 'no'}")
    ss = [x for r in rows for x in r["samples"]]
    n = max(1, len(ss))
    rate = lambda fn: sum(map(fn, ss)) / n
    print(f"## {a.name}: sample flags over {len(ss)} samples, {len(rows)} questions")
    print(f"  state {Counter(x['state'] for x in ss).most_common()}")
    print(f"  repaired {rate(lambda x: bool(x.get('repaired'))):.1%} | 0 or empty {rate(lambda x: bool(x.get('nil'))):.1%} | filter value not in data {rate(lambda x: bool(x.get('missing_value'))):.1%} | date past today {rate(lambda x: bool(x.get('date_after_today'))):.1%}")
    hs = [x for x in ss if "hint_followed" in x]
    print(f"  samples on questions with a period hint {len(hs)}, follow it {sum(x['hint_followed'] for x in hs) / max(1, len(hs)):.1%} | sources {Counter(x.get('source') for x in ss if x.get('source')).most_common()}")
    print(f"  questions: with a hint {sum(r['hints'] > 0 for r in rows) / max(1, len(rows)):.1%} | agreement mean {st.mean(r['share'] for r in rows):.3f} | one answer group {sum(r['groups'] == 1 for r in rows) / max(1, len(rows)):.1%} | abstain {sum(r['final_type'] == 'abstain' for r in rows) / max(1, len(rows)):.1%}")
    if a.labels:
        print(f"## {a.name}: first stage that broke each question")
        c = Counter(r["root"] for r in rows)
        res = defaultdict(Counter)
        for r in rows:
            res[r["root"]][r["result"]] += 1
        for k, v in sorted(c.items(), key=lambda kv: (kv[0] == "OK", -kv[1])):
            print(f"  {v:4d} ({v / len(rows):5.1%}) wrong {res[k][-1]:3d} abstain {res[k][0]:3d}  {k}")
        print(f"## {a.name}: stage x intent (not right)")
        ints = [k for k, _ in Counter(r["intent"] for r in rows if r["root"] != "OK").most_common()]
        print("  " + " " * 42 + " ".join(f"{k[:8]:>8s}" for k in ints))
        m = Counter((r["root"], r["intent"]) for r in rows)
        for k, _ in c.most_common():
            if k != "OK":
                print(f"  {k[:42]:42s}" + " ".join(f"{m[k, i] or '.':>8}" for i in ints))
        print(f"## {a.name}: stage x shop (not right)")
        sh = sorted(shops)
        print("  " + " " * 42 + " ".join(f"{k[-7:]:>8s}" for k in sh))
        m = Counter((r["root"], r["shop"]) for r in rows)
        for k, _ in c.most_common():
            if k != "OK":
                print(f"  {k[:42]:42s}" + " ".join(f"{m[k, i] or '.':>8}" for i in sh))


if __name__ == "__main__":
    main()
