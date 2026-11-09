"""Executing one sample: parse its TYPE line and SQL, run the query read-only with a timeout, turn the result into a typed
answer, and reject results the rulebook makes implausible (the error goes back to the model in a repair turn)."""
import math
import re
import threading
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP

from .hints import DEBT_WORDS
from .ingest import norm

DEBT_CALL = re.compile(r"debt_(at|new|repaid)\s*\(")
TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")


def parse(text):
    if "<think>" in text:
        if "</think>" not in text:
            return None, None
        text = text.split("</think>")[-1]
    t = re.search(r"TYPE:\s*([a-z]+)", text)
    s = re.findall(r"```(?:sql)?\s*(.*?)```", text, re.S)
    return (t.group(1) if t else None), (s[-1].strip() if s else None)


def cell(v):
    if isinstance(v, (list, tuple)):
        return [cell(x) for x in v]
    if isinstance(v, dict):
        return {str(k): cell(x) for k, x in v.items()}
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite number")
        return int(v) if v.is_integer() else round(v, 4)
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10]
    return v


def run_sql(db, sql, timeout=20):
    timer = threading.Timer(timeout, db.interrupt)
    timer.start()
    try:
        cur = db.execute(sql)
        names = [c[0] for c in cur.description]
        rows = cur.fetchmany(2001)
    finally:
        timer.cancel()
    if len(rows) > 2000:
        raise ValueError("result too large")
    return names, [[cell(v) for v in r] for r in rows]


def to_answer(kind, names, rows):
    if kind == "table":
        return {"columns": names, "rows": rows}
    if kind == "list":
        if len(names) == 1:
            out = [r[0] for r in rows]
        elif len(rows) == 1:
            out = rows[0]
        else:
            raise ValueError(f"list needs one column, got {names}")
        if any(x is None for x in out):  # a filter matched nothing. The shape message made the repair reshape it (holdout: 50 -> 17 samples)
            raise ValueError("a list item is NULL (check filters/values)")
        if any(isinstance(x, (bool, list, dict)) for x in out):  # the format check rejects such a list
            raise ValueError("every list item must be one name or number (one row per item, one column)")
        return out
    if len(rows) != 1 or len(names) != 1:
        raise ValueError(f"expected one cell, got {len(rows)} rows x {names}")
    v = rows[0][0]
    if v is None:
        raise ValueError("result is NULL (check filters/values)")
    if kind in ("number", "percent"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("expected a number")
        # rulebook: money (incl. money averages) rounds to whole đồng, half up. Other averages to 2 decimals.
        # shortcut: |v| >= 1000 taken as VND money. A non-money average that large would be wrongly rounded.
        v = float(Decimal(str(v)).quantize(Decimal("1" if kind == "number" and abs(v) >= 1000 else ".01"), rounding=ROUND_HALF_UP))
        return int(v) if v.is_integer() else v
    if kind == "boolean":
        if not isinstance(v, bool):
            raise ValueError("expected boolean")
        return v
    v = str(v)
    if kind == "date" and not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v):
        raise ValueError("date must be YYYY-MM-DD or YYYY-MM")
    return v


CHANGE = re.compile(r"(?:^|_)(?:tang|giam(?!_(?:dau|gia|can|beo|mo|sot))|thay_doi|hon_hay_kem|cao_hay_thap|len_xuong|tang_truong|chenh_lech)(?:_|$)")  # giảm đau / giảm giá name an item or a discount, not a change


GROSS = re.compile(r"(?:^|_)(?:lai_gop|loi_nhuan_gop|lai_go)(?:_|$)")


def has_debt_layer(db):
    try:
        return bool(db.execute("SELECT count(*) FROM duckdb_functions() WHERE function_name = 'debt_at'").fetchone()[0])
    except Exception:
        return False


def execute(db, text, question="", trace=None):
    """trace (optional dict) receives the SQL as it is run against the database (after the rewrites below), for the run logs."""
    kind, sql = parse(text)
    if kind not in TYPES:
        raise ValueError("missing or invalid TYPE line")
    if kind == "unanswerable":
        return {"type": kind}
    if not sql or not re.match(r"(?is)\s*(with|select|from)\b", sql) or re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma)\b", sql):
        raise ValueError("need one read-only SELECT query in ```sql```")
    sql = re.sub(r'"(%[^"]*)"', r"'\1'", sql)  # a double-quoted strftime format is a string, never a column
    if kind == "percent" and CHANGE.search(norm(question)) and "-" not in re.sub(r"'[^']*'", "", sql):
        # 14B often writes 100*new/old after planning 100*(new-old)/old. Lint it into the repair turn
        raise ValueError("a change in % is ROUND(100.0 * (new - old) / old, 2), negative when it fell; this query has no subtraction")
    if DEBT_WORDS.search(norm(question)) and not DEBT_CALL.search(sql) and has_debt_layer(db):
        # train, shops with a debt layer: macro samples 740/816 right, raw-table samples 21/504, no question only a raw sample solved
        raise ValueError("this shop has the debt layer: answer a debt question with its macros, debt_at(DATE 'X') for what is owed at the end of a day, "
                         "debt_new(DATE 'A', DATE 'B') for new debt and debt_repaid(DATE 'A', DATE 'B') for repayments in a period (see DEBTS), not from the raw tables")
    if trace is not None:
        trace["sql"] = sql.rstrip(";")
    names, rows = run_sql(db, sql.rstrip(";"))
    ans = to_answer(kind, names, rows)
    if kind == "string" and (m := re.fullmatch(r"(?i)\s*th[aá]ng\s*(\d{1,2})\s*[/-]\s*(\d{4})\s*", ans)) and 1 <= int(m[1]) <= 12:
        ans = f"{m[2]}-{int(m[1]):02d}"  # 'tháng 3/2025' and '2025-03' split the vote on public 'tháng nào' questions. Gold is YYYY-MM
    if kind == "string" and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", ans):  # a day/month answer is type date
        kind = "date"
    if kind == "number" and isinstance(ans, (int, float)) and ans < 0 and GROSS.search(norm(question)):  # train: 0 of 7 negative gross-profit samples right
        raise ValueError(f"gross profit {ans} < 0 is implausible: the cost written on a sale line is for ONE SALE UNIT (the unit of the unit price), so multiply it "
                         "by the quantity in that sale unit, not by the base-unit quantity, and do not subtract purchases or expenses")
    if kind == "percent" and isinstance(ans, (int, float)) and abs(ans) > 1000:  # train: 0 of 10 such samples right, gold max 391%
        raise ValueError(f"{ans}% is implausible: the base of the ratio is (almost) empty. If the period it compares with lies outside the shop's data dates, "
                         "the question is unanswerable (TYPE: unanswerable); otherwise check its filters")
    return {"type": kind, "answer": ans}
