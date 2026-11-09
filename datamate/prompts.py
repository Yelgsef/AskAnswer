"""The prompts the SQL samples see: the system prompt (two wordings), the user message, chat-word glosses and repair hints."""
import re
from decimal import Decimal

from .hints import date_anchors

SYSTEM = """You are an expert data analyst answering Vietnamese questions about one shop's data with ONE DuckDB SQL query.
The RULEBOOK below defines every business term; follow it exactly (order status, revenue, periods, units, ranking, ties, answer format).
When the rulebook defines a term by a computation (e.g. a shift by the order's time, a main supplier by number of receipts), compute it from the transaction rows that way, even if some table has a column or report with the same name.

How to work:
1. Read the question. In at most 6 short lines, note which rulebook rules apply, which tables/columns hold the data, the exact date window, grain (rows vs distinct orders), status filter and units.
2. Then output exactly:
TYPE: <number|percent|string|date|boolean|list|table|unanswerable>
```sql
<one read-only SELECT/WITH query>
```

SQL rules:
- Use only tables/columns listed in SCHEMA, always double-quoted: "Thành tiền". String literals in single quotes.
- Sales tables are often one row per order line: count orders with COUNT(DISTINCT <order id>); count days, customers, products at their own grain with COUNT(DISTINCT ...). Any event (delivery trip, visit, receipt) is counted the same way: COUNT(DISTINCT <the columns that identify one event>), e.g. COUNT(DISTINCT (order id, trip no)) when the rulebook counts each trip of each order. Quantities: SUM the quantity column, never count rows.
- Order status has four meanings: completed, cancelled (any hủy/huỷ wording, and undeliverable 'không giao được'), returned (trả hàng, đổi trả, hoàn trả, hàng trả về, even partial), in transit (đang giao = a normal order). Spellings vary: classify with replace(strip_accents(lower(col)), 'đ', 'd') LIKE patterns ('%huy%', '%khong giao duoc%', '%tra%'), checking the listed VALUES. A shop without a status column writes it in a note column (Ghi chú); a note not mentioning status = completed.
- Exclude a WHOLE order when it is cancelled or has any returned line if the rule says so (order id NOT IN (...)). Revenue = SUM of line amounts (after line discounts), not the order total after order-level discount.
- Names (products, customers, groups) may differ in case/spaces: compare and group with lower(trim(regexp_replace(x, '\\s+', ' ', 'g'))). Walk-in customers (khách lẻ, kl, lẻ, vãng lai, blank) are not customers: exclude them from customer counts and rankings.
- Use the DATE ANCHORS given; never CURRENT_DATE. Dates: col >= DATE 'a' AND col < DATE 'b' (b exclusive). hôm qua is ONE day. Quarter Q = months 3Q-2..3Q. Filter by the date of the event asked (order date for sales, payment date for payments, import date for imports). Slang: hnay = hôm nay, hqua/bữa qua = hôm qua, dthu = doanh thu, bnhiu/bao nhiu = bao nhiêu.
- Units come from unit columns and conversion tables only, never from numbers inside product names.
- "trả bằng / trả qua / trả tiền mặt / thanh toán bằng" names how an order was PAID (payment method), not a return.
- Status flags (BOOLEAN, when present): "_huy" / "_tra" = this row's status or note is cancelled / returned; "_don_huy" / "_don_tra" = some line of the same order is. Filter status with these flags (e.g. NOT "_don_huy" AND NOT "_don_tra" to drop whole cancelled/returned orders), not by matching status text.
- CANONICAL VIEWS (when listed) apply the rulebook's order rules once: "v_sales" = sold item/service lines of VALID orders only (not cancelled, no returned line) and "v_orders" = VALID orders only, so revenue = SUM(amount) FROM v_sales and order counts = COUNT(*) FROM v_orders need no status filter. Use "v_sales_all" / "v_orders_all" (every line/order, with is_cancelled, is_returned, line_kind) only for questions about cancelled/returned orders, vouchers, shipping or all orders regardless of status. Their columns are listed with them; use the raw tables for anything they lack (imports, payments, debts, stock, units, extra columns).
- number/percent/string/date/boolean: return exactly one row with one column. list: one column, rows in the required order (several numbers asked -> one row is fine). table: first column = object name, then measures in the order asked.
- Answer only what is asked. "top N / N <things> nào / là gì / là ai" asks for names: a list of the names (string when one), no measure columns. Use table only when the question also asks a value for each one (kèm số tiền, mỗi ... bao nhiêu, và số nợ của từng người).
- percent = ROUND(100.0 * part / total, 2). Change "tăng/giảm bao nhiêu %" = ROUND(100.0 * (new - old) / old, 2), negative when it fell. Money = ROUND(x) to whole VND. Other averages ROUND(x, 2).
- A day or month answer is TYPE date (never string): strftime(d, '%Y-%m-%d'), or '%Y-%m' when the question asks which month. boolean: a SQL boolean.
- Names: return the name exactly as stored in the data.
- unanswerable is rare: use it only if NO table/column holds the needed information, or the asked period lies completely outside the date range shown in SCHEMA. A period inside the data range with no matching rows is an answer (0 or empty), not unanswerable. Then write TYPE: unanswerable and SELECT 1."""

# Second wording of the status/revenue/name/date rules. Half the samples use each prompt: voting over both
# beats either alone (dev300 PA@0.5: 0.417 mixed vs 0.34-0.35 single, multi-path self-consistency).
SYSTEM_ALT = re.sub(r"- Order status has.*?(?=\n- Units)", lambda _: """\
- Status/category values vary in spelling and case ('Hủy', 'Đã hủy đơn', 'huỷ'): match every listed variant (e.g. lower(col) IN (...) or strip_accents(lower(col)) LIKE '%huy%'), never just one literal. Names: compare case-insensitively.
- Orders in transit (đang giao) are normal orders. Exclude a WHOLE order when it is cancelled or has any returned line if the rule says so (order id NOT IN (...)).
- Use the DATE ANCHORS given; never CURRENT_DATE. Dates: col >= DATE 'a' AND col < DATE 'b' (b exclusive). Slang: hnay = hôm nay, hqua/bữa qua = hôm qua, dthu = doanh thu, bnhiu/bao nhiu = bao nhiêu.""", SYSTEM, flags=re.S)
SYSTEM_ALT = re.sub(r"- CANONICAL VIEWS.*?\n", "- Compute from the raw shop tables yourself; do not use the v_sales / v_orders views (an independent check of them).\n", SYSTEM_ALT, flags=re.S)
assert SYSTEM_ALT != SYSTEM and "v_orders views (an independent" in SYSTEM_ALT and "CANONICAL VIEWS (when" in SYSTEM

NOTES = "Notes:\n-"  # the start of a plain sample with --notes

# Vietnamese chat abbreviations (language-level, not shop-specific), glossed next to the question when present
SLANG = {"hqua": "hôm qua", "hnay": "hôm nay", "hnao": "hôm nào", "bqua": "bữa qua", "dthu": "doanh thu",
         "bnhiu": "bao nhiêu", "bn": "bao nhiêu", "bnh": "bao nhiêu", "sl": "số lượng", "sp": "sản phẩm", "kh": "khách hàng",
         "ko": "không", "k": "không", "dc": "được", "đc": "được", "tg": "thời gian", "nv": "nhân viên", "ncc": "nhà cung cấp",
         "th": "tháng", "thg": "tháng", "z": "vậy", "mik": "mình", "mk": "mình", "j": "gì",
         "gồi": "rồi", "rùi": "rồi", "hok": "không", "hem": "không", "nhìu": "nhiều", "bít": "biết", "lun": "luôn", "jup": "giúp", "dzậy": "vậy", "zậy": "vậy",
         "ni": "này", "mô": "nào", "wan": "quán"}  # central / chat forms seen in public questions, absent from train
MONEY = re.compile(r"\b(\d+(?:[.,]\d)?)\s*(tr|triệu|trieu|củ|k)(\d?)\b", re.I)


def money_gloss(question):
    """'40tr' = 40000000, '3tr7' = 3700000, '11 củ' = 11000000, '190k' = 190000 (chat shorthand for amounts)."""
    out = []
    for m in MONEY.finditer(question):
        n, u, frac = m.groups()
        mult = 1000 if u.lower() == "k" else 1000000
        v = (Decimal(n.replace(",", ".")) + Decimal(f"0.{frac or 0}")) * mult
        out.append(f"{m.group(0)} = {int(v)} đồng")
    return out


def chat(system, biz_ctx, q, shots=""):
    hits = q["_hits"]
    words = re.findall(r"\w+", q["question"].lower())
    gloss = [f"{w} = {SLANG[w]}" for w in dict.fromkeys(words) if w in SLANG] + money_gloss(q["question"])
    user = (biz_ctx + "\n\nDATE ANCHORS:\n" + date_anchors(q["as_of_date"])
            + "".join("\n" + p for p in q.get("_periods", []))
            + ("\n\nVALUES MATCHING THE QUESTION:\n" + "\n".join(hits) if hits else "")
            + ("\n\n" + shots if shots else "")
            + f"\n\nQUESTION: {q['question']}" + (f"\n(abbreviations: {'; '.join(gloss)})" if gloss else ""))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair_hint(err):
    """One generic line for the commonest failure kinds (public trace: 1,933 of 9,912 samples still failed after one repair,
    mostly an empty result or a result of the wrong shape)."""
    if err.startswith("expected one cell, got 1 rows"):
        return "Return only the one value asked (a single column, e.g. just the name), or use TYPE list/table if the question asks for several values.\n"
    if err.startswith(("expected one cell, got 0 rows", "result is NULL", "a list item is NULL")):
        return ("No row matched the filters: check every filter value against the SCHEMA values and VALUES MATCHING THE QUESTION (spelling, "
                "or the value sits in another column or table, e.g. an address in the customer list), and the date window against DATE ANCHORS.\n")
    if re.match(r"expected one cell, got \d+ rows x \['count\(", err, re.I):
        return "This COUNT has a GROUP BY, so it returns one count per group: to count the groups (customers, products, days) use COUNT(DISTINCT ...) without GROUP BY, or SELECT COUNT(*) FROM (the grouped query).\n"
    if "type TIMESTAMP" in err and "type TIME " in err + " ":
        return "To compare a time of day, cast the timestamp: CAST(col AS TIME) >= TIME '12:00', or use EXTRACT(hour FROM col).\n"
    if err.startswith("expected one cell, got"):
        return "The question asks for one value: aggregate, or ORDER BY ... LIMIT 1 for a top-1 question; use TYPE list/table if it asks for several.\n"
    if "must appear in the GROUP BY clause" in err:
        return "A column outside every aggregate must be in GROUP BY: put the whole per-row expression inside the aggregate, e.g. SUM(amount * \"rate %\" / 100), or add that column to GROUP BY.\n"
    if "not found in FROM clause" in err:
        return "Use only the columns SCHEMA lists for each table, double-quoted; a column of another table needs a JOIN on its key.\n"
    return ""
