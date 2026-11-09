"""Per-shop semantic layer written by the LLM (no hand-written role/keyword rules).

The model reads one shop's README + SCHEMA + RULEBOOK and writes ONE SELECT that maps the raw tables to fixed
line-level columns. Generic SQL below casts them, lifts cancel/return flags to the whole order (rulebook: an order
with any cancelled/returned line is excluded as a whole) and derives v_sales / v_orders (valid only) plus *_all
variants. All samples are executed. The most common result fingerprint wins (execution-based self-consistency).
"""
import re
import threading
from collections import Counter

VIEW_SYSTEM = """You map ONE shop's raw tables (DuckDB) to a canonical sales table. Read README, SCHEMA and RULEBOOK carefully: they say which table holds the sales, which column is the order id, the date, the quantity, the line amount, the status.
Write ONE read-only DuckDB SELECT (WITH allowed) returning one row per sold line (one row per order or ticket if the shop records no lines), with exactly these columns:
  order_id      the order / invoice / ticket id (all lines of one order share it; one id = one order in the whole data: when the id is a running number that restarts per day/sheet/file, combine it with the date, e.g. strftime(date, '%Y%m%d') || '-' || id)
  order_date    DATE the order was placed (rulebook: the order date decides the period)
  order_ts      TIMESTAMP of the order, NULL if no time of day is recorded
  product_code  item code of the line, NULL if none
  product_group product group / category of the item (nhóm hàng, loại, danh mục) from the line or the catalog, NULL if none
  product_name  item / service name of the line (full name from a product catalog when the line only has a code or abbreviation), NULL if none
  qty           quantity sold on the line in the item's BASE UNIT (đơn vị cơ bản): a column already holding the converted quantity if the line table has one, otherwise the recorded quantity times the factor for the line's unit from the conversion table, catalog or README; an item with one unit keeps the recorded quantity
  unit          base unit of qty, NULL if none
  amount        line revenue in VND after line-level discounts, before order-level vouchers, without VAT/shipping
  customer_id   customer code / member card id when the shop records one (mã khách, mã thẻ ...), NULL otherwise and for walk-ins
  customer_name customer name; NULL for walk-in customers (khách lẻ, kl, lẻ, vãng lai, blank), matched in any case/spacing: lower(trim(x)) IN ('khách lẻ', ...)
  staff, channel, payment   NULL when the shop does not record them
  is_cancelled  BOOLEAN, this line/order is cancelled per the rulebook (hủy/huỷ, không giao được ...)
  is_returned   BOOLEAN, this line/order is returned per the rulebook (trả hàng, đổi trả, hoàn, khách trả lại ..., also partial)
  line_kind     'item' (goods), 'service' (a fee line the rulebook counts in revenue, e.g. labour, service charge / phí phục vụ) or 'adjust' (voucher, points, discount line, shipping fee and other lines the rulebook keeps out of revenue); decide each fee by the rulebook, not by its name
Rules: use only tables/columns listed in SCHEMA, always double-quoted. Join header and line tables only on the order id. Join a catalog on every way a line can name the item, since one book often mixes them: its code or abbreviation, and its full name, compared as lower(trim(regexp_replace(x, '\\s+', ' ', 'g'))), never duplicating lines, and keep the line's own name/code when the catalog has no match: coalesce(catalog name, line name). A name written with a placeholder in the README ('Dùng <n> điểm') is a pattern: match it with LIKE ('Dùng % điểm'), never literally. Read status from the status column, or from a note column when there is none; check every listed status VALUE and classify each one. Keep every row: do NOT filter out cancelled, returned or adjust lines; flag them. If the amount is not stored, compute it (qty * unit price - line discount).
Think briefly, then answer with the SQL in one ```sql``` block."""

COLS = """CAST(order_id AS VARCHAR) AS order_id, CAST(order_date AS DATE) AS order_date, CAST(order_ts AS TIMESTAMP) AS order_ts,
  CAST(product_code AS VARCHAR) AS product_code, CAST(product_group AS VARCHAR) AS product_group, trim(regexp_replace(CAST(product_name AS VARCHAR), '\\s+', ' ', 'g')) AS product_name, CAST(qty AS DOUBLE) AS qty,
  CAST(unit AS VARCHAR) AS unit, CAST(amount AS DOUBLE) AS amount, nullif(trim(CAST(customer_id AS VARCHAR)), '') AS customer_id, nullif(trim(CAST(customer_name AS VARCHAR)), '') AS customer_name,
  CAST(staff AS VARCHAR) AS staff, CAST(channel AS VARCHAR) AS channel, CAST(payment AS VARCHAR) AS payment,
  coalesce(CAST(is_cancelled AS BOOLEAN), false) AS line_cancelled, coalesce(CAST(is_returned AS BOOLEAN), false) AS line_returned,
  lower(trim(CAST(line_kind AS VARCHAR))) AS line_kind"""
REV = "line_kind IN ('item', 'service')"


def unit_spread(db, sql):
    """(median over items sold in 2+ units of highest / lowest price per qty, number of such items, the worst item). With qty in one base unit an item's price per qty is
    about the same on every unit. A large spread means some lines were not converted."""
    rows = db.execute(f"""WITH t AS (SELECT product_name p, lower(trim(unit)) u, SUM(TRY_CAST(amount AS DOUBLE)) / NULLIF(SUM(TRY_CAST(qty AS DOUBLE)), 0) price FROM ({sql})
        WHERE TRY_CAST(qty AS DOUBLE) > 0 AND TRY_CAST(amount AS DOUBLE) > 0 AND product_name IS NOT NULL AND unit IS NOT NULL GROUP BY 1, 2 HAVING COUNT(*) >= 3)
        SELECT p, MAX(price) / MIN(price) r FROM t WHERE price > 0 GROUP BY p HAVING COUNT(*) >= 2 ORDER BY r""").fetchall()
    if not rows:
        return None, 0, None
    return rows[len(rows) // 2][1], len(rows), rows[-1][0]


def mixed_price_items(db, table):
    """Items whose price per qty differs over 3x between their own item lines (5th vs 95th percentile, 20+ priced lines). In one
    base unit an item's price per qty is about constant. A gap the size of a unit factor means lines in the larger unit were not
    converted, or lines already in the base unit were multiplied again. Unlike unit_spread it needs no unit column."""
    return [p for (p,) in db.execute(f"""SELECT product_name FROM (SELECT product_name, TRY_CAST(amount AS DOUBLE) / TRY_CAST(qty AS DOUBLE) x FROM {table}
        WHERE line_kind = 'item' AND TRY_CAST(qty AS DOUBLE) > 0 AND TRY_CAST(amount AS DOUBLE) > 0 AND product_name IS NOT NULL)
        GROUP BY 1 HAVING COUNT(*) >= 20 AND quantile_cont(x, 0.95) > 3 * quantile_cont(x, 0.05) ORDER BY COUNT(*) DESC""").fetchall()]


def qty_test(db, prefix):
    """Drafting only: the fingerprint has no qty, so a view that converts no line, or every line, wins votes as well as a right one."""
    bad = mixed_price_items(db, f"{prefix}sales")
    if len(bad) >= 2:  # one item can have a real outlier (a free gift, a bulk price)
        raise ValueError(f"price per qty differs over 3x between lines of the same item on {len(bad)} items, e.g. '{bad[0]}': qty is not in one base unit. "
                         "Multiply by the factor only the lines recorded in the larger unit (compare the line's own unit with the conversion table), "
                         "keep lines already in the base unit as recorded")


def parse_select(text):
    text = text.split("</think>")[-1]
    s = re.findall(r"```(?:sql)?\s*(.*?)```", text, re.S)
    sql = s[-1].strip().rstrip(";") if s else ""
    if not re.match(r"(?is)\s*(with|select)\b", sql) or re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma)\b", sql):
        raise ValueError("need one read-only SELECT in ```sql```")
    return sql


def build(db, sql, prefix="v_"):
    """Materialize the four tables from the model's SELECT. Return a fingerprint, raise if unusable."""
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}sales_all AS
        WITH src AS ({sql}), t0 AS (SELECT {COLS} FROM src),
        -- rulebook: one customer = one customer code, else one name up to spelling (case, accents, spaces, punctuation)
        t1 AS (SELECT * REPLACE (coalesce(customer_id, nullif(regexp_replace(replace(strip_accents(lower(customer_name)), 'đ', 'd'), '[^a-z0-9]', '', 'g'), '')) AS customer_id) FROM t0),
        t AS (SELECT * EXCLUDE (customer_id, customer_name), customer_id AS customer_key,
              CASE WHEN customer_id IS NOT NULL THEN mode(customer_name) OVER (PARTITION BY customer_id) END AS customer_name FROM t1)
        SELECT *, bool_or(line_cancelled) OVER w AS is_cancelled, bool_or(line_returned) OVER w AS is_returned,
               NOT bool_or(line_cancelled OR line_returned) OVER w AS is_valid
        FROM t WINDOW w AS (PARTITION BY order_id)""")
    db.execute(f"CREATE OR REPLACE TABLE {prefix}sales AS SELECT * FROM {prefix}sales_all WHERE is_valid AND {REV}")
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}orders_all AS SELECT order_id, min(order_date) AS order_date, min(order_ts) AS order_ts,
        bool_or(is_cancelled) AS is_cancelled, bool_or(is_returned) AS is_returned, bool_and(is_valid) AS is_valid,
        any_value(customer_key) AS customer_key, any_value(customer_name) AS customer_name, any_value(staff) AS staff, any_value(channel) AS channel, any_value(payment) AS payment,
        coalesce(sum(amount) FILTER (WHERE {REV}), 0) AS revenue, count(*) FILTER (WHERE {REV}) AS n_lines
        FROM {prefix}sales_all GROUP BY order_id""")
    db.execute(f"CREATE OR REPLACE TABLE {prefix}orders AS SELECT * FROM {prefix}orders_all WHERE is_valid")
    rows, orders, rev, ncan, nret, dated, bad = db.execute(f"""SELECT count(*), count(DISTINCT order_id),
        round(sum(amount) FILTER (WHERE is_valid AND {REV})), count(DISTINCT order_id) FILTER (WHERE is_cancelled),
        count(DISTINCT order_id) FILTER (WHERE is_returned), avg(CAST(order_date IS NOT NULL AS INT)),
        count(*) FILTER (WHERE line_kind IS NULL OR line_kind NOT IN ('item', 'service', 'adjust')) FROM {prefix}sales_all""").fetchone()
    if not rows or not rev or rev <= 0 or dated < 0.95 or bad:
        raise ValueError(f"unusable view: rows={rows} revenue={rev} share of rows with order_date={dated:.2f} rows with invalid line_kind={bad}")
    dup = rows - db.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {prefix}sales_all)").fetchone()[0]
    if dup > 0.01 * rows:  # identical lines were already merged at load (rulebook), so copies here come from a join fan-out
        raise ValueError(f"{dup} lines are exact copies: a join duplicates lines; join each lookup table on a unique key (one row per key)")
    nul = db.execute(f"SELECT avg(CAST(product_name IS NULL AS INT)) FROM {prefix}sales_all WHERE line_kind = 'item'").fetchone()[0] or 0
    if 0.02 < nul < 1:  # a catalog join that drops unmatched names loses their sales from every per-product question
        raise ValueError(f"product_name is NULL on {nul:.0%} of item lines: keep the line's own name when the catalog has no match")
    multi = db.execute(f"SELECT avg(CAST(n > 1 AS INT)) FROM (SELECT count(DISTINCT order_date) AS n FROM {prefix}sales_all GROUP BY order_id)").fetchone()[0]
    if multi > 0.02:  # one order has one date: else order_id merges different orders
        raise ValueError(f"order_id is not unique per order: {multi:.0%} of order_ids span several dates; build a unique id")
    adj = db.execute(f"""SELECT count(*) FROM {prefix}sales_all WHERE is_valid AND line_kind = 'adjust'
        AND product_name IN (SELECT product_name FROM {prefix}sales_all WHERE line_kind = 'item')""").fetchone()[0]
    if adj > 0.005 * rows:  # an 'adjust' line named like a sold product is that product's sale (e.g. one with a line discount)
        raise ValueError(f"{adj} lines of valid orders are 'adjust' but carry a product name that is sold as an item: "
                         "adjust is only for voucher / points / order-discount / non-revenue fee lines, a discounted product line is an item")
    top = db.execute(f"""SELECT any_value(customer_name), count(*) AS n FROM {prefix}orders WHERE customer_key IS NOT NULL
        GROUP BY customer_key ORDER BY n DESC LIMIT 2""").fetchall()
    nord = db.execute(f"SELECT count(*) FROM {prefix}orders").fetchone()[0]
    if top and top[0][1] > 0.05 * nord and top[0][1] > 10 * (top[1][1] if len(top) > 1 else 0):  # one 'customer' on a large share of orders is a placeholder
        raise ValueError(f"customer_name '{top[0][0]}' is on {top[0][1] / nord:.0%} of valid orders, over 10x any other customer: that is the walk-in label; "
                         "map every spelling of it to NULL, comparing lower(trim(name))")
    return rows, orders, rev, ncan, nret


_raw = {}


def raw_values(db):
    """Every text value of the shop's own tables, lower-cased (not the views or scratch tables): what a mapped column may hold."""
    cols = tuple(db.execute("SELECT table_name, column_name FROM information_schema.columns WHERE data_type = 'VARCHAR' AND table_name <> 'calendar' "
                            "AND NOT starts_with(table_name, 'v_') AND NOT starts_with(table_name, '_cand_') ORDER BY 1, 2").fetchall())
    key = (id(db), cols)  # the column list too: a closed connection's id can come back for another shop
    if key not in _raw:
        _raw[key] = {str(x[0]).strip().lower() for tab, col in cols
                     for x in db.execute(f'SELECT DISTINCT "{col}" FROM "{tab}" WHERE "{col}" IS NOT NULL LIMIT 50000').fetchall()}
    return _raw[key]


def missed_names(db, prefix, where, limit=4):
    """The commonest item names a catalog join missed, and the column of the shop's tables that writes each one: what a repair needs to see."""
    names = db.execute(f"SELECT product_name, count(*) FROM {prefix}sales_all WHERE line_kind = 'item' AND product_name IS NOT NULL AND {where} "
                       f"GROUP BY 1 ORDER BY 2 DESC LIMIT {limit}").fetchall()
    cols = db.execute("SELECT table_name, column_name FROM information_schema.columns WHERE data_type = 'VARCHAR' AND table_name <> 'calendar' "
                      "AND NOT starts_with(table_name, 'v_') AND NOT starts_with(table_name, '_cand_') AND NOT starts_with(column_name, '_') ORDER BY 1, 2").fetchall()
    said = []
    for name, n in names:
        at = [f'"{tab}"."{col}"' for tab, col in cols if db.execute(f"""SELECT 1 FROM "{tab}" WHERE lower(trim("{col}")) = lower(trim(?)) LIMIT 1""", [name]).fetchone()]
        said.append(f"'{name}' ({n} lines, written in {', '.join(at[:3]) or 'no column'})")
    return ". Missed lines: " + "; ".join(said) if said else ""


def catalog_test(db, prefix):
    """Drafting only (a reused view is not re-judged): a product_group constant no table holds ('khác', 'không rõ'), or a
    NULL group on part of the lines, stands for the lines the catalog join missed, so every per-group question loses them."""
    rows = db.execute(f"SELECT count(*) FROM {prefix}sales_all").fetchone()[0]
    made = [(g, n) for g, n in db.execute(f"SELECT product_group, count(*) FROM {prefix}sales_all WHERE line_kind = 'item' AND product_group IS NOT NULL "
                                          "GROUP BY 1 ORDER BY 2 DESC").fetchall() if g.strip().lower() not in raw_values(db)]
    if sum(n for _, n in made) > 0.02 * rows:
        where = "product_group IN (" + ", ".join("'" + g.replace("'", "''") + "'" for g, _ in made) + ")"
        raise ValueError(f"product_group '{made[0][0]}' is on {made[0][1]} lines but is no value of any table: those lines did not match the "
                         "catalog; match it on every way a line names the item (code or abbreviation, and full name)" + missed_names(db, prefix, where))
    nul = db.execute(f"SELECT avg(CAST(product_group IS NULL AS INT)) FROM {prefix}sales_all WHERE line_kind = 'item'").fetchone()[0] or 0
    if 0.02 < nul < 1:  # some lines have a group, others none: the same missed join, written as NULL
        raise ValueError(f"product_group is NULL on {nul:.0%} of item lines while the other lines have one: those lines did not match the "
                         "catalog; match it on every way a line names the item (code or abbreviation, and full name)" + missed_names(db, prefix, "product_group IS NULL"))


def read_tables(db, sql):
    """The shop's own tables a mapping SELECT reads."""
    tabs = [t for (t,) in db.execute("SELECT table_name FROM information_schema.tables").fetchall()]
    return [t for t in tabs if t != "calendar" and not t.startswith(("v_", "_cand_")) and re.search(r'(?i)"' + re.escape(t) + r'"|\b' + re.escape(t) + r'\b', sql)]


def time_test(db, prefix, sql):
    """order_ts with no time of day on any line while a table the mapping reads records times (often a date column next to a
    creation-time column): every shift or hour question then finds no order."""
    n, timed = db.execute(f"SELECT count(*), count(*) FILTER (WHERE order_ts IS NOT NULL AND CAST(order_ts AS TIME) <> TIME '00:00:00') FROM {prefix}sales_all").fetchone()
    if not n or timed >= 0.05 * n:
        return
    for t in read_tables(db, sql):
        for c, ty in db.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{t}'").fetchall():
            ty = ty.upper()
            try:
                if "TIMESTAMP" in ty:
                    k, m = db.execute(f"""SELECT count(*) FILTER (WHERE CAST("{c}" AS TIME) <> TIME '00:00:00'), count("{c}") FROM "{t}" """).fetchone()
                elif ty == "TIME":
                    k, m = db.execute(f"""SELECT count("{c}"), count(*) FROM "{t}" """).fetchone()
                else:
                    continue
            except Exception:
                continue
            if m and k > 0.5 * m:
                raise ValueError(f'order_ts has no time of day on any of {n} lines, but "{t}"."{c}" records times: take order_ts from the column '
                                 "that holds the time the order was created (rulebook: shifts and hours go by it), combined with the date if it holds only a time")


def split_test(db, prefix):
    """An item whose lines are priced per qty a whole factor k apart (k >= 2) on a few percent of its lines: qty is not in one
    base unit there, e.g. a unit factor applied to lines already in the base unit, or not applied to lines in the larger unit."""
    rows = db.execute(f"""WITH s AS (SELECT product_name p, amount / qty pr FROM {prefix}sales_all WHERE line_kind = 'item' AND qty > 0 AND amount > 0 AND product_name IS NOT NULL),
        m AS (SELECT p, median(pr) md, count(*) n FROM s GROUP BY p HAVING count(*) >= 10),
        r AS (SELECT s.p, m.n, greatest(s.pr / m.md, m.md / s.pr) x FROM s JOIN m USING (p)),
        o AS (SELECT p, n, count(*) FILTER (WHERE x >= 1.8 AND abs(x - round(x)) <= 0.1 * round(x)) k, median(x) FILTER (WHERE x >= 1.8 AND abs(x - round(x)) <= 0.1 * round(x)) f FROM r GROUP BY p, n)
        SELECT p, n, k, round(f) FROM o WHERE k >= 3 AND (k >= 0.02 * n OR k >= 20) ORDER BY k DESC""").fetchall()
    if rows:
        p, n, k, f = rows[0]
        more = f" (and {len(rows) - 1} more items)" if len(rows) > 1 else ""
        raise ValueError(f"item '{p}': {k} of its {n} lines are priced per qty about {f:.0f} times apart from the others{more}, so qty is not in one base unit on them. "
                         "Multiply by a unit's factor only the lines recorded in that unit (match the conversion table on the item AND on the line's unit), times 1 for lines already in the base unit")


def tests(db, prefix, sql):
    catalog_test(db, prefix)
    time_test(db, prefix, sql)
    split_test(db, prefix)
    qty_test(db, prefix)


def check(db, texts):
    """Execute every draft into scratch tables -> [(fingerprint, sql) or (None, error)]."""
    out = []
    for t in texts:
        timer = threading.Timer(60, db.interrupt)  # a draft that runs for a minute is a join blow-up
        timer.start()
        try:
            sql = parse_select(t)
            fp = build(db, sql, "_cand_")
            tests(db, "_cand_", sql)
            out.append((fp, sql))
        except Exception as e:
            out.append((None, str(e)[:300]))
        finally:
            timer.cancel()
    for n in ("sales_all", "sales", "orders_all", "orders"):
        db.execute(f"DROP TABLE IF EXISTS _cand_{n}")
    return out


def qty_by_price(sql):
    """The view's SELECT with qty in base units, the factors read from the prices: on one item, a unit priced k times its
    cheapest unit holds k of them. For a shop whose qty repair keeps failing unit_spread, so no conversion table is needed."""
    return f"""WITH _b AS ({sql}),
_p AS (SELECT product_name AS n, lower(trim(unit)) AS u, median(amount / qty) AS pr FROM _b WHERE qty > 0 AND amount > 0 AND product_name IS NOT NULL GROUP BY 1, 2),
_k AS (SELECT n, u, greatest(1, round(pr / min(pr) OVER (PARTITION BY n))) AS k FROM _p)
SELECT _b.* REPLACE (_b.qty * coalesce(_k.k, 1) AS qty) FROM _b LEFT JOIN _k ON _b.product_name = _k.n AND lower(trim(_b.unit)) = _k.u"""


def _qty_sum(db):
    return db.execute("SELECT round(sum(TRY_CAST(qty AS DOUBLE))) FROM v_sales_all WHERE line_kind = 'item'").fetchone()[0]


def _by_qty_vote(sqls, rank):
    """The fingerprint has no qty: within one fingerprint the drafts vote on the item qty total (holdout biz_106 seed 3: 5 drafts 145723, 1 draft 166763, the fill tie-break took the 1)."""
    votes = Counter(rank[s][0] for s in sqls)
    return max(rank, key=lambda s: (votes[rank[s][0]], *rank[s][1:]))


def pick_soft(db, texts):
    """No draft passed: the drafts that only failed the catalog test still map dates, orders, amounts and quantities, and the
    questions that need none of the groups outnumber the ones that do. Most common fingerprint, built, or (None, 0)."""
    fps = {}
    for t in texts:
        for part in reversed(t.split("\n#REPAIR\n")):
            try:
                sql = parse_select(part)
                fps.setdefault(build(db, sql, "_cand_"), []).append(sql)
                break
            except Exception:
                continue
    for n in ("sales_all", "sales", "orders_all", "orders"):
        db.execute(f"DROP TABLE IF EXISTS _cand_{n}")
    if not fps:
        return None, 0
    sqls = max(fps.values(), key=len)
    rank = {}
    for s in dict.fromkeys(sqls):  # same numbers: prefer the qty most drafts agree on, then the draft whose catalog join holds, then the one filling the most lines
        build(db, s)
        try:
            tests(db, "v_", s)
            ok = 1
        except ValueError:
            ok = 0
        rank[s] = [_qty_sum(db), ok, db.execute("SELECT count(product_group) + count(product_code) FROM v_sales_all WHERE line_kind = 'item'").fetchone()[0]]
    best = _by_qty_vote(sqls, rank)
    build(db, best)
    return best, len(sqls)


def pick(db, checked):
    """Keep the SELECT of the most common fingerprint (ties: first) and build the views from it."""
    fps = {}
    for fp, sql in checked:
        if fp:
            fps.setdefault(fp, []).append(sql)
    if not fps:
        return None, 0
    sqls = max(fps.values(), key=len)
    rank = {}
    for s in dict.fromkeys(sqls):  # same numbers: prefer the qty most drafts agree on, then the draft whose catalog join fills the most item lines
        build(db, s)
        rank[s] = [_qty_sum(db), db.execute("SELECT count(product_group) + count(product_code) FROM v_sales_all WHERE line_kind = 'item'").fetchone()[0]]
    best = _by_qty_vote(sqls, rank)  # ties: first
    build(db, best)
    return best, len(sqls)


def hide_raw(db, meta, sql):
    """(meta without the raw sales tables the views map, one line naming their columns): samples that read those tables skip every status, unit and name rule the views apply.
    Lookup tables stay. A table the mapping reads counts as raw sales when it holds at least a fifth as many rows as the sales lines."""
    lines = db.execute("SELECT count(*) FROM v_sales_all").fetchone()[0]
    used = {x.lower() for x in re.findall(r'(?i)\b(?:from|join)\s+"?(\w+)"?', sql)}
    raw = [t for t in meta["tables"] if t["name"].lower() in used and t["name"] != "calendar" and not t["name"].startswith("v_") and t["rows"] >= 0.2 * lines]
    if not raw:
        return meta, ""
    names = {t["name"] for t in raw}
    cols = "; ".join(f'"{t["name"]}"(' + ", ".join(f'"{c[0]}"' for c in t["columns"] if c[0] not in ("_sheet", "_source")) + ")" for t in raw)
    return dict(meta, tables=[t for t in meta["tables"] if t["name"] not in names]), ("\n\nRAW SALES TABLES, only for a column the canonical views do not have (the views already apply the rulebook's status, unit and name rules): " + cols)


def describe(db, sql, n, k):
    nv, no = db.execute("SELECT (SELECT count(*) FROM v_sales), (SELECT count(*) FROM v_orders)").fetchone()
    cols = ("order_ts", "product_code", "product_group", "product_name", "qty", "unit", "customer_key", "customer_name", "staff", "channel", "payment")
    filled = db.execute("SELECT " + ", ".join(f"count({c})" for c in cols) + " FROM v_sales_all").fetchone()
    empty = [c for c, x in zip(cols, filled) if not x]
    return (f'"v_sales" ({nv} rows) = item/service lines of VALID orders only (not cancelled, no returned line): revenue = SUM(amount), no status filter needed.\n'
            f'"v_orders" ({no} rows) = VALID orders only, one row per order: order_id, order_date, order_ts, customer_key, customer_name, staff, channel, payment, revenue, n_lines.\n'
            '"v_sales_all" / "v_orders_all" = every line / order with is_cancelled, is_returned, is_valid, line_kind (item|service|adjust).\n'
            "v_sales columns: order_id, order_date, order_ts, product_code, product_group, product_name, qty, unit, amount, customer_key, customer_name, staff, channel, payment, line_kind. qty is already in base units: for best sellers and quantities sold, use SUM(qty) and do not convert again. "
            "A line_kind 'service' line (a fee such as phí phục vụ) counts in revenue but is no item: rank, count or list items WHERE line_kind = 'item'.\n"
            "Customers (rulebook): one customer = one customer_key (the customer code, else the name ignoring case/accents/spaces/punctuation); customer_name = its usual spelling; "
            "both NULL for walk-ins, who are no customer. Count customers with COUNT(DISTINCT customer_key), rank them GROUP BY customer_key and answer with customer_name."
            + (f" Always NULL here, so this shop does not record them: {', '.join(empty)}; a question that needs one of them is unanswerable (rulebook §7), not 0." if empty else "")
            + f"\nBuilt from the raw tables by this mapping ({n}/{k} independent drafts agree):\n{sql}")
