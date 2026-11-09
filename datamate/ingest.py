"""Load one business (CSV/XLSX/SQLite) into DuckDB with typed, named tables and a schema prompt."""
import csv
import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd


def norm(x):
    x = unicodedata.normalize("NFKD", str(x).replace("đ", "d").replace("Đ", "D"))
    return re.sub(r"[^a-z0-9]+", "_", x.encode("ascii", "ignore").decode().lower()).strip("_")


def ckey(x):
    """Column-name key: like norm, but camelCase headers (MaHoaDon) split into words."""
    return norm(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(x)))


def quote(x):
    return '"' + str(x).replace('"', '""') + '"'


HEADER_KEYS = ("stt", "ma", "ten", "ngay", "thoi_gian", "gio", "so_luong", "sl", "don_gia", "gia", "thanh_tien",
               "tien", "so_hoa_don", "trang_thai", "khach", "khach_hang", "hoa_don", "hang", "hang_hoa", "san_pham",
               "mon", "don_vi", "dvt", "tong_tien", "tong", "nhom", "loai", "ghi_chu", "nhan_vien", "tho", "xe",
               "lam_gi", "so", "size", "hang_sx", "qty", "price", "amount", "date", "status", "id", "name")


def header_score(row):
    vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
    if len(vals) < 2:
        return -1
    hits = sum(any(ckey(v) == k or ckey(v).startswith(k + "_") for k in HEADER_KEYS) for v in vals)
    nums = sum(bool(re.fullmatch(r"[\d,. /:-]+", v)) for v in vals)
    return hits * 5 + min(len(set(vals)), 12) - nums * 2 if hits else -1


def fix_header(rows):
    scores = [header_score(r) for r in rows[:25]]
    idx = max(range(len(scores)), key=scores.__getitem__) if scores else 0
    if not scores or scores[idx] < 7:
        idx = 0
    names, counts = [], defaultdict(int)
    for i, v in enumerate(rows[idx] if rows else []):
        col = str(v).strip() if pd.notna(v) and str(v).strip() else f"col_{i + 1}"
        counts[col] += 1
        names.append(f"{col}_{counts[col]}" if counts[col] > 1 else col)
    width = max([len(names)] + [len(r) for r in rows[idx + 1:]])
    names += [f"col_{i + 1}" for i in range(len(names), width)]
    body = [list(r) + [None] * (width - len(r)) for r in rows[idx + 1:]]
    return pd.DataFrame(body, columns=names, dtype=object)


DMY = r"^(?P<day>\d{1,2})[/.-](?P<month>\d{1,2})[/.-](?P<year>\d{4})"
YMD = r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
THG = r"^(?P<day>\d{1,2}) thg (?P<month>\d{1,2}),? (?P<year>\d{4})"
DM = r"^(?P<day>\d{1,2})/(?P<month>\d{1,2})"
HMS = r"(?:[ T,]+(?P<hour>\d{1,2})[:h](?P<minute>\d{2})(?::(?P<second>\d{2}))?(?:\.\d+)?)?$"


def to_dates(t, year=None):
    """Vectorized dd/mm/yyyy, yyyy-mm-dd, 'd thg m yyyy', d/m (+ sheet/file year), Excel serial, else NaT."""
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    for pat in (DMY + HMS, YMD + HMS, THG + HMS) + ((DM + HMS,) if year else ()):
        p = t.str.extract(pat, flags=re.IGNORECASE)
        if "year" not in p:
            p["year"] = year
        p = p.dropna(subset=["day", "month"])
        if len(p):
            p = p[["year", "month", "day", "hour", "minute", "second"]].fillna(0).astype(int)
            out.loc[p.index] = out.loc[p.index].fillna(pd.to_datetime(p, errors="coerce"))
    serial = t.str.fullmatch(r"[3-6]\d{4}(\.0+)?").fillna(False)
    if serial.any():
        out.loc[serial] = pd.Timestamp("1899-12-30") + pd.to_timedelta(t[serial].astype(float).astype(int), unit="D")
    return out


def to_number(t):
    """'1.234.000', '1,234,000', '198 000', '55.000đ', '25k', '8tr600', '1,27tr' -> float, else NaN."""
    clean = t.str.replace(r"\s*(?:₫|vnđ|vnd|đ|đồng)\s*$", "", regex=True, flags=re.IGNORECASE).str.replace(" ", " ")
    clean = clean.where(~clean.str.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+").fillna(False), clean.str.replace(" ", "", regex=False))
    out = pd.Series(float("nan"), index=t.index)
    m = clean.str.extract(r"^([+-]?\d+(?:[.,]\d+)?)\s*(k|tr|triệu)(\d{1,3})?$", flags=re.IGNORECASE).dropna(subset=[0])
    if len(m):
        scale = m[1].str.lower().map({"k": 1e3, "tr": 1e6, "triệu": 1e6})
        frac = m[2].fillna("").str.ljust(3, "0").replace("000", "0").astype(float) * scale / 1e3
        out.loc[m.index] = (m[0].str.replace(",", ".", regex=False).astype(float) * scale + frac.where(m[2].notna(), 0)).round()
    us = clean.str.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?").fillna(False)
    eu = clean.str.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?").fillna(False)
    clean.loc[us] = clean.loc[us].str.replace(",", "", regex=False)
    clean.loc[eu] = clean.loc[eu].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    dec = clean.str.fullmatch(r"[+-]?\d+,\d+").fillna(False)
    clean.loc[dec] = clean.loc[dec].str.replace(",", ".", regex=False)
    return out.fillna(pd.to_numeric(clean, errors="coerce"))


ORDER_ID = r"(^|_)(ma|so|id)_?(don|hd|hoa_don|phieu|bill|ct|chung_tu|giao_dich)|^(ma|so)_?(dh|hd)$|^(hd|hoa_don|don_hang|don_so|so_don|order_?id|bill|stt)$"
STATUS_COL = r"(^|_)(trang_?thai|tinh_?trang|status|xu_?ly|ghi_?chu)"
CANCELLED = r"(?:^|_)(?:huy|cancel\w*|khong_giao_duoc)(?:_|$)"
RETURNED = r"(?:^|_)(?:tra_hang|tra_lai|tra_ve|doi_tra|hoan_tra|hoan_hang|return\w*)(?:_|$)|khach_(?:mang_)?tra$"
NOTE_ROW = re.compile(r"(?i)^\s*(ghi chú|ghi chu|tổng|tong cong|cộng|total)\b")


def prepare(df, year=None):
    df = df.dropna(how="all").copy()
    if not df.empty and len(df.columns) > 1:
        text = df.astype("string").apply(lambda col: col.str.strip().str.normalize("NFC")).replace("", pd.NA)
        repeats = sum(text[c].eq(str(c)).fillna(False).astype(int) for c in df.columns)  # repeated header rows
        sparse = text.notna().sum(axis=1) <= max(2, len(df.columns) // 3)
        note = text.apply(lambda col: col.str.match(NOTE_ROW).fillna(False)).any(axis=1)  # footer/subtotal rows
        keys = [c for c in df.columns if re.search(ORDER_ID + r"|(^|_)(ngay|date)", ckey(c))]
        keyless = text[keys].isna().any(axis=1) | text[keys].apply(lambda col: col.str.match(NOTE_ROW).fillna(False)).any(axis=1) if keys else False
        df = df.loc[(repeats < max(2, len(df.columns) // 2)) & ~(note & (sparse | keyless))].copy()
    for c in list(df.columns):
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        native = s.map(lambda v: hasattr(v, "year") and hasattr(v, "month"))
        t = s.map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))).astype("string").str.strip().replace("", pd.NA)
        valid = t.dropna()
        if valid.empty:
            df[c] = t
            continue
        key = ckey(c)
        if native.any() or valid.str.contains(r"^\d{1,4}[/.-]\d|\d thg \d", regex=True).mean() > 0.8 or re.search(r"(^|_)(ngay|date|thoi_gian)", key):
            dt = to_dates(t, year)
            if native.any():
                dt.loc[native] = pd.to_datetime(s[native].map(pd.Timestamp))
            if dt.notna().sum() / len(valid) > 0.8:
                df[c] = dt
                continue
        is_id = bool(re.search(r"(^|_)(ma|id|code|stt|sdt|phone|dien_thoai|barcode|mst|bien_so|so_dt|imei|serial|so_seri|dia_chi)(_|$)", key))
        if is_id or valid.str.match(r"^0\d+$").any():
            df[c] = t
            continue
        num = to_number(t)
        if num.notna().sum() / len(valid) > 0.97:
            df[c] = num
            continue
        qty = t.str.extract(r"^([+-]?\d+(?:[.,]\d+)?)\s*([^\d\s.,=:/-][^=:/]{0,15})$")  # "2 gói", "1 ly"
        both = num.fillna(pd.to_numeric(qty[0].str.replace(",", ".", regex=False), errors="coerce"))
        if both.notna().sum() / len(valid) > 0.97 and qty[1].notna().any():
            df[c] = both
            df.insert(df.columns.get_loc(c) + 1, f"{c} (đơn vị ghi kèm)", qty[1].astype("string"))
            continue
        df[c] = t
    # Rulebook §8: continuation lines of a grouped order leave order-level cells blank -> copy from the first line.
    ids = sorted((c for c in df.columns if re.search(ORDER_ID, ckey(c))), key=lambda c: ckey(c) == "stt")  # STT last
    if ids and len(df) and df[ids[0]].iloc[0] is not pd.NA and pd.notna(df[ids[0]].iloc[0]):
        blank = df[ids[0]].isna()
        if blank.mean() > 0.05:
            group = [c for c in df.columns if df.loc[blank, c].isna().mean() > 0.95]
            # fill continuation lines from their own order's first line only. A plain ffill would carry
            # a sparse order-level cell (note "trả hàng", order discount) into later orders
            first = df[group].groupby((~blank).cumsum()).transform("first")
            df.loc[blank, group] = first.loc[blank]
    if ids:
        # Rulebook: blank cells on later lines of an order take the order's value (date, customer, note, status...).
        g = df.groupby(df[ids[0]], sort=False)
        for c in df.columns:
            if c != ids[0] and df[c].isna().any() and not pd.api.types.is_numeric_dtype(df[c]) and (g[c].nunique() <= 1).mean() > 0.99:
                df[c] = df[c].fillna(g[c].transform("first"))
        df = df.drop_duplicates()  # Rulebook §8: identical lines of the same order are an export error, count once
    df = df.dropna(how="all", subset=[c for c in df.columns if not str(c).startswith("_")] or None)
    # Status classes as flags, so every sampled query excludes cancelled/returned orders the same way.
    for c in [c for c in df.columns if re.search(STATUS_COL, ckey(c)) and df[c].dtype == "string"]:
        k = df[c].map(lambda v: norm(v) if isinstance(v, str) else "")
        huy, tra = k.str.contains(CANCELLED), k.str.contains(RETURNED)
        if huy.any() or tra.any():
            df["_huy"], df["_tra"] = huy, tra
            if ids:
                order = df[ids[0]]
                df["_don_huy"] = df["_huy"].groupby(order).transform("any").fillna(df["_huy"]).astype(bool)
                df["_don_tra"] = df["_tra"].groupby(order).transform("any").fillna(df["_tra"]).astype(bool)
            break
    return df


def union_key(name):
    """Monthly/yearly split tables share a base name once periods are removed."""
    return re.sub(r"_+", "_", re.sub(r"(?:^|_)(?:t|thang|nam)?_?(?:0?[1-9]|1[0-2])?_?20\d{2}(?=_|$)|(?:^|_)(?:t|thang)_?(?:0?[1-9]|1[0-2])(?=_|$)", "_", name)).strip("_") or name


def read_rows(f):
    """Yield (table_name, rows) for every sheet/table in a file."""
    ext = f.suffix.lower()
    if ext == ".xlsx":
        from python_calamine import CalamineWorkbook
        book = CalamineWorkbook.from_path(str(f))
        try:
            for sheet in book.sheet_names:
                rows = book.get_sheet_by_name(sheet).to_python(skip_empty_area=False)
                yield f"{f.stem}_{sheet}" if len(book.sheet_names) > 1 else f.stem, sheet, rows
        finally:
            book.close()
    elif ext == ".csv":
        try:
            text = f.read_text(encoding="utf-8-sig")
        except UnicodeError:
            text = f.read_text(encoding="cp1258")
        try:
            dialect = csv.Sniffer().sniff(text[:16384], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        yield f.stem, None, list(csv.reader(text.splitlines(), dialect))
    elif ext in (".db", ".sqlite", ".sqlite3"):
        with sqlite3.connect(f"file:{f.resolve()}?mode=ro", uri=True) as src:
            for (table,) in src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
                cur = src.execute(f"SELECT * FROM {quote(table)}")
                yield table, None, [[d[0] for d in cur.description]] + [list(r) for r in cur.fetchall()]


def profile(db, n, source):
    """Row count and per-column type + top values (text) or range (others) of one table/view, for the schema prompt."""
    prof = []
    for col, typ in [(r[0], r[1]) for r in db.execute(f"DESCRIBE {quote(n)}").fetchall()]:
        if typ in ("VARCHAR",):
            d = db.execute(f"SELECT count(DISTINCT {quote(col)}) FROM {quote(n)}").fetchone()[0]
            vals = [r[0] for r in db.execute(f"SELECT {quote(col)} FROM {quote(n)} WHERE {quote(col)} IS NOT NULL GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {25 if d <= 25 else 4}").fetchall()]
            prof.append((col, typ, d, vals))
        else:
            lo, hi = db.execute(f"SELECT min({quote(col)}), max({quote(col)}) FROM {quote(n)}").fetchone()
            prof.append((col, typ, None, [str(lo), str(hi)]))
    return {"name": n, "source": source, "rows": db.execute(f"SELECT count(*) FROM {quote(n)}").fetchone()[0], "columns": prof}


def spelling_key(v):
    """A name without case, accents or extra spaces."""
    s = unicodedata.normalize("NFD", str(v).replace("đ", "d").replace("Đ", "D"))
    return " ".join("".join(c for c in s if unicodedata.category(c) != "Mn").lower().split())


def spelling_map(counts):
    """{value: count} -> {value: spelling to use}, for names that differ only in the way they are written (rulebook: such
    names are one customer, one item). Forms with different accents stay apart (Bò, Bơ). A form without accents joins only
    when one accented form shares its key, since otherwise it could be either."""
    groups = defaultdict(list)
    for v, n in counts.items():
        k = spelling_key(v)
        if k and len(k) <= 80 and any(c.isalpha() for c in k):
            groups[k].append((v, n))
    out = {}
    for k, vs in groups.items():
        if len(vs) < 2:
            continue
        forms = defaultdict(list)
        for v, n in vs:
            forms[" ".join(unicodedata.normalize("NFC", str(v)).lower().split())].append((v, n))
        accented = [f for f in forms if f != k]
        best = lambda m: max(m, key=lambda x: (not x[0].isupper(), x[1]))[0]
        if len(accented) == 1:
            forms = {accented[0]: vs}
        elif accented:
            forms.pop(k, None)
        for m in forms.values():
            use = best(m)
            out.update({v: use for v, _ in m if v != use})
    return out


def canon_spellings(db, tables):
    """Writes every name of a shop one way, across all its tables, so an item or a customer is one value in the raw tables,
    the catalog joins and the views alike. Returns the number of cells changed."""
    cols, counts = [], defaultdict(int)
    for t in tables:
        if t == "calendar":
            continue
        for col, typ, *_ in db.execute(f"DESCRIBE {quote(t)}").fetchall():
            if typ == "VARCHAR" and not col.startswith("_"):
                cols.append((t, col))
                for v, n in db.execute(f"SELECT {quote(col)}, count(*) FROM {quote(t)} WHERE {quote(col)} IS NOT NULL GROUP BY 1").fetchall():
                    counts[v] += n
    mapping = spelling_map(counts)
    if not mapping:
        return 0
    db.register("_m", pd.DataFrame({"v": list(mapping), "c": list(mapping.values())}))
    db.execute("CREATE TEMP TABLE _spell AS SELECT * FROM _m")
    db.unregister("_m")
    changed = 0
    for t, col in cols:
        n = db.execute(f"SELECT count(*) FROM {quote(t)} x JOIN _spell m ON x.{quote(col)} = m.v").fetchone()[0]
        if n:
            db.execute(f"UPDATE {quote(t)} SET {quote(col)} = m.c FROM _spell m WHERE {quote(t)}.{quote(col)} = m.v")
            changed += n
    db.execute("DROP TABLE _spell")
    return changed


def load_business(root, business, canon=False):
    root = Path(root)
    folder = root / "data" / business
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    db = duckdb.connect(":memory:")
    tables = {}  # name -> source description
    for f in sorted(folder.rglob("*")):
        if not f.is_file() or f.name.startswith("~$"):
            continue
        for name, sheet, rows in read_rows(f):
            rows = [r for r in rows if any(v is not None and str(v).strip() for v in r)]
            if len(rows) < 2:
                continue
            year = re.search(r"20\d{2}", str(sheet or "")) or re.search(r"20\d{2}", f.stem)
            df = prepare(fix_header(rows), int(year.group()) if year else None)
            if df.empty:
                continue
            if sheet is not None:
                df["_sheet"] = sheet  # rulebook: a date without year takes the year of its sheet
            name = norm(name) or "table"
            base, i = name, 2
            while name in tables:
                name, i = f"{base}_{i}", i + 1
            db.register("_src", df)
            db.execute(f"CREATE TABLE {quote(name)} AS SELECT * FROM _src")
            db.unregister("_src")
            tables[name] = f.name + (f"#{sheet}" if sheet else "")
    if not tables:
        raise ValueError(f"No usable tables: {folder}")
    if canon:
        canon_spellings(db, list(tables))
    if (root / "calendar.csv").exists():
        db.execute("CREATE TABLE calendar AS SELECT * FROM read_csv_auto(?, header=true)", [str(root / "calendar.csv")])
        tables["calendar"] = "calendar.csv"

    def cols(n):
        return [(r[0], r[1]) for r in db.execute(f"DESCRIBE {quote(n)}").fetchall()]

    groups = defaultdict(list)
    for n in tables:
        groups[tuple(c for c, _ in cols(n) if c != "_sheet")].append(n)
    hidden = set()
    for names in groups.values():
        if len(names) < 2:
            continue
        view = union_key(names[0]) + "_all"
        parts = [f"SELECT *, '{tables[n]}' AS _source FROM {quote(n)}" for n in names]
        db.execute(f"CREATE VIEW {quote(view)} AS " + " UNION ALL BY NAME ".join(parts))
        tables[view] = f"UNION of {len(names)} tables: " + ", ".join(tables[n] for n in names)
        hidden.update(names)

    info = [profile(db, n, tables[n]) for n in tables if n not in hidden]
    readme = "\n\n".join(p.read_text(encoding="utf-8-sig") for p in sorted(folder.rglob("*.md")))
    return db, {"tables": info, "readme": readme}


def entity_hits(db, meta, question, limit=12):
    """Exact values of text columns whose words appear in the question (product/customer names, statuses)."""
    q = norm(question)
    qw = set(q.split("_"))
    hits = []
    for t in meta["tables"]:
        for col, typ, d, _ in t["columns"]:
            if typ != "VARCHAR" or not d or d > 20000:
                continue
            for (v,) in db.execute(f"SELECT DISTINCT {quote(col)} FROM {quote(t['name'])} WHERE {quote(col)} IS NOT NULL").fetchall():
                nv = norm(v)
                w = set(nv.split("_"))
                if len(nv) >= 3 and not nv.isdigit() and (("_" + nv + "_") in ("_" + q + "_") or (len(w) >= 2 and len(w & qw) / len(w) >= 0.75)):
                    hits.append((len(nv), t["name"], col, v))
    hits.sort(reverse=True)
    return [f"{quote(t)}.{quote(c)} = '{v}'" for _, t, c, v in hits[:limit]]


def schema_text(meta):
    out = []
    for t in meta["tables"]:
        out.append(f"TABLE {quote(t['name'])}  -- {t['rows']} rows, from {t['source']}")
        for col, typ, d, vals in t["columns"]:
            if d is None:
                out.append(f"  {quote(col)} {typ}  range [{vals[0]} .. {vals[1]}]")
            else:
                tag = "values" if d <= 25 else f"{d} distinct, e.g."
                out.append(f"  {quote(col)} {typ}  {tag} {json.dumps([str(v)[:60] for v in vals], ensure_ascii=False)}")
    return "\n".join(out)


_SAMPLE_TABLE_HINTS = ("ban_hang", "sales", "sale", "doanh_thu", "don_hang", "order", "hoa_don", "invoice", "chi_tiet",
                       "detail", "line", "san_pham", "hang_hoa", "product", "catalog", "khach", "customer", "don_vi", "unit",
                       "status", "trang_thai", "quy_doi", "conversion")
_SAMPLE_IMPORTANT = re.compile(r"(^|_)(ma|id|code|so|don|hoa|order|date|ngay|time|thoi_gian|qty|quantity|sl|so_luong|amount|price|gia|tien|total|tong|discount|giam|chiet|unit|dvt|don_vi|product|san_pham|hang|item|customer|khach|status|trang|note|ghi_chu)(_|$)")
_SAMPLE_NUMERIC = re.compile(r"(^|_)(qty|quantity|sl|so_luong|amount|price|gia|tien|total|tong|discount|giam|chiet|doanh_thu)(_|$)")


def _sample_alias(base, columns):
    alias = base
    i = 2
    while alias in columns:
        alias = f"{base}_{i}"
        i += 1
    return alias


def _sample_value(v, max_chars):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (datetime, date, time)):
        s = v.isoformat()
        return json.dumps(s, ensure_ascii=False)
    if isinstance(v, float) and math.isnan(v):
        return "NULL"
    try:
        missing = pd.isna(v)
        if isinstance(missing, bool) and missing:
            return "NULL"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float, Decimal)):
        s = str(v)
        return s[:max_chars]
    s = str(v)[:max_chars]
    return json.dumps(s, ensure_ascii=False)


def _sample_table_score(table):
    name = ckey(table.get("name", ""))
    score = sum(2 for hint in _SAMPLE_TABLE_HINTS if hint in name)
    for item in table.get("columns", ()):
        if not item:
            continue
        score += 1 if _SAMPLE_IMPORTANT.search(ckey(item[0])) else 0
    return score


def _sample_rows(db, table, max_rows):
    """Select bounded, deterministic row candidates; duplicate records are indistinguishable by design."""
    name = table["name"]
    info = [item for item in table.get("columns", ()) if item and item[0] is not None]
    if not info:
        info = [(r[0], r[1]) for r in db.execute(f"DESCRIBE {quote(name)}").fetchall()]
    columns = [str(item[0]) for item in info]
    if not columns:
        return []
    qcols = [quote(c) for c in columns]
    col_keys = {c: ckey(c) for c in columns}
    helper = _sample_alias("__datamate_sample_index", set(columns))
    interesting_helper = _sample_alias("__datamate_interesting_index", set(columns) | {helper})
    kind_helper = _sample_alias("__datamate_sample_kind", set(columns) | {helper, interesting_helper})
    count_helpers = {}
    rare_cols = [c for c in columns if re.search(r"(^|_)(unit|dvt|don_vi|status|trang_thai|tinh_trang|loai)(_|$)", col_keys[c])]
    for c in rare_cols:
        alias = _sample_alias("__datamate_count_" + norm(c), set(columns) | {helper, interesting_helper, *count_helpers.values()})
        count_helpers[c] = alias

    important = [c for c in columns if _SAMPLE_IMPORTANT.search(col_keys[c])]
    status_cols = [c for c in columns if re.search(STATUS_COL, col_keys[c]) or re.search(r"(^|_)(state|loai)(_|$)", col_keys[c])]
    numeric_cols = [c for c in columns if _SAMPLE_NUMERIC.search(col_keys[c])]
    discount_cols = [c for c in columns if re.search(r"(^|_)(discount|giam|chiet|khuyen_mai|voucher)(_|$)", col_keys[c])]
    q = lambda c: quote(c)
    null_expr = " OR ".join(f"{q(c)} IS NULL" for c in important) or "FALSE"
    status_expr = " OR ".join(
        f"lower(CAST({q(c)} AS VARCHAR)) LIKE '%huy%' OR lower(CAST({q(c)} AS VARCHAR)) LIKE '%hủy%' OR "
        f"lower(CAST({q(c)} AS VARCHAR)) LIKE '%huỷ%' OR lower(CAST({q(c)} AS VARCHAR)) LIKE '%cancel%' OR "
        f"lower(CAST({q(c)} AS VARCHAR)) LIKE '%tra%' OR lower(CAST({q(c)} AS VARCHAR)) LIKE '%trả%' OR "
        f"lower(CAST({q(c)} AS VARCHAR)) LIKE '%return%' OR lower(CAST({q(c)} AS VARCHAR)) LIKE '%hoan%' OR "
        f"lower(CAST({q(c)} AS VARCHAR)) LIKE '%khong giao%'"
        for c in status_cols
    ) or "FALSE"
    negative_expr = " OR ".join(f"TRY_CAST({q(c)} AS DOUBLE) < 0" for c in numeric_cols) or "FALSE"
    discount_expr = " OR ".join(
        f"NULLIF(trim(CAST({q(c)} AS VARCHAR)), '') IS NOT NULL AND COALESCE(TRY_CAST({q(c)} AS DOUBLE), 1) <> 0" for c in discount_cols
    ) or "FALSE"
    rare_limit = max(3, int(table.get("rows") or 0) // 10)
    rare_expr = " OR ".join(f"{q(c)} IS NOT NULL AND {q(alias)} <= {rare_limit}" for c, alias in count_helpers.items()) or "FALSE"
    interesting_expr = f"({null_expr}) OR ({status_expr}) OR ({negative_expr}) OR ({discount_expr}) OR ({rare_expr})"
    priority_expr = f"CASE WHEN {null_expr} THEN 0 WHEN {status_expr} THEN 1 WHEN {negative_expr} THEN 2 WHEN {discount_expr} THEN 3 WHEN {rare_expr} THEN 4 ELSE 5 END"

    # No generic schema guarantees a unique key. Sorting by every value is the deterministic fallback;
    # identical rows can still not be distinguished, which is harmless because they serialize identically.
    order_terms = [f"CASE WHEN {q(c)} IS NULL THEN 0 ELSE 1 END, CAST({q(c)} AS VARCHAR)" for c in columns]
    order_by = ", ".join(order_terms)
    head_n = min(4, max_rows)
    dist_n = min(3, max_rows - head_n)
    interesting_n = max_rows - head_n - dist_n
    row_count = table.get("rows")
    try:
        row_count = int(row_count)
    except (TypeError, ValueError):
        row_count = int(db.execute(f"SELECT count(*) FROM {quote(name)}").fetchone()[0])
    if row_count <= 0:
        return []
    positions = []
    for p in (max(1, row_count // 2), max(1, (3 * row_count) // 4), row_count):
        if p not in positions:
            positions.append(p)
    positions = positions[:dist_n]
    base = f"SELECT row_number() OVER (ORDER BY {order_by}) AS {q(helper)}, {', '.join(qcols)} FROM {quote(name)}"
    marked = base
    if count_helpers:
        marked = f"SELECT *, {', '.join(f'count(*) OVER (PARTITION BY {q(c)}) AS {q(a)}' for c, a in count_helpers.items())} FROM ({base}) AS base_rows"
    interesting_limit = max(interesting_n * 3, interesting_n)
    wanted = [f"SELECT 'head' AS {q(kind_helper)}, {q(helper)} AS {q(interesting_helper)}, {', '.join(qcols)} FROM marked WHERE {q(helper)} <= {head_n}"] if head_n else []
    if positions:
        wanted.append(f"SELECT 'distributed' AS {q(kind_helper)}, {q(helper)} AS {q(interesting_helper)}, {', '.join(qcols)} FROM marked WHERE {q(helper)} IN ({', '.join(map(str, positions))})")
    if interesting_limit:
        wanted.append(f"SELECT 'interesting' AS {q(kind_helper)}, {q(helper)} AS {q(interesting_helper)}, {', '.join(qcols)} FROM interesting WHERE {q(interesting_helper)} <= {interesting_limit}")
    if not wanted:
        return []
    sql = f"""WITH base AS ({base}), marked AS ({marked}), interesting AS (
        SELECT *, row_number() OVER (ORDER BY {priority_expr}, {q(helper)}) AS {q(interesting_helper)}
        FROM marked WHERE {interesting_expr}
    ), wanted AS ({' UNION ALL '.join(wanted)})
    SELECT {q(kind_helper)}, {q(interesting_helper)}, {', '.join(qcols)} FROM wanted
    ORDER BY CASE {q(kind_helper)} WHEN 'head' THEN 0 WHEN 'distributed' THEN 1 ELSE 2 END, {q(interesting_helper)}"""
    return db.execute(sql).fetchall()


def sample_text(db, meta, max_rows=10, max_cell_chars=120, max_total_chars=16000):
    """Format bounded representative rows from the visible ingested tables for a view-drafting prompt."""
    max_rows, max_cell_chars, max_total_chars = int(max_rows), max(1, int(max_cell_chars)), int(max_total_chars)
    if max_rows <= 0 or max_total_chars <= 0:
        return ""
    all_tables = [t for t in meta.get("tables", ()) if ckey(t.get("name", "")) != "calendar"]
    physical_bases = {union_key(t.get("name", "")) for t in all_tables if not str(t.get("source", "")).startswith("UNION of ")}
    tables = [t for t in all_tables if not (str(t.get("source", "")).startswith("UNION of ") and
                                            t.get("name", "").endswith("_all") and
                                            t.get("name", "")[:-4] in physical_bases)]
    tables.sort(key=lambda t: (-_sample_table_score(t), ckey(t.get("name", "")), str(t.get("name", ""))))
    blocks = []
    for table in tables:
        name = table.get("name")
        if not name:
            continue
        columns = [str(item[0]) for item in table.get("columns", ()) if item]
        lines = [f"TABLE {quote(name)}" + (f" (source: {table.get('source')})" if table.get("source") else ""),
                 "Columns: " + ", ".join(quote(c) for c in columns), "Rows:"]
        try:
            rows = _sample_rows(db, table, max_rows)
            seen = set()
            for record in rows:
                values = tuple(_sample_value(v, max_cell_chars) for v in record[2:])
                if values in seen:
                    continue
                seen.add(values)
                lines.append("[" + ", ".join(values) + "]")
        except Exception:
            lines.append("[sampling unavailable]")
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    return text[:max_total_chars]


if __name__ == "__main__":
    import sys, time
    for root in sys.argv[1:]:
        for b in sorted((Path(root) / "data").iterdir()):
            t0 = time.time()
            db, m = load_business(root, b.name)
            print(f"{b.name}: {len(m['tables'])} tables, {len(schema_text(m))} chars, {time.time() - t0:.1f}s")
