"""Hints computed from the question text alone: date windows, question-shape hints, lot tables, out-of-range periods."""
import re
from datetime import date, timedelta

from .ingest import ckey, norm, quote

def date_anchors(as_of):
    d = date.fromisoformat(as_of)
    mon = d - timedelta(days=d.weekday())
    m1 = d.replace(day=1)
    pm_end = m1 - timedelta(days=1)
    pm1 = pm_end.replace(day=1)
    pm_same = pm1.replace(day=min(d.day, pm_end.day))
    nxt = lambda x: x + timedelta(days=1)
    return "\n".join([
        f"today (hôm nay, bữa nay, hnay) = {d} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d.weekday()]}); half-open end = {nxt(d)}",
        f"the data may go on after today; those later rows have not happened yet (rulebook), so a question naming no period covers the data up to and including today ({d}); a named period keeps its own window; cancel/return status is the order's recorded status",
        f"yesterday (hôm qua, bữa qua, hqua) = {d - timedelta(days=1)} = [{d - timedelta(days=1)}, {d})",
        f"this week (tuần này) = [{mon}, {nxt(d)}); last week (tuần trước, tuần rồi) = [{mon - timedelta(days=7)}, {mon})",
        f"this month (tháng này) = [{m1}, {nxt(d)}); last month same days = [{pm1}, {nxt(pm_same)}); whole last month = [{pm1}, {m1})",
        f"year to date (từ đầu năm) = [{d.replace(month=1, day=1)}, {nxt(d)})",
        *(f"quarters of {y}: " + "; ".join(f"Q{i} (quý {i}) = [{y}-{3*i-2:02d}-01, {y + (i == 4)}-{(3*i+1) % 12 or 12:02d}-01)" for i in range(1, 5))
          for y in (d.year - 1, d.year)),
        f"last 7 days = [{d - timedelta(days=6)}, {nxt(d)}); last 30 days = [{d - timedelta(days=29)}, {nxt(d)}); last N days = [today - (N-1) days, {nxt(d)})",
    ])


def period_hints(question, as_of, db=None):
    """Explicit windows for the periods a question names (rulebook: month/quarter without year, Tết by lunar calendar)."""
    d, t, out = date.fromisoformat(as_of), norm(question), []
    first = lambda y, m: date(y + (m - 1) // 12, (m - 1) % 12 + 1, 1)
    def year_of(m_start):  # rulebook: a month/quarter without year is this year's unless it starts after today
        return d.year if first(d.year, m_start) <= d else d.year - 1
    for m, y, m2, y2 in re.findall(r"(?:^|_)(?:(?:thang_|thg_?|th_?)(\d{1,2})(?:_nam|_)?_?(\d{4}|\d{2})?|t(\d{1,2})_(\d{4}|\d{2}))(?=_|$)", t):  # tháng 5, th11/25, T5/2025 (bare T8 is often a product code)
        m, y = int(m or m2), y or y2
        if 1 <= m <= 12:
            y = (int(y) + (2000 if len(y) == 2 else 0)) if y else year_of(m)
            out.append(f"tháng {m}/{y} = [{first(y, m)}, {first(y, m + 1)})")
    # a month written with no word and no day: 03/2026, 1-2026, 8/26 (rulebook "Tháng M/Y". Public: 153 questions had no
    # hint). A 2-digit year must be near today, so 5/12 (a day/month) is left alone. Quý 2/2026 is a quarter.
    for mt in re.finditer(r"(?<![\d/.-])(\d{1,2}) ?[/-] ?(20\d\d|\d\d)(?![\d/.-])", question):
        if re.search(r"(?i)(?:^|\W)(?:qu[yý]|q|th[aá]ng|thg|th|t)\s*$", question[:mt.start()]):
            continue
        m, y = mt.groups()
        m, yy = int(m), int(y)
        if 1 <= m <= 12 and (len(y) == 4 or d.year - 2005 <= yy <= d.year - 1999):
            y = yy + (2000 if len(y) == 2 else 0)
            out.append(f"tháng {m}/{y} = [{first(y, m)}, {first(y, m + 1)})")
    for qn, y in re.findall(r"(?:^|_)quy_([1-4])(?:_nam|_)?_?(\d{4})?(?=_|$)", t):
        qn = int(qn); y = int(y) if y else year_of(3 * qn - 2)
        out.append(f"quý {qn}/{y} = [{first(y, 3 * qn - 2)}, {first(y, 3 * qn + 1)})")
    for y in re.findall(r"(?:^|_)nam_(\d{4})(?=_|$)", re.sub(r"(?:thang|quy)_\d{1,2}_nam_\d{4}", "", t)):
        out.append(f"năm {y} = [{y}-01-01, {int(y) + 1}-01-01)")
    if re.search(r"(?:^|_)tet(?=_|$)", t) and db is not None:
        ys = [int(y) for y in re.findall(r"tet(?:_nam)?_(\d{4})", t)] or [d.year]
        for y in ys:
            try:
                a, b = db.execute("SELECT min(date) FILTER (WHERE lunar_year = ? - 1 AND lunar_month = 12 AND lunar_day = 23 AND NOT lunar_leap), "
                                  "min(date) FILTER (WHERE lunar_year = ? AND lunar_month = 1 AND lunar_day = 10 AND NOT lunar_leap) FROM calendar", [y, y]).fetchone()
            except Exception:
                break
            if a and b:
                n = (b - a).days + 1
                out.append(f"Tết {y} = [{a}, {b + timedelta(days=1)}) ({n} days); trước Tết (same {n} days before) = [{a - timedelta(days=n)}, {a})")
    # 'N ngày trước đó / trước Tết' is a window before another day, not the last N days
    for n in re.findall(r"(?:^|_)(\d{1,3})_ngay_(?:gan_nhat|gan_day|qua|vua_qua|truoc(?!_(?:do|tet|khi|ngay|hom|dot|lan))|cuoi)", t):
        if len(re.findall(rf"(?:^|_){n}_ngay(?=_|$)", t)) > 1:  # 'N ngày kể từ hôm đó so với N ngày trước': both windows hang on another day
            continue
        n = int(n)
        out.append(f"{n} ngày gần nhất (incl. today) = [{d - timedelta(days=n - 1)}, {d + timedelta(days=1)})")
    days = []
    for dd, mm, yy in re.findall(r"(?:^|_)(\d{1,2})_(\d{1,2})_(\d{4})(?=_|$)", t):
        try:
            x = date(int(yy), int(mm), int(dd)); out.append(f"ngày {x} = [{x}, {x + timedelta(days=1)})"); days.append(x)
        except ValueError:
            pass
    # rulebook 'Hiệu ứng tăng giá': N days from D (D included) against the N days right before D. Samples miscount these bounds
    k = re.search(r"(?:^|_)(\d{1,3})_ngay_ke_tu(?=_|$)", t)
    if k and len(set(days)) == 1:
        n, x = int(k.group(1)), days[0]
        out.append(f"{n} ngày kể từ {x} (incl. {x}) = [{x}, {x + timedelta(days=n)})")
        if re.search(r"(?:^|_)truoc(?=_|$)", t[k.end():]):
            out.append(f"{n} ngày ngay trước {x} = [{x - timedelta(days=n)}, {x})")
    return list(dict.fromkeys(out))


COUNT_DAYS = re.compile(r"(?:^|_)(?:may|bao_nhieu|bn|bnhiu|bao_nheiu|so)_(?:ngay|bua|hom)(?:_|$)")
THRESHOLD = re.compile(r"tro_l|tro_xuong|(?:^|_)(?:tren|duoi|it_nhat|vuot|qua)_\d|≥|>=")


PRICE_UP = re.compile(r"(?:^|_)(?:len_gia|tang_gia|doi_gia|len_\d)")
N_DAYS = re.compile(r"(?:^|_)\d{1,3}_ngay")
DAY_MONTH = re.compile(r"(?:^|_)\d{1,2}_\d{1,2}(?:_\d{4})?(?=_|$)")
PROFIT = re.compile(r"(?:^|_)(?:lai_gop|loi_gop|lai_rong|loi_rong|loi_that|lai_that|loi_nhuan|tru_chi_phi)(?:_|$)")
TRIPS = re.compile(r"(?:^|_)(?:dot|luot|chuyen)_(?:\w+_)?(?:giao|hang)(?:_|$)|giao_(?:\w+_){0,3}(?:may|bao_nhieu|bn|bnhiu|bao_nhiu)_(?:dot|luot|chuyen)(?:_|$)")
LOT = re.compile(r"het_han(?:_|$)|han_dung(?!_(?:hom|bua|ngay|ky))|han_su_dung|(?:^|_)hsd(?:_|$)|(?:^|_)lo_(?:hang|ton|thuoc|gan|sap|con|nao|co)(?:_|$)")  # 'hạn đúng hôm nay' = 'hạn dùng' after norm


def lot_hint(question, meta):
    """Lots sell out: every 'lô còn tồn / gần hết hạn' sample took MIN(expiry) over all lots of the item (holdout biz_123), so name the tables keyed by lot."""
    if not LOT.search(norm(question)):
        return []
    tabs = [f"{quote(t['name'])}({', '.join(c for c, *_ in t['columns'] if c != '_sheet')})" for t in meta["tables"]
            if any(re.search(r"(?:^|_)(?:lo|lot|batch|so_lo)$", ckey(c)) for c, *_ in t["columns"])]
    if len(tabs) < 2:
        return []
    return ["LOT STOCK (rulebook: only lots still in stock count): tables keyed by lot: " + "; ".join(tabs) + ". A lot's stock at day D = its received base quantity "
            "minus SUM(quantity taken - quantity returned) of the rows linking sale lines to that lot, on orders up to D. Compute it per lot in a CTE, summing the "
            "links in their own subquery GROUP BY lot before joining them to the lot (joining receipts and links first multiplies rows), and keep "
            "stock > 0 (and expiry >= today for 'not expired') BEFORE taking MIN(expiry), counting items or summing lot value (stock x the lot's own purchase price per base unit)."]


def grain_hints(question):
    """Question shapes the samples misread: 'mấy ngày bán được từ X trở lên' counts days by their daily total (public samples
    counted orders >= X). 'từ khi X lên giá P, N ngày đầu ...' names no date, so D has to come from the price table
    (public: all 3 such questions abstained, samples guessed D differently)."""
    t, out = norm(question), []
    if COUNT_DAYS.search(t) and THRESHOLD.search(t + question):
        out.append("GRAIN: count DAYS, not orders: total the amount per day (GROUP BY the date), then count the days whose daily total meets the condition.")
    if PRICE_UP.search(t) and N_DAYS.search(t) and not DAY_MONTH.search(t):
        out.append("PRICE INCREASE DATE D is not given: in a CTE take D from the price table = the first date the item's price became the new price, "
                   "higher than its price right before; then compare quantity sold in [D, D + N days) with [D - N days, D).")
    if PROFIT.search(t):  # public samples wrote SUM(cost) * SUM(qty) or subtracted the sales amount twice
        out.append("PROFIT (rulebook): gross profit = revenue of valid sale lines - SUM(quantity in the SALE unit * cost recorded on the sale line), line by line: "
                   "that cost is per sale unit (the unit of the unit price), so never multiply it by a base-unit quantity (qty of v_sales, 'SL cơ bản'); "
                   "net profit / lời thật of a month = its gross profit - SUM of every expense row dated in that month.")
    if TRIPS.search(t):
        out.append("DELIVERY TRIPS (rulebook): each trip (đợt, 'Lượt giao') of each order counts once: COUNT(DISTINCT (order id, trip number)) over the delivery log rows dated in the period. The log has one row per item per trip, so not COUNT(*), and not COUNT(DISTINCT trip number).")
    return out


def data_span(meta):
    """Date span of the shop's biggest table that has a transaction-date column (not expiry/birth dates)."""
    best = (0, None, None)
    for t in meta["tables"]:
        ds = [v for col, typ, d, v in t["columns"] if d is None and typ.startswith(("DATE", "TIMESTAMP")) and v[0] != "None"
              and re.search(r"ngay|date|thoi_gian|time", ckey(col)) and not re.search(r"(^|_)(han|han_dung|het_han|hsd|nsx|sinh|san_xuat|expir\w*)(_|$)", ckey(col))]
        if ds and t["name"] != "calendar" and t["rows"] > best[0]:
            best = (t["rows"], min(v[0][:10] for v in ds), max(v[1][:10] for v in ds))
    return best[1:]


def outside(periods, span):
    """Rulebook 7: every named period lies completely outside the shop's data dates -> unanswerable, not 0.
    Train: fires on 14/3000 questions, all 14 gold unanswerable."""
    w = [m for p in periods for m in re.findall(r"\[(\d{4}-\d\d-\d\d), (\d{4}-\d\d-\d\d)\)", p)[:1]]
    lo, hi = span
    return bool(w and lo) and all(e <= lo or s > hi for s, e in w)


DEBT_WORDS = re.compile(r"(?:^|_)(?:no|nowj|cong_no|mua_chiu|ghi_no)(?:_|$)")  # nợ (also its telex typo), công nợ, mua chịu, ghi nợ, on norm(question)


def question_hints(question, as_of, db, meta):
    """Every hint line placed after the date anchors of one question."""
    return period_hints(question, as_of, db) + grain_hints(question) + lot_hint(question, meta)
