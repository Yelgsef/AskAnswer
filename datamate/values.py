"""Filter values a query compares against that the shop's tables do not hold (used by analysis/flags.py)."""
import re
from difflib import get_close_matches

from .ingest import norm, quote


def known_values(db, max_distinct=50000):
    """norm(value) -> value for every text value of the shop, canonical tables included."""
    known = {}
    for tab, col in db.execute("SELECT table_name, column_name FROM information_schema.columns WHERE data_type = 'VARCHAR' AND table_name <> 'calendar' "
                               "AND NOT starts_with(table_name, '_')").fetchall():
        for (v,) in db.execute(f"SELECT DISTINCT {quote(col)} FROM {quote(tab)} WHERE {quote(col)} IS NOT NULL LIMIT {max_distinct}").fetchall():
            known.setdefault(norm(v), str(v))
    return known


def missing_filters(sql, known):
    """Equality filter values that no column of the shop holds, with the closest spelling that exists."""
    out = []
    for lit in dict.fromkeys(re.findall(r"(?<![<>!])=\s*'([^']+)'", sql)):
        k = norm(lit)
        if k and not k.isdigit() and k not in known:
            near = get_close_matches(k, known, n=1, cutoff=0.75)
            out.append((lit, known[near[0]] if near else None))
    return out
