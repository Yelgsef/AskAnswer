"""Safety, structural and limited semantic validation for generated macros."""
import re
import threading
import hashlib

from .ingest import quote


PARAM_TYPES = {"DATE", "TIMESTAMP", "VARCHAR", "DOUBLE", "FLOAT", "INTEGER", "BIGINT", "BOOLEAN", "DECIMAL"}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
UNSAFE = re.compile(r"(?i)\b(insert|update|delete|merge|create|replace|drop|alter|copy|attach|detach|install|load|pragma|export|import|call|execute|read_csv|read_parquet|parquet_scan|httpfs|sqlite_scan|postgres_scan|mysql_scan|glob|secret)\b")
TABLE_REF = re.compile(r"(?is)\b(?:from|join)\s+(\"(?:\"\"|[^\"])+\"|[A-Za-z_][A-Za-z0-9_]*)")


def _error(message):
    return {"status": "REJECTED", "error": str(message)[:500], "evidence": []}


def _type_family(typ):
    t = str(typ or "").upper()
    if t.startswith(("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "DECIMAL", "NUMERIC", "REAL", "DOUBLE", "FLOAT")):
        return "NUMERIC"
    if t.startswith("TIMESTAMP"):
        return "TIMESTAMP"
    if t.startswith("DATE"):
        return "DATE"
    if t.startswith(("VARCHAR", "CHAR", "TEXT", "STRING")):
        return "VARCHAR"
    if t.startswith("BOOLEAN"):
        return "BOOLEAN"
    return t


def _same_type(actual, expected):
    a, e = _type_family(actual), _type_family(expected)
    return a == e or a == "NUMERIC" and e == "NUMERIC"


def _literal(parameter):
    typ = parameter["type"]
    if typ == "DATE":
        return "DATE '2026-01-15'"
    if typ == "TIMESTAMP":
        return "TIMESTAMP '2026-01-15 12:00:00'"
    if typ == "VARCHAR":
        return "'macro_sample'"
    if typ == "BOOLEAN":
        return "false"
    if typ in ("INTEGER", "BIGINT"):
        return "1"
    return "1.0"


def _timed(db, cur, sql, timeout):
    timer = threading.Timer(timeout, db.interrupt)
    timer.daemon = True
    timer.start()
    try:
        return cur.execute(sql)
    finally:
        timer.cancel()


def _normalize_parameters(parameters):
    if not isinstance(parameters, list):
        return None, "parameters must be a list"
    out = []
    for p in parameters:
        if not isinstance(p, dict) or not IDENTIFIER.fullmatch(str(p.get("name", ""))):
            return None, "parameter names must be simple SQL identifiers"
        typ = str(p.get("type", "")).upper()
        if typ not in PARAM_TYPES:
            return None, f"unsupported parameter type: {typ}"
        out.append({"name": p["name"], "type": typ})
    if len({p["name"].lower() for p in out}) != len(out):
        return None, "duplicate parameter names"
    return out, None


def _normalize_returns(returns):
    if not isinstance(returns, list) or not returns:
        return None, "returns must be a non-empty list"
    out = []
    for r in returns:
        if not isinstance(r, dict) or not IDENTIFIER.fullmatch(str(r.get("name", ""))):
            return None, "return names must be simple SQL identifiers"
        typ = str(r.get("type", "")).upper()
        if not typ:
            return None, "every return needs a type"
        out.append({"name": r["name"], "type": typ})
    if len({r["name"].lower() for r in out}) != len(out):
        return None, "duplicate return names"
    return out, None


def _macro_call(name, parameters):
    return f"{quote(name)}({', '.join(_literal(p) for p in parameters)})"


def _table_references(sql):
    refs = []
    for value in TABLE_REF.findall(sql):
        if value.startswith('"'):
            refs.append(value[1:-1].replace('""', '"'))
        else:
            refs.append(value)
    ctes = {x.lower() for x in re.findall(r"(?is)(?:\bwith\b|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(", sql)}
    return [x for x in refs if x.lower() not in ctes]


def _invariant_checks(db, cur, table, invariants, timeout):
    evidence = []
    if not invariants:
        return "STRUCTURALLY_VALID", evidence
    columns = {str(r[0]).lower() for r in cur.execute(f"DESCRIBE {quote(table)}").fetchall()}
    for inv in invariants:
        if not isinstance(inv, dict):
            return "STRUCTURALLY_VALID", evidence + ["unsupported invariant shape; retained as unverified"]
        kind = inv.get("kind")
        names = inv.get("columns", [])
        if isinstance(names, str):
            names = [names]
        if kind not in {"unique", "nonnegative", "not_null"} or not names or any(str(n).lower() not in columns for n in names):
            return "STRUCTURALLY_VALID", evidence + [f"invariant not independently checkable: {kind}"]
        qnames = ", ".join(quote(n) for n in names)
        if kind == "unique":
            sql = f"SELECT count(*) FROM (SELECT {qnames} FROM {quote(table)} GROUP BY {qnames} HAVING count(*) > 1)"
            bad = _timed(db, cur, sql, timeout).fetchone()[0]
            if bad:
                return "STRUCTURALLY_VALID", evidence + [f"unique invariant failed: {', '.join(names)}"]
            evidence.append(f"unique({', '.join(names)}) passed")
        elif kind == "nonnegative":
            bad = _timed(db, cur, f"SELECT count(*) FROM {quote(table)} WHERE {' OR '.join(f'{quote(n)} < 0' for n in names)}", timeout).fetchone()[0]
            if bad:
                return "STRUCTURALLY_VALID", evidence + [f"nonnegative invariant failed: {', '.join(names)}"]
            evidence.append(f"nonnegative({', '.join(names)}) passed")
        else:
            bad = _timed(db, cur, f"SELECT count(*) FROM {quote(table)} WHERE {' OR '.join(f'{quote(n)} IS NULL' for n in names)}", timeout).fetchone()[0]
            if bad:
                return "STRUCTURALLY_VALID", evidence + [f"not_null invariant failed: {', '.join(names)}"]
            evidence.append(f"not_null({', '.join(names)}) passed")
    return "SEMANTICALLY_VERIFIED", evidence


def validate_candidate(db, candidate, plan, timeout=20, max_rows=2000):
    """Validate one candidate in a transaction that is always rolled back."""
    if not isinstance(candidate, dict):
        return _error("candidate must be an object")
    if candidate.get("macro_name") != plan.get("name"):
        return _error("candidate macro_name does not match its plan")
    parameters, error = _normalize_parameters(candidate.get("parameters", plan.get("parameters", [])))
    if error:
        return _error(error)
    if parameters != plan.get("parameters", []):
        return _error("candidate parameters do not match the plan")
    returns, error = _normalize_returns(plan.get("returns"))
    if error:
        return _error(error)
    output_columns = candidate.get("output_columns") or [r["name"] for r in returns]
    if output_columns != [r["name"] for r in returns] or any(not IDENTIFIER.fullmatch(str(c)) for c in output_columns):
        return _error("candidate output_columns do not match the plan")
    body = str(candidate.get("sql_body", "")).strip()
    if len(body) > 12000 or not re.match(r"(?is)^(with|select)\b", body) or ";" in body or "\x00" in body:
        return _error("sql_body must be one SELECT/WITH statement")
    if "--" in body or "/*" in body or UNSAFE.search(body):
        return _error("sql_body contains a disallowed SQL construct")
    source_tables = candidate.get("source_tables", plan.get("required_tables", []))
    if not isinstance(source_tables, list) or any(not isinstance(t, str) for t in source_tables):
        return _error("source_tables must be a list")
    available_tables = {str(r[0]).lower() for r in db.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    if any(t.lower() not in available_tables for t in source_tables):
        return _error("source_tables contains an unknown table")
    refs = _table_references(body)
    if any(ref.lower() not in available_tables for ref in refs) or any(ref.lower() in {"information_schema", "sqlite_master"} or ref.lower().startswith(("duckdb_", "pg_")) for ref in refs):
        return _error("sql_body references a table outside the business schema")
    if source_tables and any(t.lower() not in {ref.lower() for ref in refs} for t in source_tables):
        return _error("sql_body does not reference every declared source table")
    parameters, _ = _normalize_parameters(parameters)
    macro_name = "__datamate_validate_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    result_a, result_b = macro_name + "_a", macro_name + "_b"
    cur = db.cursor()
    evidence = []
    try:
        cur.execute("SET enable_external_access = false")
        cur.execute("BEGIN TRANSACTION")
        cur.execute(f"CREATE MACRO {quote(macro_name)}({', '.join(p['name'] for p in parameters)}) AS TABLE {body}")
        call = _macro_call(macro_name, parameters)
        _timed(db, cur, f"CREATE TEMP TABLE {quote(result_a)} AS SELECT * FROM {call} LIMIT {max_rows + 1}", timeout)
        n = cur.execute(f"SELECT count(*) FROM {quote(result_a)}").fetchone()[0]
        if n > max_rows:
            return _error(f"macro output exceeds {max_rows} rows")
        description = cur.execute(f"DESCRIBE {quote(result_a)}").fetchall()
        names = [r[0] for r in description]
        types = [r[1] for r in description]
        if names != output_columns:
            return _error(f"returned columns {names} do not match {output_columns}")
        if any(not _same_type(actual, expected["type"]) for actual, expected in zip(types, returns)):
            return _error(f"returned types {types} do not match {[r['type'] for r in returns]}")
        rows_a = cur.execute(f"SELECT * FROM {quote(result_a)}").fetchall()
        _timed(db, cur, f"CREATE TEMP TABLE {quote(result_b)} AS SELECT * FROM {call} LIMIT {max_rows + 1}", timeout)
        n_b = cur.execute(f"SELECT count(*) FROM {quote(result_b)}").fetchone()[0]
        if n_b != n or sorted(map(repr, rows_a)) != sorted(map(repr, cur.execute(f"SELECT * FROM {quote(result_b)}").fetchall())):
            return _error("repeated execution is not deterministic")
        evidence.append(f"bounded read-only execution passed ({n} rows)")
        status, semantic_evidence = _invariant_checks(db, cur, result_a, candidate.get("invariants", plan.get("invariants", [])), timeout)
        evidence.extend(semantic_evidence)
        return {"status": status, "error": None, "evidence": evidence, "columns": names, "types": types}
    except Exception as e:
        return _error(e)
    finally:
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass
