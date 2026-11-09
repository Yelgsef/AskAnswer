"""Optional per-business planning, generation and validation of reusable SQL macros."""
import json
import re
import time

from .ingest import schema_text
from .macro_registry import MacroRegistry
from .macro_validator import PARAM_TYPES, validate_candidate


PLAN_SYSTEM = """You are a conservative business-logic planner for one unseen business. Read the RULEBOOK, README, SCHEMA, representative rows and the numbered questions. Identify only reusable computations that genuinely help multiple questions or one complex repeated operation. Do not invent missing fields. Canonical views and existing debt macros are preferred when they contain the evidence. An empty plan is valid.
Return only JSON: {"macros":[{"name":"descriptive_operation_name","description":"...","parameters":[{"name":"start_date","type":"DATE"}],"returns":[{"name":"entity_id","type":"VARCHAR"},{"name":"value","type":"DOUBLE"}],"required_tables":["table"],"required_columns":["table.column"],"rulebook_evidence":["..."],"question_indices":[0],"priority":1,"invariants":[{"kind":"unique","columns":["entity_id"]}]}]}. Supported parameter types are DATE, TIMESTAMP, VARCHAR, DOUBLE, FLOAT, INTEGER, BIGINT, BOOLEAN and DECIMAL. Supported independently checkable invariants are unique, nonnegative and not_null. Do not include reference answers or question IDs."""

GENERATOR_SYSTEM = """You generate one reusable DuckDB table macro for an already approved plan. Use only the supplied rulebook, schema, canonical views, representative rows and questions. Return only one JSON object with macro_name, description, parameters, output_columns, sql_body, source_tables and invariants. sql_body must be exactly one read-only SELECT/WITH body, use named macro parameters, preserve date boundaries and aggregation grain, and never use DDL, DML, file/network access or another dynamic macro. The body must work over the entire business dataset, not only sampled values."""


def _clip(text, limit):
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "\n[context truncated]"


def _json_object(text):
    text = str(text or "").split("</think>")[-1].strip()
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    text = blocks[-1].strip() if blocks else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value


def _bounded_candidate(candidate):
    if not isinstance(candidate, dict):
        return None
    out = dict(candidate)
    out["description"] = _clip(out.get("description"), 1000)
    out["sql_body"] = _clip(out.get("sql_body"), 12000)
    for key in ("parameters", "output_columns", "source_tables", "invariants"):
        if isinstance(out.get(key), list):
            out[key] = out[key][:32]
    return out


def _parameters(value):
    if not isinstance(value, list):
        raise ValueError("parameters must be a list")
    out = []
    for p in value:
        if not isinstance(p, dict) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(p.get("name", ""))):
            raise ValueError("parameter names must be SQL identifiers")
        typ = str(p.get("type", "")).upper()
        if typ not in PARAM_TYPES:
            raise ValueError(f"unsupported parameter type {typ}")
        out.append({"name": p["name"], "type": typ})
    if len({p["name"].lower() for p in out}) != len(out):
        raise ValueError("duplicate parameters")
    return out


def _returns(value):
    if not isinstance(value, list) or not value:
        raise ValueError("returns must be a non-empty list")
    out = []
    for r in value:
        if not isinstance(r, dict) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(r.get("name", ""))):
            raise ValueError("return names must be SQL identifiers")
        typ = str(r.get("type", "")).upper()
        if not typ:
            raise ValueError("return type is required")
        out.append({"name": r["name"], "type": typ})
    if len({r["name"].lower() for r in out}) != len(out):
        raise ValueError("duplicate returns")
    return out


def validate_plan(plan, meta, question_count, max_count=6):
    if not isinstance(plan, dict) or not isinstance(plan.get("macros"), list):
        return [], ["planner output must be an object with a macros list"]
    tables = {str(t.get("name", "")).lower(): t for t in meta.get("tables", ())}
    if len(plan["macros"]) > max_count:
        return [], [f"plan exceeds macro limit {max_count}"]
    valid, errors, names = [], [], set()
    for i, raw in enumerate(plan["macros"]):
        try:
            if not isinstance(raw, dict) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]{0,79}", str(raw.get("name", ""))):
                raise ValueError("name must be a descriptive identifier")
            name = str(raw["name"]).strip()
            if name.lower() in names:
                raise ValueError("duplicate macro name")
            names.add(name.lower())
            if not str(raw.get("description", "")).strip():
                raise ValueError("description is required")
            if not isinstance(raw.get("rulebook_evidence"), list) or not raw["rulebook_evidence"]:
                raise ValueError("rulebook_evidence is required")
            parameters = _parameters(raw.get("parameters", []))
            returns = _returns(raw.get("returns"))
            required_tables = raw.get("required_tables", [])
            if not isinstance(required_tables, list) or any(str(t).lower() not in tables for t in required_tables):
                raise ValueError("required_tables contains an unknown table")
            required_columns = raw.get("required_columns", [])
            if not isinstance(required_columns, list):
                raise ValueError("required_columns must be a list")
            for ref in required_columns:
                table, dot, column = str(ref).partition(".")
                if not dot or table.lower() not in tables or str(column).lower() not in {str(c[0]).lower() for c in tables[table.lower()].get("columns", ())}:
                    raise ValueError(f"unknown required column {ref}")
            indices = raw.get("question_indices", [])
            if not isinstance(indices, list) or any(not isinstance(n, int) or n < 0 or n >= question_count for n in indices):
                raise ValueError("question_indices must point into this business question list")
            invariants = raw.get("invariants", [])
            if not isinstance(invariants, list) or any(not isinstance(x, dict) or x.get("kind") not in {"unique", "nonnegative", "not_null"} for x in invariants):
                raise ValueError("unsupported invariant")
            valid.append({**raw, "name": name, "description": _clip(raw.get("description"), 1000),
                          "rulebook_evidence": [_clip(x, 500) for x in raw.get("rulebook_evidence", [])[:8]],
                          "parameters": parameters, "returns": returns,
                          "required_tables": list(dict.fromkeys(required_tables)), "required_columns": required_columns,
                          "question_indices": sorted(set(indices)), "invariants": invariants})
        except Exception as e:
            errors.append(f"macro {i}: {str(e)[:300]}")
    return valid, errors


def context(readme, rulebook, meta, samples, questions, max_chars, available_layers=""):
    qtext = "\n".join(f"[{i}] as_of_date={q.get('as_of_date')}: {q.get('question', '')}" for i, q in enumerate(questions))
    text = ("RULEBOOK:\n" + str(rulebook or "") + "\n\nREADME:\n" + str(readme or "") +
            "\n\nSCHEMA AND CANONICAL VIEWS:\n" + schema_text(meta) +
            ("\n\nAVAILABLE EXISTING LAYERS:\n" + available_layers if available_layers else "") +
            ("\n\nREPRESENTATIVE ROWS:\n" + samples if samples else "") +
            "\n\nQUESTIONS FOR THIS BUSINESS:\n" + qtext)
    return _clip(text, max_chars)


def _chat_json(engine, system, user, args, *, n=1):
    params = engine.params(n=n, temperature=0.2, top_p=0.95, top_k=20,
                           max_tokens=max(2048, int(getattr(args, "view_think", 0)) + 2048), seed=getattr(args, "seed", 0))
    outputs = engine.chat([[{"role": "system", "content": system}, {"role": "user", "content": user}]], params,
                          thinking=bool(getattr(args, "view_think", 0)))
    return [o.text for o in outputs[0].outputs]


def run(db, meta, readme, rulebook, samples, questions, engine, args, available_layers=""):
    """Run the optional stage once for one business and return registry plus bounded artifacts."""
    started = time.time()
    registry = MacroRegistry()
    context_text = context(readme, rulebook, meta, samples, questions, getattr(args, "macro_max_context_chars", 24000), available_layers)
    artifacts = {"enabled": True, "context_chars": len(context_text), "plan": None, "plan_raw": None,
                 "plan_errors": [], "candidates": [], "validation": [], "registry": None}
    try:
        plan_raw = _chat_json(engine, PLAN_SYSTEM, context_text, args)[0]
        artifacts["plan_raw"] = _clip(plan_raw, 12000)
        parsed = _json_object(plan_raw)
        plans, errors = validate_plan(parsed, meta, len(questions), getattr(args, "macro_max_count", 6))
        artifacts["plan"] = {"macros": plans}
        artifacts["plan_errors"] = errors
    except Exception as e:
        artifacts["plan_errors"] = [f"invalid planner output: {str(e)[:400]}"]
        artifacts["seconds"] = round(time.time() - started, 3)
        artifacts["registry"] = registry.as_dict()
        return registry, artifacts
    candidate_n = min(4, max(1, int(getattr(args, "k", 1))))
    for plan in plans:
        user = context_text + "\n\nAPPROVED PLAN:\n" + json.dumps(plan, ensure_ascii=False) + "\n\nReturn one candidate JSON object."
        try:
            raw_candidates = _chat_json(engine, GENERATOR_SYSTEM, user, args, n=candidate_n)
        except Exception as e:
            artifacts["candidates"].append({"macro_name": plan["name"], "error": str(e)[:400]})
            continue
        for raw in raw_candidates:
            record = {"macro_name": plan["name"], "raw": _clip(raw, 12000), "candidate": None, "validation": None, "attempt": 0}
            try:
                candidate = _json_object(raw)
                record["candidate"] = _bounded_candidate(candidate)
                validation = validate_candidate(db, candidate, plan)
            except Exception as e:
                validation = {"status": "REJECTED", "error": f"invalid candidate JSON: {str(e)[:300]}", "evidence": []}
            record["validation"] = validation
            artifacts["candidates"].append(record)
            for attempt in range(1, max(0, int(getattr(args, "macro_repairs", 2))) + 1):
                if validation.get("status") != "REJECTED":
                    break
                repair_user = user + "\n\nPREVIOUS CANDIDATE:\n" + _clip(raw, 8000) + "\n\nVALIDATION ERROR:\n" + _clip(validation.get("error"), 500) + "\nReturn a corrected JSON object only."
                try:
                    raw = _chat_json(engine, GENERATOR_SYSTEM, repair_user, args)[0]
                    candidate = _json_object(raw)
                    validation = validate_candidate(db, candidate, plan)
                    record.update(raw=_clip(raw, 12000), candidate=_bounded_candidate(candidate), validation=validation, attempt=attempt)
                except Exception as e:
                    validation = {"status": "REJECTED", "error": f"repair failed: {str(e)[:300]}", "evidence": []}
                    record.update(raw=_clip(raw, 12000), validation=validation, attempt=attempt)
            artifacts["validation"].append({"macro_name": plan["name"], "attempt": record["attempt"], **record["validation"]})
            if record["validation"].get("status") == "SEMANTICALLY_VERIFIED" and record.get("candidate"):
                try:
                    registry.register(db, candidate, plan, record["validation"])
                except Exception as e:
                    registry.rejections.append({"name": plan["name"], "reason": f"registration failed: {str(e)[:300]}"})
    artifacts["registry"] = registry.as_dict()
    artifacts["seconds"] = round(time.time() - started, 3)
    return registry, artifacts
