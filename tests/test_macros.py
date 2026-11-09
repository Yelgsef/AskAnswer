"""Synthetic tests for optional dynamic business macros."""
import json
from types import SimpleNamespace

import duckdb

from datamate import macros
from datamate.ingest import profile
from datamate.macro_registry import MacroRegistry, safe_name
from datamate.macro_validator import validate_candidate
from datamate.pipeline import Pipeline, Shop
from datamate.predict import parse_args


def make_db(duplicate=False):
    db = duckdb.connect()
    db.execute('CREATE TABLE "sales" ("order_id" VARCHAR, "amount" DOUBLE, "status" VARCHAR)')
    rows = [("A", 10.0, "done"), ("B", 20.0, "done")]
    if duplicate:
        rows.append(("A", 5.0, "done"))
    db.executemany('INSERT INTO "sales" VALUES (?, ?, ?)', rows)
    meta = {"tables": [profile(db, "sales", "sales.csv")], "readme": ""}
    return db, meta


def plan(invariants=None):
    return {"name": "sales totals", "description": "Reusable sales values", "parameters": [],
            "returns": [{"name": "entity_id", "type": "VARCHAR"}, {"name": "value", "type": "DOUBLE"}],
            "required_tables": ["sales"], "required_columns": ["sales.order_id", "sales.amount"],
            "rulebook_evidence": ["sales amount is recorded per row"], "question_indices": [0],
            "invariants": invariants or []}


def candidate(body, invariants=None):
    return {"macro_name": "sales totals", "description": "Reusable sales values", "parameters": [],
            "output_columns": ["entity_id", "value"], "sql_body": body, "source_tables": ["sales"],
            "invariants": invariants or []}


def test_plan_rejects_unknown_schema_and_accepts_empty_plan():
    _, meta = make_db()
    valid, errors = macros.validate_plan({"macros": [dict(plan(), required_tables=["missing"])]}, meta, 1)
    assert not valid and errors
    assert macros.validate_plan({"macros": []}, meta, 1) == ([], [])


def test_invalid_planner_json_is_a_fallback_state():
    class Engine:
        def params(self, **kwargs):
            return SimpleNamespace(**kwargs)

        def chat(self, chats, params, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(text="not json")])]

    db, meta = make_db()
    args = SimpleNamespace(macro_max_context_chars=2000, macro_max_count=6, macro_repairs=0, k=1, view_think=0, seed=0)
    registry, artifacts = macros.run(db, meta, "", "rules", "", [{"question": "total", "as_of_date": "2026-01-01"}], Engine(), args)
    assert registry.entries == []
    assert artifacts["plan_errors"]


def test_generator_failure_keeps_existing_pipeline_fallback():
    db, meta = make_db()

    class Engine:
        def __init__(self):
            self.calls = 0

        def params(self, **kwargs):
            return SimpleNamespace(**kwargs)

        def chat(self, chats, params, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps({"macros": [plan()]}))])]
            raise RuntimeError("generator unavailable")

    args = SimpleNamespace(macro_max_context_chars=2000, macro_max_count=6, macro_repairs=0, k=1, view_think=0, seed=0)
    registry, artifacts = macros.run(db, meta, "", "rules", "", [{"question": "total", "as_of_date": "2026-01-01"}], Engine(), args)
    assert registry.entries == []
    assert artifacts["candidates"][0]["error"] == "generator unavailable"


def test_validator_rejects_unsafe_and_unknown_sql():
    db, _ = make_db()
    p = plan()
    assert validate_candidate(db, candidate("SELECT * FROM read_csv_auto('secret.csv')"), p)["status"] == "REJECTED"
    assert validate_candidate(db, candidate('SELECT "missing" AS entity_id, 1.0 AS value FROM "sales"'), p)["status"] == "REJECTED"


def test_validator_rejects_wrong_output_types_and_unverified_candidates():
    db, _ = make_db()
    p = plan()
    wrong = candidate('SELECT "order_id" AS entity_id, "status" AS value FROM "sales"')
    assert validate_candidate(db, wrong, p)["status"] == "REJECTED"
    unverified = validate_candidate(db, candidate('SELECT "order_id" AS entity_id, "amount" AS value FROM "sales"'), p)
    assert unverified["status"] == "STRUCTURALLY_VALID"


def test_semantic_invariants_gate_registration():
    db, _ = make_db()
    p = plan([{"kind": "unique", "columns": ["entity_id"]}, {"kind": "nonnegative", "columns": ["value"]}])
    checked = validate_candidate(db, candidate('SELECT "order_id" AS entity_id, "amount" AS value FROM "sales"', p["invariants"]), p)
    assert checked["status"] == "SEMANTICALLY_VERIFIED"
    registry = MacroRegistry()
    entry = registry.register(db, candidate('SELECT "order_id" AS entity_id, "amount" AS value FROM "sales"', p["invariants"]), p, checked)
    assert entry["validation_status"] == "SEMANTICALLY_VERIFIED"
    assert entry["rulebook_evidence"]
    assert registry.context(0).startswith("Only call verified")


def test_failed_independent_invariant_stays_unverified():
    db, _ = make_db(duplicate=True)
    p = plan([{"kind": "unique", "columns": ["entity_id"]}])
    checked = validate_candidate(db, candidate('SELECT "order_id" AS entity_id, "amount" AS value FROM "sales"', p["invariants"]), p)
    assert checked["status"] == "STRUCTURALLY_VALID"


def test_registry_names_are_deterministic_isolated_and_deduplicated():
    db1, _ = make_db()
    db2, _ = make_db()
    p = plan([{"kind": "unique", "columns": ["entity_id"]}])
    c = candidate('SELECT "order_id" AS entity_id, "amount" AS value FROM "sales"', p["invariants"])
    checked = validate_candidate(db1, c, p)
    r1, r2 = MacroRegistry(), MacroRegistry()
    first = r1.register(db1, c, p, checked)
    second = r1.register(db1, c, p, checked)
    other = r2.register(db2, c, p, checked)
    assert first["registered_name"] == other["registered_name"] == safe_name("sales totals")
    assert second is None and r1.rejections
    assert r2.entries != r1.rejections


def test_max_output_is_rejected_before_registration():
    db, _ = make_db()
    p = plan()
    checked = validate_candidate(db, candidate('SELECT "order_id" AS entity_id, "amount" AS value FROM "sales"'), p, max_rows=1)
    assert checked["status"] == "REJECTED"


def test_context_accepts_incomplete_readme_without_reference_answers():
    _, meta = make_db()
    text = macros.context("", "rules", meta, "", [{"question": "total", "as_of_date": "2026-01-01"}], 1000)
    assert "RULEBOOK:" in text and "QUESTIONS FOR THIS BUSINESS:" in text
    assert "answer" not in text.lower()


def test_pipeline_exposes_only_registered_macros_to_question_prompts():
    db, meta = make_db()
    p = plan([{"kind": "unique", "columns": ["entity_id"]}])
    c = candidate('SELECT "order_id" AS entity_id, "amount" AS value FROM "sales"', p["invariants"])
    checked = validate_candidate(db, c, p)
    registry = MacroRegistry()
    registry.register(db, c, p, checked)
    pipe = Pipeline.__new__(Pipeline)
    pipe.a, pipe.bank, pipe.systems = SimpleNamespace(fewshot_k=2), None, ["SYSTEM"]
    shop = Shop("shop", db, meta, meta, macro_registry=registry, ctx="BUSINESS")
    chats = pipe.build_chats(shop, [{"question": "total", "as_of_date": "2026-01-01"}])
    assert registry.entries[0]["registered_name"] in chats[0][0]["user"]


def test_dynamic_macro_flags_are_disabled_by_default():
    args = parse_args(["--data", "data", "--output", "out", "--model", "model"])
    assert not args.dynamic_macros
    assert (args.macro_max_count, args.macro_repairs, args.macro_max_context_chars) == (6, 2, 24000)
