"""View-draft prompt integration for representative rows."""
from types import SimpleNamespace

import duckdb

from datamate.ingest import profile, schema_text
from datamate.pipeline import Pipeline
from datamate.predict import parse_args


class FakeEngine:
    def __init__(self):
        self.chats = []

    def params(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def chat(self, chats, params, **kwargs):
        self.chats.append(chats)
        return [SimpleNamespace(outputs=[SimpleNamespace(text="```sql\nSELECT 1\n```") for _ in range(params.n)]) for _ in chats]


def loaded_shop():
    db = duckdb.connect()
    db.execute('CREATE TABLE "sales" ("order_id" VARCHAR, "product_name" VARCHAR, "amount" INTEGER)')
    db.execute("INSERT INTO \"sales\" VALUES ('A', 'Trà đào', 1)")
    meta = {"tables": [profile(db, "sales", "sales.csv")], "readme": "business notes"}
    return db, meta


def run_draft(monkeypatch, sample_rows):
    db, meta = loaded_shop()
    engine = FakeEngine()
    monkeypatch.setattr("datamate.pipeline.views.check", lambda db, texts: [("same", "") for _ in texts])
    args = SimpleNamespace(views="llm", view_sample_rows=sample_rows, k=1, view_think=0, temperature=0.1, seed=0)
    pipe = Pipeline.__new__(Pipeline)
    pipe.a, pipe.engine, pipe.rulebook = args, engine, "rules"
    pipe.vfixed, pipe.vtexts, pipe.vchecked = {}, {}, {}
    pipe.draft_views({"shop": (db, meta)}, 0)
    return engine.chats[0][0][1]["content"], meta


def test_disabled_sampling_preserves_original_view_prompt(monkeypatch):
    monkeypatch.setattr("datamate.pipeline.sample_text", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")))
    prompt, meta = run_draft(monkeypatch, 0)

    assert prompt == "README:\nbusiness notes\n\nSCHEMA:\n" + schema_text(meta)
    assert "REPRESENTATIVE ROWS:" not in prompt


def test_enabled_sampling_adds_rows_to_view_prompt(monkeypatch):
    prompt, _ = run_draft(monkeypatch, 10)

    assert "REPRESENTATIVE ROWS:" in prompt
    assert '"Trà đào"' in prompt


def test_view_sample_rows_flag_defaults_off_and_accepts_positive_value():
    base = ["--data", "data", "--output", "out", "--model", "model"]
    assert parse_args(base).view_sample_rows == 0
    assert parse_args(base + ["--view-sample-rows", "7"]).view_sample_rows == 7
