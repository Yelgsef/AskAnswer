"""The submission writer and checker."""
import json

from datamate.submission import check, problem, write_submission


def test_written_file_passes_the_check(tmp_path):
    qs = [{"id": "q_1"}, {"id": "q_2"}]
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n", encoding="utf-8")
    write_submission(tmp_path / "submission.jsonl", qs, {"q_1": {"type": "number", "answer": 3}})
    rows = [json.loads(l) for l in (tmp_path / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "q_1", "type": "number", "answer": 3}, {"id": "q_2", "type": "abstain"}]
    assert check(tmp_path / "submission.jsonl", tmp_path / "questions.jsonl") == []


def test_problems():
    assert problem({"type": "date", "answer": "15/08/2026"})
    assert problem({"type": "number", "answer": "5"})
    assert problem({"type": "table", "answer": {"columns": ["a"], "rows": [[1, 2]]}})
    assert problem({"type": "list", "answer": ["a", 2]}) is None
