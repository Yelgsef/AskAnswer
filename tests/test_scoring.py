import pytest

from datamate.scoring import penalized_accuracy, score

STR = {"type": "string", "answer": "Sting dâu 330ml", "alias": ["SP0042"]}
LIST = {"type": "list", "answer": ["Cà phê sữa (M)", "Bạc xỉu"], "alias": [["SP0003"], ["SP0017"]]}
TABLE = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"],
                                     "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]},
         "alias": [["KH01"], ["KH02"]]}


@pytest.mark.parametrize("pred, ref, want", [
    (None, STR, 0),
    ({"type": "abstain"}, STR, 0),
    ({"type": "string", "answer": "  STING  dâu-330ML! "}, STR, 1),
    ({"type": "string", "answer": "sp0042"}, STR, 1),
    ({"type": "string", "answer": "Sting dầu 330ml"}, STR, -1),        # wrong diacritic
    ({"type": "list", "answer": ["Sting dâu 330ml"]}, STR, -1),        # wrong type
    ({"type": "number", "answer": 4820000}, {"type": "number", "answer": 4820000}, 1),
    ({"type": "number", "answer": 4820000.0000001}, {"type": "number", "answer": 4820000}, -1),  # numbers match exactly (Evaluation tab)
    ({"type": "number", "answer": "4820000"}, {"type": "number", "answer": 4820000}, -1),
    ({"type": "number", "answer": 4820001}, {"type": "number", "answer": 4820000}, -1),
    ({"type": "percent", "answer": -12.449}, {"type": "percent", "answer": -12.45}, 1),
    ({"type": "percent", "answer": 12.45}, {"type": "percent", "answer": -12.45}, -1),
    ({"type": "percent", "answer": 0.1245}, {"type": "percent", "answer": 12.45}, -1),
    ({"type": "date", "answer": "2025-09"}, {"type": "date", "answer": "2025-09"}, 1),
    ({"type": "date", "answer": "15/08/2026"}, {"type": "date", "answer": "2026-08-15"}, -1),
    ({"type": "boolean", "answer": True}, {"type": "boolean", "answer": True}, 1),
    ({"type": "boolean", "answer": 1}, {"type": "boolean", "answer": True}, -1),
    ({"type": "list", "answer": ["sp0003", "bạc xỉu"]}, LIST, 1),
    ({"type": "list", "answer": ["Bạc xỉu", "Cà phê sữa (M)"]}, LIST, -1),  # order matters
    ({"type": "list", "answer": [3, 10]}, {"type": "list", "answer": [3, 10], "alias": []}, 1),
    ({"type": "table", "answer": {"columns": ["so_tien_no", "khach_hang"],
                                  "rows": [[1500000, "anh tú"], [3200000, "KH01"]]}}, TABLE, 1),
    ({"type": "table", "answer": {"columns": ["ten", "no"],
                                  "rows": [["Anh Tú", 1500000], ["Chị Lan", 3200000]]}}, TABLE, 1),  # by position
    ({"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"],
                                  "rows": [["Chị Lan", 3200000]]}}, TABLE, -1),
    ({"type": "unanswerable", "reason": "x"}, {"type": "unanswerable", "answer": None}, 1),
    ({"type": "number", "answer": 0}, {"type": "unanswerable", "answer": None}, -1),
    ({"type": "unanswerable"}, {"type": "number", "answer": 5}, -1),
])
def test_score(pred, ref, want):
    assert score(pred, ref) == want


def test_gold_scores_perfect_on_train(train_answers):
    gold = {r["id"]: r for r in train_answers}
    res = penalized_accuracy(gold, gold)
    assert res["PA"] == 1.0 and res["n"] == 3000


def test_missing_is_abstain():
    gold = {"a": {"type": "number", "answer": 1}, "b": {"type": "number", "answer": 2}}
    res = penalized_accuracy({"a": {"type": "number", "answer": 1}}, gold)
    assert (res["PA"], res["abstain"]) == (0.5, 1)
