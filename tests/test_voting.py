"""Answer keys, the vote, the answer/abstain decision and the debt-question pool."""
from datamate.voting import answer_key, debt_pool, decide, final_answer, vote

N = lambda v: {"type": "number", "answer": v}


def test_table_key_is_its_row_set():
    a = {"type": "table", "answer": {"columns": ["ten", "no"], "rows": [["An", 5], ["Binh", 3]]}}
    b = {"type": "table", "answer": {"columns": ["khach", "so_no"], "rows": [["binh", 3], ["an", 5]]}}
    assert answer_key(a) == answer_key(b)


def test_string_key_ignores_case_and_punctuation():
    assert answer_key({"type": "string", "answer": "Cà phê sữa (M)"}) == answer_key({"type": "string", "answer": "cà phê sữa m"})


def test_vote_merges_number_and_percent_of_one_value():
    ans, conf = vote([N(12.5), {"type": "percent", "answer": 12.5}, {"type": "percent", "answer": 12.5}, N(3), None], 5)
    assert ans == {"type": "percent", "answer": 12.5} and conf == 3 / 5


def test_vote_with_no_executed_sample_abstains():
    assert vote([None, None], 2) == ({"type": "abstain"}, 0.0)


def test_decide_threshold_zero_and_unanswerable():
    assert decide(N(5), 0.5, 0.5)
    assert not decide(N(5), 0.49, 0.5)
    assert not decide(N(0), 1.0, 0.5)                                    # 0 is almost never a gold answer
    assert not decide({"type": "list", "answer": [0, 0]}, 1.0, 0.5)
    assert decide({"type": "boolean", "answer": False}, 1.0, 0.5)        # False is a real answer
    assert not decide({"type": "unanswerable"}, 1.0, 0.5)


def test_final_answer():
    assert final_answer(N(5), 0.8, 0.5) == N(5)
    assert final_answer(N(5), 0.8, 0.5, out_of_range=True)["type"] == "unanswerable"
    assert final_answer({"type": "unanswerable"}, 0.8, 0.5)["type"] == "abstain"
    assert final_answer({"type": "unanswerable"}, 0.8, 0.5, answer_unanswerable=True) == {"type": "unanswerable"}


def test_debt_pool_keeps_macro_samples_on_debt_questions():
    macro = "TYPE: number\n```sql\nSELECT ROUND(SUM(debt)) FROM debt_at(DATE '2026-01-31')\n```"
    raw = "TYPE: number\n```sql\nSELECT SUM(\"Còn nợ\") FROM \"don_hang\"\n```"
    assert debt_pool([raw, macro, raw], 3, debt_q=True) == [1]
    assert debt_pool([raw, macro, raw], 3, debt_q=False) == [0, 1, 2]
