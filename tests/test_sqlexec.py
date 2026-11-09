"""Executing a sample on a tiny in-memory shop."""
import duckdb
import pytest

from datamate.sqlexec import execute, parse, to_answer


@pytest.fixture()
def db():
    con = duckdb.connect()
    con.execute("CREATE TABLE ban_hang (ngay DATE, mon VARCHAR, tien BIGINT)")
    con.execute("INSERT INTO ban_hang VALUES ('2026-03-01', 'Trà đào', 30000), ('2026-03-02', 'Bạc xỉu', 25000), ('2026-04-01', 'Trà đào', 30000)")
    return con


def test_parse_reads_type_and_last_sql_block():
    assert parse("notes\nTYPE: number\n```sql\nSELECT 1\n```") == ("number", "SELECT 1")
    assert parse("<think>still thinking") == (None, None)  # a cut-off thought has no query


def test_number_answer(db):
    out = execute(db, 'TYPE: number\n```sql\nSELECT SUM("tien") FROM "ban_hang" WHERE "ngay" < DATE \'2026-04-01\'\n```')
    assert out == {"type": "number", "answer": 55000}


def test_month_string_becomes_a_date(db):
    out = execute(db, "TYPE: string\n```sql\nSELECT 'tháng 3/2026'\n```")
    assert out == {"type": "date", "answer": "2026-03"}


def test_writes_are_refused(db):
    with pytest.raises(ValueError, match="read-only"):
        execute(db, "TYPE: number\n```sql\nDELETE FROM ban_hang\n```")


def test_percent_change_needs_a_subtraction(db):
    with pytest.raises(ValueError, match="change in %"):
        execute(db, "TYPE: percent\n```sql\nSELECT ROUND(100.0 * 3 / 2, 2)\n```", question="Doanh thu tăng bao nhiêu % so với tháng trước?")


def test_shapes():
    assert to_answer("list", ["mon"], [["Trà đào"], ["Bạc xỉu"]]) == ["Trà đào", "Bạc xỉu"]
    assert to_answer("table", ["mon", "tien"], [["Trà đào", 1]]) == {"columns": ["mon", "tien"], "rows": [["Trà đào", 1]]}
    assert to_answer("number", ["x"], [[1234.5]]) == 1235          # money rounds half up to whole đồng
    with pytest.raises(ValueError, match="one cell"):
        to_answer("number", ["a", "b"], [[1, 2]])
