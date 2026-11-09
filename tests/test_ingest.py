"""Representative rows from already-ingested DuckDB tables."""
import duckdb
from openpyxl import Workbook

from datamate.ingest import load_business, profile, sample_text


def meta_for(db, *tables):
    return {"tables": [profile(db, table, f"{table}.csv") for table in tables], "readme": ""}


def row_lines(text):
    return [line for line in text.splitlines() if line.startswith("[")]


def test_sample_text_keeps_related_vietnamese_values_and_nulls():
    db = duckdb.connect()
    db.execute('CREATE TABLE "ban_hang" ("Mã đơn" VARCHAR, "Trạng thái" VARCHAR, "Sản phẩm" VARCHAR, "Số lượng" DOUBLE, "Đơn vị" VARCHAR, "Thành tiền" BIGINT)')
    db.execute("INSERT INTO \"ban_hang\" VALUES (?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?)",
               ["B2", "Hoàn thành", "Trà đào", 2, "ly", 60000, "B1", "Hủy", "Bạc xỉu", 1, "ly", -25000,
                "B3", None, "Cà phê", None, None, None])

    text = sample_text(db, meta_for(db, "ban_hang"), max_rows=10)

    assert 'TABLE "ban_hang"' in text
    assert 'Columns: "Mã đơn", "Trạng thái", "Sản phẩm", "Số lượng", "Đơn vị", "Thành tiền"' in text
    assert '"Trà đào"' in text and '"Hủy"' in text and "NULL" in text


def test_sample_text_handles_related_tables_and_skips_calendar():
    db = duckdb.connect()
    db.execute('CREATE TABLE "orders" ("order_id" VARCHAR, "customer_id" VARCHAR, "status" VARCHAR)')
    db.execute('CREATE TABLE "order_lines" ("order_id" VARCHAR, "product_code" VARCHAR, "qty" INTEGER)')
    db.execute('CREATE TABLE "products" ("product_code" VARCHAR, "product_name" VARCHAR, "unit" VARCHAR)')
    db.execute('CREATE TABLE "calendar" ("day" DATE)')
    db.execute("INSERT INTO \"orders\" VALUES ('O1', 'C1', 'done')")
    db.execute("INSERT INTO \"order_lines\" VALUES ('O1', 'P1', 2)")
    db.execute("INSERT INTO \"products\" VALUES ('P1', 'Sữa', 'chai')")

    text = sample_text(db, meta_for(db, "orders", "order_lines", "products", "calendar"))

    assert 'TABLE "orders"' in text and 'TABLE "order_lines"' in text and 'TABLE "products"' in text
    assert 'TABLE "calendar"' not in text


def test_sample_text_handles_excel_sheets_empty_and_small_tables(tmp_path):
    folder = tmp_path / "data" / "shop"
    folder.mkdir(parents=True)
    book = Workbook()
    first = book.active
    first.title = "2025"
    first.append(["Mã đơn", "Số lượng"])
    first.append(["25-1", 1])
    second = book.create_sheet("2026")
    second.append(["Mã đơn", "Số lượng"])
    second.append(["26-1", 2])
    book.save(folder / "book.xlsx")

    db, meta = load_business(tmp_path, "shop")
    text = sample_text(db, meta)

    assert 'TABLE "book_all"' in text
    assert "book.xlsx#2025" in text and "book.xlsx#2026" in text
    assert '"25-1"' in text and '"26-1"' in text


def test_sample_text_handles_empty_and_small_tables():
    db = duckdb.connect()
    db.execute('CREATE TABLE "empty_sheet" ("Mã đơn" VARCHAR, "Số lượng" INTEGER)')
    db.execute('CREATE TABLE "tiny" ("Mã đơn" VARCHAR, "Số lượng" INTEGER)')
    db.execute("INSERT INTO \"tiny\" VALUES ('T1', 1)")

    text = sample_text(db, meta_for(db, "empty_sheet", "tiny"))

    assert 'TABLE "empty_sheet"' in text and 'TABLE "tiny"' in text
    assert '["T1", 1]' in text


def test_sample_text_is_deterministic_and_deduplicates_rows():
    db = duckdb.connect()
    db.execute('CREATE TABLE "sales" ("order_id" VARCHAR, "status" VARCHAR, "amount" INTEGER)')
    db.execute("INSERT INTO \"sales\" VALUES ('A', 'done', 1), ('A', 'done', 1), ('B', 'cancel', -2)")
    meta = meta_for(db, "sales")

    first = sample_text(db, meta)
    second = sample_text(db, meta)

    assert first == second
    assert row_lines(first).count('["A", "done", 1]') == 1


def test_sample_text_respects_prompt_limit():
    db = duckdb.connect()
    db.execute('CREATE TABLE "sales" ("order_id" VARCHAR, "product_name" VARCHAR, "amount" INTEGER)')
    db.execute("INSERT INTO \"sales\" VALUES ('A', ?, 1), ('B', ?, 2)", ["x" * 500, "y" * 500])

    text = sample_text(db, meta_for(db, "sales"), max_cell_chars=20, max_total_chars=140)

    assert len(text) <= 140
