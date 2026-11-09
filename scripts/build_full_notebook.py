"""Build one notebook that runs the whole workflow (01 EDA -> 02 inference -> 03 evaluation -> 04 training data -> 05 LoRA training)
from the five notebooks, so it never drifts from them.

    .venv/bin/python scripts/build_full_notebook.py                                   # notebooks/00_full_pipeline.ipynb (no outputs)
    .venv/bin/python scripts/build_full_notebook.py --from-dir ~/projects/datamate_notebooks_outputs \
        --out ~/projects/datamate_notebooks_outputs/00_full_pipeline.ipynb            # the same cells with each part's executed outputs

One setup and one import cell at the top. Each part keeps its parameters cell and its cells, the per-part package install
(`install("llm")`, `install("train")`, a no-op in a local checkout) becomes the part's first code cell, and headings move down one level.
The full notebook is the teaching version: the markdown in LESSON replaces the parts' engineering notes, and EXTRA adds a few cells
(an example per answer type, one question through the pipeline)."""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
PARTS = ["01_eda", "02_inference", "03_evaluation", "04_training_data", "05_train_lora"]
INTRO = """# DataMate · Baseline text-to-SQL (Zalo AI Challenge 2026, Track 2)

**Bài toán:** trả lời câu hỏi tiếng Việt về dữ liệu kinh doanh của một cửa hàng (các bảng csv / xlsx / sqlite và một README). Đáp án có kiểu
cố định (số, phần trăm, chuỗi, danh sách, ngày, bảng…), hoặc bỏ trống khi không chắc.

**Ý tưởng của baseline:**
- Mô hình ngôn ngữ (Qwen3.5-9B + LoRA) **viết SQL**, còn DuckDB **thực hiện phép tính** trên dữ liệu thật.
- Mỗi câu hỏi lấy nhiều mẫu SQL, chạy từng mẫu rồi **bỏ phiếu** theo đáp án. Khi các mẫu không đồng thuận, hệ thống **bỏ trống**
  thay vì đoán, vì câu trả lời sai bị trừ điểm.
- LoRA được train trên chính các lời giải đúng mà mô hình tìm ra trên tập train (rejection sampling).

| Phần | Nội dung | Phần cứng |
|---|---|---|
| 01 · EDA | dữ liệu gồm những gì: shop, tệp, kiểu đáp án, dạng câu hỏi, rulebook | CPU |
| 02 · Suy luận | chạy pipeline và ghi file nộp `submission.jsonl` | GPU bf16 từ 24 GB |
| 03 · Đánh giá | điểm PA, nhóm lỗi, ngưỡng bỏ phiếu | CPU |
| 04 · Dữ liệu train LoRA | dựng dữ liệu SFT từ các lời giải đúng | CPU, GPU cho bước ghi chú |
| 05 · Train LoRA | QLoRA trên Qwen3.5-9B | GPU bf16 từ 24 GB |

Chạy từ trên xuống. Mỗi phần có một ô tham số ở đầu phần. Khi không có GPU bf16 từ 24 GB (ví dụ GPU T4 trên Kaggle), các bước cần mô hình
dùng kết quả đã chạy sẵn trên GPU 24 GB, nên notebook vẫn chạy hết và vẫn vẽ được kết quả."""

# Text for learners: replaces the markdown cell whose first line matches (after headings move down one level).
LESSON = {
    "## 01 · Khám phá dữ liệu (EDA)": """## 01 · Khám phá dữ liệu (EDA)

Trước khi xây pipeline, ta xem dữ liệu trông như thế nào:
- mỗi cửa hàng (shop) có bao nhiêu câu hỏi và những tệp dữ liệu gì
- đáp án có những kiểu nào, câu hỏi thường hỏi gì
- rulebook quy định cách tính (doanh thu, hàng bán chạy, công nợ…) ra sao
- một shop sau khi nạp vào DuckDB trông thế nào: đây chính là phần schema mà mô hình nhìn thấy""",
    "### Dạng câu hỏi": """### Dạng câu hỏi

Tỷ lệ câu hỏi chứa một số dạng khó, ví dụ khung thời gian ("tuần rồi", "quý 2"), công nợ, số tiền viết tắt ("3tr7") hay từ địa phương.
Pipeline tính sẵn gợi ý cho các dạng này (`datamate/hints.py`) để mô hình không phải tự suy đoán.""",
    "## 02 · Suy luận (inference)": """## 02 · Suy luận (inference)

Pipeline xử lý từng shop qua năm bước:
1. **Nạp dữ liệu** vào DuckDB và dựng sẵn view chuẩn (`v_sales`, `v_orders`) cùng macro công nợ, để câu SQL ngắn và ít sai.
2. **Tạo prompt** gồm rulebook, schema, khung thời gian tính sẵn, vài ví dụ tương tự và câu hỏi.
3. **Lấy mẫu:** mô hình viết 8 câu SQL. Câu khó (các mẫu không đồng thuận) được thêm 4 mẫu ở chế độ thinking.
4. **Thực thi và sửa lỗi:** DuckDB chạy từng câu SQL. Câu bị lỗi được gửi lại cho mô hình kèm thông báo lỗi để sửa.
5. **Bỏ phiếu:** gom các đáp án giống nhau. Hệ thống chỉ trả lời khi nhóm lớn nhất đủ đồng thuận, ngược lại thì bỏ trống.

Kết quả là file `submission.jsonl` đúng định dạng nộp bài. `LIMIT = 10` chạy thử 10 câu, còn cả 1000 câu public mất khoảng 3,5 giờ
trên một GPU 24 GB.""",
    "### Kết quả": """### Kết quả

Số câu đã trả lời theo shop, vài câu mẫu cùng đáp án và tỷ lệ đồng thuận (`conf`). Với `SPLIT = "train"` có thêm điểm theo đáp án chuẩn.""",
    "## 03 · Đánh giá": """## 03 · Đánh giá

Điểm của cuộc thi là **PA (penalized accuracy)**: câu đúng +1, câu sai −1, câu bỏ trống 0, rồi chia cho số câu. Vì câu sai bị trừ điểm,
bỏ trống những câu không chắc thường có lợi hơn là đoán.

Phần này gồm:
1. Chấm một lượt chạy trên **train** (có đáp án): điểm tổng, theo shop, theo dạng câu, nhóm lỗi và ngưỡng bỏ phiếu.
2. So các **bản nộp public** với nhau, không cần nhãn.""",
    "### 1. Điểm của lượt chạy train": """### 1. Điểm của lượt chạy train

- **PA:** điểm của cuộc thi.
- **precision:** tỷ lệ đúng trong các câu có trả lời. **coverage:** tỷ lệ câu có trả lời.
- **Khoảng tin cậy:** lấy mẫu lại theo shop (bootstrap), cho biết điểm dao động bao nhiêu nếu đổi sang tập shop khác.""",
    "### Ngưỡng bỏ phiếu và oracle@K": """### Ngưỡng bỏ phiếu và oracle@K

- **Ngưỡng bỏ phiếu:** tỷ lệ mẫu đồng thuận tối thiểu để trả lời. Ngưỡng cao thì ít sai hơn nhưng bỏ trống nhiều hơn. Biểu đồ bỏ phiếu lại
  trên các mẫu đã lưu với từng ngưỡng.
- **oracle@K:** tỷ lệ câu có ít nhất một mẫu đúng trong K mẫu, tức mức điểm tối đa nếu luôn chọn được mẫu đúng.""",
    "## 04 · Dữ liệu train LoRA": """## 04 · Dữ liệu train LoRA

LoRA học từ chính các lời giải đúng mà mô hình tìm ra trên tập train (rejection sampling):
1. **Chạy pipeline trên train** (có đáp án). Mỗi câu có 8 mẫu SQL, câu khó có thêm mẫu thinking.
2. **Giữ lời giải đúng:** so đáp án của từng mẫu với đáp án chuẩn. Ưu tiên lời giải dùng macro công nợ, rồi view chuẩn, cuối cùng mới tới
   bảng thô. Câu khó (ít mẫu đúng) được giữ nhiều dòng hơn câu dễ.
3. **Ghi chú trước SQL:** thêm vài dòng "Notes:" (khung thời gian, cách tính) trước câu SQL, đúng định dạng mô hình viết lúc suy luận.
   Bước này cần GPU.

Chỉ dùng dữ liệu train. Câu public / private không bao giờ được đưa vào dữ liệu train.""",
    "## 05 · Train LoRA": """## 05 · Train LoRA

- **QLoRA:** mô hình gốc Qwen3.5-9B được nạp ở dạng 4-bit và giữ nguyên, chỉ train các ma trận LoRA hạng 16 gắn vào attention và MLP
  (chưa tới 1% số tham số), nên vừa một GPU 24 GB.
- **Loss chỉ tính trên phần trả lời:** mỗi dòng dài 9k đến 19k token vì prompt chứa rulebook và schema, nhưng loss chỉ tính trên phần
  ghi chú và SQL mà mô hình cần viết (khoảng 4% số token).
- **Unsloth** giúp train nhanh và tiết kiệm bộ nhớ. Bản đã nộp (LoRA v3) train 2910 dòng trong khoảng 8 giờ trên một GPU 24 GB.

`LIMIT_ROWS` và `MAX_STEPS` cho phép chạy thử vài bước để xem quy trình và đường loss.""",
    "### Kernel fla": """### Kernel fla

24 trên 32 lớp của Qwen3.5-9B là Gated DeltaNet, một dạng attention tuyến tính. Thư viện `fla-core` cung cấp kernel Triton nhanh hơn
khoảng 90 lần so với bản PyTorch tham chiếu. Ô này kiểm tra kernel cho kết quả khớp với bản tham chiếu trước khi train.""",
    "### Dùng adapter vừa train": """### Dùng adapter vừa train

Đặt `LORA` ở phần 02 bằng đường dẫn `OUT/adapter` (ví dụ `"runs/lora/sql_rft_nb/adapter"`) để chạy suy luận với adapter vừa train,
rồi so điểm với adapter cũ ở phần 03.""",
}

# A few cells for learners, inserted after the code cell whose source starts with the anchor.
EXTRA = {
    "types = Counter(": [
        ("markdown", "Mỗi kiểu đáp án một ví dụ:"),
        ("code", """example = {}
for q in train_q:
    a = train_a[q["id"]]
    example.setdefault(a["type"], (q["business_id"], q["question"], a.get("answer")))
pd.DataFrame([{"type": t, "shop": shop, "question": text[:100], "answer": json.dumps(ans, ensure_ascii=False)[:70]}
              for t, (shop, text, ans) in example.items()])"""),
    ],
    "trace = [r for p in sorted(out.glob": [
        ("markdown", """### Một câu hỏi đi qua pipeline

Câu hỏi có tỷ lệ đồng thuận thấp nhất (thường là câu khó, được thêm mẫu thinking): chế độ, câu SQL và đáp án của từng mẫu, rồi kết quả bỏ phiếu.
- Mẫu không viết ra được SQL hợp lệ (ví dụ ghi chú lặp mãi không dừng) không được tính phiếu.
- Tỷ lệ đồng thuận = số phiếu của đáp án thắng / tổng số mẫu."""),
        ("code", """def last_sql(text):
    return " ".join(text.split("```sql")[-1].split("```")[0].split()) if "```sql" in text else ""

q = min(trace, key=lambda r: (r["conf"], -len(r["samples"])))
print(q["question"])
samples = pd.DataFrame([{"mẫu": k + 1,
                         "chế độ": "thinking" if "</think>" in s["text"] else "thường",
                         "đáp án": json.dumps(s["pred"]["answer"], ensure_ascii=False)[:50] if s.get("pred") else "",
                         "SQL": last_sql(s["text"])[:110] or "(lỗi: " + (s.get("error") or "").split(" | ")[0] + ")"}
                        for k, s in enumerate(q["samples"])])
display(samples)
votes = Counter(a for a in samples["đáp án"] if a)
print("bỏ phiếu:", dict(votes.most_common()), f"| {len(samples)} mẫu")
print("đáp án cuối:", json.dumps(q["final"], ensure_ascii=False), "| tỷ lệ đồng thuận:", round(q["conf"], 2))"""),
    ],
}


def demote(md: str) -> str:
    return re.sub(r"^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", md, flags=re.M)


def install_kind(setup_source: str) -> str:
    return re.search(r'install\("(\w+)"\)', setup_source).group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path, default=REPO / "notebooks", help="where the five notebooks are read (executed copies keep outputs)")
    ap.add_argument("--out", type=Path, default=REPO / "notebooks/00_full_pipeline.ipynb")
    a = ap.parse_args()
    books = {name: nbformat.read(a.from_dir / f"{name}.ipynb", as_version=4) for name in PARTS}
    first = books[PARTS[0]]
    full = nbformat.v4.new_notebook(metadata=copy.deepcopy(first.metadata))
    full.cells.append(nbformat.v4.new_markdown_cell(INTRO))
    setup, imports = copy.deepcopy(first.cells[1]), copy.deepcopy(first.cells[2])
    setup.source = setup.source.replace('install("base")', 'install("base")            # each part below installs what it needs (Kaggle only)')
    full.cells += [setup, imports]
    for name in PARTS:
        nb = books[name]
        intro = copy.deepcopy(nb.cells[0])
        intro.source = demote(intro.source)
        intro.source = LESSON.get(intro.source.split("\n")[0], intro.source)
        full.cells.append(intro)
        kind = install_kind(nb.cells[1].source)
        if kind != "base":
            note = "  # Kaggle: a separate session for this part (the training stack differs from vLLM)" if kind == "train" else ""
            full.cells.append(nbformat.v4.new_code_cell(f'install("{kind}"){note}'))
        for cell in nb.cells[3:]:
            cell = copy.deepcopy(cell)
            if cell.cell_type == "markdown":
                cell.source = demote(cell.source)
                cell.source = LESSON.get(cell.source.split("\n")[0], cell.source)
            full.cells.append(cell)
            if cell.cell_type == "code":
                for anchor, extra in EXTRA.items():
                    if cell.source.startswith(anchor):
                        full.cells += [nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src) for kind, src in extra]
    count = 0
    for cell in full.cells:
        cell.pop("id", None)
        if cell.cell_type == "code":
            if cell.get("outputs"):
                count += 1
                cell.execution_count = count
                for out in cell.outputs:
                    if "execution_count" in out:
                        out["execution_count"] = count
            else:
                cell.execution_count = None
    nbformat.validator.normalize(full)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(full, a.out)
    print(f"{len(full.cells)} cells -> {a.out}")


if __name__ == "__main__":
    main()
