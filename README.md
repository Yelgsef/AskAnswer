# DataMate - Zalo AI Challenge 2026

Hệ thống trả lời câu hỏi tiếng Việt về dữ liệu bán hàng của từng cửa hàng (csv / xlsx / sqlite, mỗi shop một kiểu lộn xộn).
Với mỗi câu hỏi, hệ thống trả về một đáp án có kiểu (số, phần trăm, ngày, chuỗi, danh sách, bảng, `unanswerable`) hoặc bỏ trống
khi không chắc. Điểm là Penalized Accuracy: đúng +1, bỏ trống 0, sai -1.

**Ý tưởng chính:** Qwen3.5-9B cùng một LoRA tự train viết nhiều câu SQL DuckDB cho mỗi câu hỏi. Các câu SQL được chạy trên dữ
liệu đã chuẩn hóa, kết quả được bỏ phiếu, và hệ thống chỉ trả lời khi đủ số mẫu đồng ý.

- Kết quả: **0.683** public (bản nộp v9). Lịch sử đầy đủ ở [docs/leaderboard.md](docs/leaderboard.md).
- Ràng buộc: chạy offline, một model ≤ 15B tham số (9.65B + LoRA 43M), xong private test trong ≤ 6 giờ.

## Mục lục

1. [Cách hoạt động](#cách-hoạt-động)
2. [Yêu cầu](#yêu-cầu)
3. [Cài đặt](#cài-đặt)
4. [Inference](#inference)
5. [Docker (bản chạy private)](#docker-bản-chạy-private)
6. [Đánh giá và phân tích lỗi](#đánh-giá-và-phân-tích-lỗi)
7. [Train LoRA](#train-lora)
8. [Notebook (máy local, Kaggle và Colab)](#notebook-máy-local-kaggle-và-colab)
9. [Cấu trúc repo](#cấu-trúc-repo)
10. [Kết quả public](#kết-quả-public)
11. [Tuân thủ luật thi](#tuân-thủ-luật-thi)
12. [License](#license)

## Cách hoạt động

```
questions.jsonl ─┐
data/<shop>/  ───┼─► ingest      nạp mọi bảng vào DuckDB, đoán header, chuẩn hóa ngày / tiền / tên, profile cột
rulebook.md   ───┘      │
                        ▼
                     views       mỗi shop một lần: model viết K bản nháp SELECT ánh xạ dữ liệu thô sang bảng chuẩn
                     debts       v_sales / v_orders và lớp công nợ (macro debt_at, debt_new, debt_repaid), kiểm tra bản
                        │        nháp trên dữ liệu (danh mục, giờ, đơn vị), sửa lỗi, chọn bản được đa số
                        ▼
                     hints       khung ngày, gợi ý theo dạng câu hỏi, giá trị trong dữ liệu khớp câu hỏi
                     prompts     2 system prompt × K/2 mẫu, few-shot lấy từ ngân hàng ví dụ của train
                        │
                        ▼
                     llm         vLLM + LoRA: 8 mẫu SQL thường, thêm 4 mẫu thinking cho câu khó và câu công nợ
                     sqlexec     chạy read-only có timeout, ép kiểu đáp án, lỗi → một vòng sửa (repair)
                        │
                        ▼
                     voting      gom đáp án theo đúng cách chấm, bỏ phiếu, trả lời khi ≥ 50% mẫu đồng ý, bỏ trống
                                 khi đáp án thắng là 0 / rỗng
                        │
                        ▼
                     submission  ghi submission.jsonl sau mỗi shop (hết giờ vẫn giữ phần đã trả lời)
```

`datamate/pipeline.py` nối các bước. Mỗi bước là một method riêng, nên có thể dựng lại prompt và kết quả bỏ phiếu của một lượt đã
lưu mà không cần GPU (`analysis/replay_run.py`).

## Yêu cầu

|          | Tối thiểu                                                 | Đã chạy                                                        |
| -------- | --------------------------------------------------------- | -------------------------------------------------------------- |
| GPU      | 24 GB, hỗ trợ bf16 (T4 / P100 không chạy được)            | 1 × RTX PRO 4000 Blackwell 24 GB, 2 × RTX 5090 cho bản private |
| RAM      | 30 GB                                                     | 30 GB                                                          |
| Phần mềm | CUDA ≥ 12.8, Python ≥ 3.12, `git-lfs`                     | vLLM 0.32.0, DuckDB 2.0.0                                      |
| Ổ đĩa    | khoảng 20 GB cho model gốc, khoảng 70 GB nếu build Docker |                                                                |

Phần đọc dữ liệu, chấm điểm, test và các notebook 01 / 03 chỉ cần CPU.

## Cài đặt

**1. Clone repo và tải LoRA (Git LFS):**

```bash
git clone https://github.com/hugebenevolence/zalo-ai-challenge-datamate-26.git
cd zalo-ai-challenge-datamate-26
git lfs install && git lfs pull          # tải adapters/sql_rft_v3
```

**2. Tạo môi trường Python:**

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[llm,dev]"    # chỉ cần CPU: bỏ "llm", dùng .[dev]
```

**3. Tải model gốc** (đúng revision đã dùng khi nộp, xem `docker/model_manifest.json`):

```bash
.venv/bin/hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/Qwen3.5-9B
```

**4. Đặt dữ liệu của BTC** vào `data/` (thư mục này không có trong git):

```
data/
├── train/
│   ├── questions.jsonl     {"id", "business_id", "question", "as_of_date"}
│   ├── answers.jsonl       {"id", "type", "answer", "alias"}   (chỉ train có)
│   ├── rulebook.md
│   ├── calendar.csv
│   └── data/<shop>/        README.md và các tệp dữ liệu của shop
└── public_test/            như train, không có answers.jsonl
```

**5. Kiểm tra cài đặt**:

```bash
.venv/bin/python -m pytest -q
```

## Inference

Mọi lệnh chạy dùng chung bộ flag của bản đã nộp trong `scripts/flags.sh`. Mỗi GPU chạy một tiến trình, mỗi tiến trình nhận một phần
các shop. Kết quả được gộp thành một file `submission.jsonl`.

```bash
# Public test, dùng lại view và lớp công nợ đã lưu của bản 0.683 (configs/public_v9). Khoảng 3,5 giờ trên 1 GPU 24 GB.
bash scripts/run_public.sh runs/public

# Public test, soạn lại mọi view và lớp công nợ từ đầu, giống hệt cách chạy private trong Docker.
FRESH=1 bash scripts/run_public.sh runs/public_fresh

# 632 câu holdout của train (splits/holdout600.txt), có chấm điểm ở runs/holdout/score.json.
bash scripts/run_holdout.sh runs/holdout

# Kiểm tra định dạng file nộp.
.venv/bin/python -m datamate.submission check runs/public/submission.jsonl --questions data/public_test/questions.jsonl
```

File cần nộp: `runs/public/submission.jsonl`. Log và trace từng câu nằm trong `runs/public/shard<i>/`.

Điều chỉnh bằng biến môi trường:

| Biến                    | Mặc định                                   | Dùng khi                                   |
| ----------------------- | ------------------------------------------ | ------------------------------------------ |
| `NUM_GPUS`              | số GPU thấy được                           | chỉ dùng một phần GPU                      |
| `GPU_MEM`               | `0.92`                                     | GPU đang có tiến trình khác (ví dụ `0.75`) |
| `ENGINE`                | xem `scripts/flags.sh`                     | hết bộ nhớ: giảm `max_num_seqs`            |
| `MODEL_DIR`, `LORA_DIR` | `models/Qwen3.5-9B`, `adapters/sql_rft_v3` | model hoặc LoRA ở chỗ khác                 |
| `PYTHON`                | `.venv/bin/python`                         | dùng môi trường Python khác                |

`python -m datamate.predict --help` liệt kê mọi flag của pipeline.

Dynamic business macros are disabled by default. For an optional comparison run, pass `--dynamic-macros`; tune with
`--macro-max-count`, `--macro-repairs` and `--macro-max-context-chars`. The submitted defaults in `scripts/flags.sh` and Docker are unchanged.

> Mỗi lượt chạy sinh mẫu mới, nên cùng một cấu hình có thể lệch khoảng ±1 điểm public. Muốn so hai cấu hình thì so hai lượt
> chạy đầy đủ, không ghép kết quả từ nhiều lượt.

## Docker (bản chạy private)

Theo hướng dẫn Docker bản cuối của BTC (`zac2026_DataMate_docker_guideline_final.pdf`). Cần làm xong bước 1 và 3 của phần cài đặt
(LoRA và model gốc được chép vào image), và có sẵn dữ liệu train, dữ liệu SFT, log train ở máy build.

```bash
python3 scripts/pack_training_code.py      # build/training_code: code, dữ liệu, log, checkpoint, data_manifest.json (SHA-256)
docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_<team_name>:final .
# lệnh chấm của BTC: 2 GPU, không mạng, /data chỉ đọc, shm 16 GB
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
    -v /duong/dan/toi/split:/data:ro -v "$PWD/result:/result" zaic2026_datamate_<team_name>:final /code/predict.sh
```

- Layout trong image: `/code/predict.sh` (chạy không tham số), `/code/README.md` (mô tả cho BTC, nguồn ở `docker/README.md`),
  `/code/src/datamate`, `/code/models/{Qwen3.5-9B,lora}`, `/code/training_code/` (README ở `docker/training_code_README.md`).
- `/data` là một thư mục split giống `data/public_test/` (questions.jsonl, rulebook.md, calendar.csv, data/).
- Kết quả:
  - `/result/submission.jsonl` được ghi lại sau mỗi shop, nên nếu bị dừng giữa chừng thì các câu đã trả lời vẫn được tính;
  - `/result/logs/<id>.json`: mỗi câu một bản ghi trên một dòng, gồm `id`, `answer`, `elapsed_ms`, `thinking`, `code`, `output`;
  - `/result/work/`: view, lớp công nợ và trace của từng GPU.
- Container chạy offline và luôn soạn view mới cho từng shop, không dùng `configs/public_v9`.
- Mỗi tiến trình dừng sau `TIMEOUT_SECONDS` (mặc định 21000 giây).
- Base image là vLLM 0.32.0, không phải bản 0.10.1.1 BTC gợi ý, vì 0.10.1.1 chưa chạy được Qwen3.5. Hướng dẫn cho phép image khác
  miễn hỗ trợ RTX 5090.
- Train lại trong image: `bash /code/training_code/train.sh` (môi trường train riêng ở `/opt/venv-train`).
- Nén để nộp:
  ```bash
  set -o pipefail
  docker save zaic2026_datamate_<team_name>:final | pigz -p 8 > zaic2026_datamate_<team_name>.tar.gz
  pigz -dc zaic2026_datamate_<team_name>.tar.gz | wc -c                                  # dung lượng .tar chưa nén, phải ≤ 100 GB
  md5sum zaic2026_datamate_<team_name>.tar.gz > zaic2026_datamate_<team_name>.tar.gz.md5
  ```

## Đánh giá và phân tích lỗi

Đáp án train chỉ được đọc bởi các công cụ đánh giá. Code inference không import `datamate/scoring.py`.

```bash
# Điểm của một file nộp trên train (hoặc một tập con qua --questions)
.venv/bin/python -m datamate.scoring --data data/train --pred RUN/submission.jsonl --output RUN/score.json

# PA, Oracle@K, câu bị mất ở bước nào, so cặp hai lượt chạy
.venv/bin/python analysis/err.py RUN [RUN2]
```

| Công cụ                    | Làm gì                                                                                                            | Cần nhãn       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------- |
| `analysis/err.py`          | PA, coverage, Oracle@K, câu sai mất ở bước nào (không mẫu nào đúng / bỏ phiếu / ngưỡng), so cặp                   | có             |
| `analysis/error_matrix.py` | ma trận lỗi: dạng câu × nhóm lỗi, loại đáp án, đường đi (macro / view / bảng thô), shop, tách câu sai có hệ thống | có             |
| `analysis/gate_cv.py`      | chạy lại các cách chọn đáp án trên trace, chọn ngưỡng bằng leave-one-shop-out                                     | có             |
| `analysis/flags.py`        | flag theo từng bước (shop, câu hỏi, mẫu), có nhãn thì chỉ ra bước đầu tiên làm sai                                | không bắt buộc |
| `analysis/view_flags.py`   | kiểm tra view đã lưu với bảng thô: giờ, đơn vị, khách hàng, trạng thái, dòng trùng                                | không          |
| `analysis/replay_run.py`   | dựng lại prompt và bỏ phiếu của lượt đã lưu, thay đổi không đổi hành vi phải cho 0 khác biệt                      | không          |

Holdout (`splits/holdout600.txt`, 6 shop) không còn sạch với LoRA v3: v3 đã được train trên một phần câu của các shop này. Với v3,
điểm holdout chỉ dùng để kiểm tra nhanh, không dùng để chọn cấu hình (chi tiết ở [docs/leaderboard.md](docs/leaderboard.md)).

## Train LoRA

Cách dễ làm theo nhất là hai notebook `04_training_data` và `05_train_lora`. Nếu dùng dòng lệnh, tạo môi trường train riêng:

```bash
python3 -m venv .venv-train
.venv-train/bin/pip install -r training/requirements.txt
.venv-train/bin/pip install fla-core==0.6.0 --no-deps     # kernel Gated DeltaNet
```

Các bước:

```bash
# 1. Lượt chạy có nhãn trên 24 shop train ngoài holdout, giữ lại prompt (đầu vào của dữ liệu SFT), thêm các flag của scripts/flags.sh
.venv/bin/python -m datamate.predict --data data/train --ids splits/rft_train24.txt --output runs/train_pool --save-prompts ...

# 2. Chọn các mẫu đúng làm dữ liệu SFT, rồi thêm ghi chú trước SQL
.venv/bin/python training/build_sft.py runs/train_pool --fewshot assets/fewshot_bank.jsonl --out data/sft/sql_rft.jsonl --easy 800 --max 4000
.venv/bin/python training/add_notes.py data/sft/sql_rft.jsonl data/sft/sql_rft_notes.jsonl

# 3. Kiểm tra kernel fla khớp bản tham chiếu PyTorch, rồi train tiếp từ LoRA v3
.venv-train/bin/python training/check_fla.py
.venv-train/bin/python training/train_lora.py data/sft/sql_rft_notes.jsonl --out runs/lora/sql_rft_v4 --model models/Qwen3.5-9B \
    --init-adapter adapters/sql_rft_v3 --max-len 32768 --r 16 --alpha 32 --lr 5e-5 --epochs 1 --grad-accum 8 --end '<|im_end|>' \
    --no-thinking --completion-logits [--resume]
```

Ghi chú:

- Mỗi dòng SFT dài từ 9k đến 19k token vì prompt chứa rulebook và schema của shop. `--max-len` dưới khoảng 16k sẽ bỏ phần lớn dữ liệu.
- `training/build_fewshot_bank.py` tạo lại `assets/fewshot_bank.jsonl`, còn `training/merge_lora.py` gộp LoRA vào model gốc.

## Notebook (máy local, Kaggle và Colab)

| Notebook           | Làm gì                                                                                                 | Cần                       |
| ------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------- |
| `00_full_pipeline` | cả 5 phần bên dưới trong một notebook (ghép tự động bằng `scripts/build_full_notebook.py`)             | như từng phần             |
| `01_eda`           | shop, tệp, loại đáp án, dạng câu hỏi, khác biệt rulebook, một shop sau khi nạp                         | CPU                       |
| `02_inference`     | chạy pipeline trên một split (vài câu hoặc cả split), gộp và kiểm tra file nộp, chấm điểm nếu là train | GPU bf16 ≥ 24 GB          |
| `03_evaluation`    | điểm theo loại / shop / dạng câu, nhóm lỗi, ngưỡng, Oracle@K, so các bản nộp không cần nhãn            | CPU                       |
| `04_training_data` | dữ liệu SFT từ lượt chạy train có nhãn, ghi chú trước SQL, thống kê                                    | CPU, GPU cho bước ghi chú |
| `05_train_lora`    | kiểm tra kernel fla, train QLoRA (tiếp tục được khi phiên bị ngắt), đồ thị loss                        | GPU                       |

- Chạy theo thứ tự 01 → 05 (hoặc `00_full_pipeline` từ trên xuống). Ô đầu tiên tự nhận biết đang ở máy local, Kaggle hay Colab.
- Mỗi notebook có một ô `parameters`, chạy được bằng papermill: `papermill notebooks/02_inference.ipynb out.ipynb -p LIMIT 10`.
- Cả 5 notebook đã chạy trên máy local (GPU 24 GB). `00_full_pipeline` đã chạy hết trên Kaggle (CPU). Nhánh Colab chưa chạy thử.
- Không có GPU bf16 từ 24 GB (Kaggle T4 / P100, CPU): các bước cần model (suy luận ở 02, ghi chú ở 04, train ở 05) dùng kết quả đã chạy
  trên GPU 24 GB, lưu trong `runs/saved/` của bản sao repo đính kèm (không có trong git), để notebook vẫn chạy hết và vẽ được kết quả.
  `DATAMATE_NO_GPU=1` giả lập trường hợp này trên máy có GPU.
- Muốn giữ một bản có output (ví dụ để dạy) mà repo vẫn sạch: `python scripts/execute_notebooks.py --out <thư mục ngoài repo> 01_eda 03_evaluation`
  (đổi tham số bằng `--set LIMIT=20`), rồi `python scripts/build_full_notebook.py --from-dir <thư mục đó> --out <thư mục đó>/00_full_pipeline.ipynb`.
- Notebook được commit **không kèm output** (output chứa dữ liệu thi), `tests/test_notebooks.py` kiểm tra điều này. Trước khi commit:
  `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`.

Trên Kaggle:

1. **Settings:** bật Internet, chọn Accelerator GPU cho 02 / 04 / 05. Model gốc được tải về `/tmp/models` mỗi phiên (khoảng 19 GB).
   Kết quả nằm ở `/kaggle/working/runs` và được giữ sau phiên.
2. **Code:** đính kèm một dataset là bản sao repo (kèm `adapters/` thật và các `runs/` mà 03 / 04 đọc). Ô đầu tiên dùng bản này nếu có.
   Nếu không, thêm `GITHUB_TOKEN` (token đọc được repo) trong **Add-ons → Secrets** để ô đầu tiên clone repo và tải LoRA qua Git LFS.
3. **Dữ liệu:** thêm một dataset chứa `train/` và `public_test/` như `data/` ở trên. Notebook tự tìm trong `/kaggle/input`.
4. **Giới hạn phiên:** `05_train_lora` lưu checkpoint mỗi `SAVE_STEPS` bước, đặt `RESUME = True` để train tiếp. Train trên GPU
   16 GB chưa được thử. Nếu chỉ để học, đặt `MAX_LEN = 16384` và `LIMIT_ROWS` nhỏ.

Trên Colab:

1. **Drive:** tạo `MyDrive/datamate/` gồm `code/` (bản sao repo như dataset Kaggle ở trên) và `data/` (gồm `train/`, `public_test/`).
   Không có `code/` thì thêm secret `GITHUB_TOKEN` (biểu tượng chìa khoá, bật quyền cho notebook) để ô đầu tiên clone repo.
2. **Runtime:** GPU cho 02 / 04 / 05 (bf16, từ 24 GB, ví dụ A100 hoặc L4). Ô đầu tiên mount Drive. Model gốc tải về `/content/models`,
   kết quả lưu ở `MyDrive/datamate/runs` (giữ sau phiên).
3. **05_train_lora:** chạy trong một phiên riêng (stack train khác vLLM của 02 / 04), giống trên Kaggle.

## Cấu trúc repo

```
datamate/                 package inference (python -m datamate.predict)
  predict.py              CLI: đọc tham số, khởi động model, chạy Pipeline
  pipeline.py             các bước: nạp shop → view / công nợ → prompt → lấy mẫu → bỏ phiếu → ghi kết quả
  llm.py                  lớp bọc vLLM (file duy nhất import vllm)
  ingest.py               nạp và chuẩn hóa dữ liệu shop, profile cột
  views.py, debts.py      bảng chuẩn v_sales / v_orders và lớp công nợ: soạn, kiểm tra, chọn, mô tả
  macros.py               lập kế hoạch, sinh và sửa macro nghiệp vụ tùy theo từng business
  macro_validator.py      kiểm tra an toàn, cấu trúc và bất biến độc lập của macro
  macro_registry.py       đăng ký macro đã được xác minh theo từng business
  hints.py, prompts.py    gợi ý từ câu hỏi, system prompt, user message, gợi ý sửa lỗi
  sqlexec.py, voting.py   chạy một mẫu SQL, bỏ phiếu và quyết định trả lời
  fewshot.py              ngân hàng ví dụ train và truy xuất
  scoring.py              Penalized Accuracy theo luật công bố (không dùng khi inference)
  submission.py           ghi, gộp shard, kiểm tra định dạng file nộp
  values.py               giá trị lọc không có trong dữ liệu (cho analysis/flags.py)
  notebook.py             phần dùng chung của notebook: đường dẫn, khả năng GPU, chạy từng bước kèm log
scripts/                  run_public.sh, run_holdout.sh, flags.sh (flag của bản đã nộp), pack_training_code.py (đóng gói training_code cho Docker)
docker/                   Dockerfile, predict.sh (entry point /code/predict.sh), README.md (= /code/README.md), train.sh và
                          training_code_README.md (vào /code/training_code), requirements.txt, model_manifest.json
training/                 dữ liệu SFT, ghi chú, ngân hàng few-shot, train LoRA, kiểm tra kernel fla, gộp LoRA
analysis/                 công cụ đánh giá offline (bảng ở trên)
notebooks/                00_full_pipeline (cả 5 phần), 01_eda → 02_inference → 03_evaluation → 04_training_data → 05_train_lora
configs/public_v9/        view và lớp công nợ của lượt 0.683 (chỉ để chạy lại public)
assets/fewshot_bank.jsonl ngân hàng ví dụ: câu hỏi train và SQL đúng
adapters/sql_rft_v3/      LoRA v3 (Git LFS)
splits/                   danh sách id: holdout600, holdout_shops, val300, smoke6, rft_train24
tests/                    pytest, chỉ cần CPU
docs/                     đề bài, ghi chú dữ liệu, lịch sử điểm
```

Không có trong repo: `data/` (dữ liệu BTC), `models/` (trọng số gốc), `runs/` (kết quả chạy, log, checkpoint).

## Kết quả public

| Bản       | Thay đổi chính                                                                                   | Public    |
| --------- | ------------------------------------------------------------------------------------------------ | --------- |
| v2        | LoRA v2, few-shot, macro công nợ                                                                 | 0.491     |
|           | `qty` theo đơn vị cơ bản, trả lời `unanswerable` theo đa số                                      | 0.516     |
|           | sinh mẫu mới, chuẩn hóa tên, hệ số đơn vị suy từ giá                                             | 0.60      |
| v6        | ghi chú trước SQL, ví dụ macro công nợ                                                           | 0.63      |
| sql-hints | v6 + gợi ý theo dạng câu hỏi, có danh sách đồng nghĩa lấy từ tên món public (đã bỏ ở v9)         | 0.696     |
| v8        | LoRA v3 (thêm dữ liệu do model 27B giải làm teacher, chỉ trên train), kiểm tra view theo dữ liệu | 0.639     |
| **v9**    | v8 + `sql-hints`, khóa bỏ phiếu theo cách chấm, sửa lớp công nợ                                  | **0.683** |

## Tuân thủ luật thi

- Inference offline, không có người can thiệp, tổng tham số ≤ 15B (Qwen3.5-9B Apache-2.0 + LoRA 43M).
- Không hard-code theo id câu hỏi hay id shop. `configs/public_v9` chỉ dùng để chạy lại public, bản Docker cho private luôn soạn
  view mới.
- Không train trên tập test, chú thích thêm chỉ làm trên train (§4.2). Gợi ý trong prompt là luật của rulebook hoặc kiến thức
  tiếng Việt chung, đã đối chiếu với đáp án train. Những gì chỉ khớp câu hỏi public (từ đồng nghĩa lấy từ tên món public, gợi ý
  dòng sản phẩm) đã bị loại.
- Toàn bộ private test chạy ≤ 6 giờ trên 2 × RTX 5090, log từng câu được giữ để chạy lại (Rule 8, lệch ≤ 0.03).

## License

Code, LoRA adapter và ngân hàng ví dụ trong repo theo [MIT License](LICENSE). Model Qwen3.5-9B tải riêng và
theo license Apache-2.0 của Qwen. Dữ liệu của BTC không nằm trong repo và không thuộc license này.

## Docs

- [Đề bài](docs/challenge.md)
- [Ghi chú dữ liệu](docs/data-notes.md)
- [Lịch sử điểm](docs/leaderboard.md)
- [Repo baseline gốc](https://github.com/hugebenevolence/zalo-ai-challenge-datamate-26)
