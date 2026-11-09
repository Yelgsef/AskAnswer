# Lịch sử điểm public và cách đọc

Public = điểm cổng thi trả về cho `submission.jsonl` của một lượt chạy trên 1000 câu public (14 shop `dev_001`-`dev_014`).
Số câu đúng / sai ước lượng từ điểm và số câu đã trả lời: đúng = (trả lời + 1000 × PA) / 2. Không có nhãn public: điểm chỉ dùng
ở mức tổng, không bao giờ làm nhãn cho từng câu.

| Ngày  | Bản (file nộp)                 | Thay đổi                                                                                                         | Trả lời | Đúng / sai (ước lượng) | Public    |
| ----- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ------- | ---------------------- | --------- |
| 11-03 | LoRA v1                        | SQL trên DuckDB, LoRA v1 (rejection sampling trên train)                                                         | 736     | 582 / 154              | 0.428     |
| 11-04 | v2 (`sql_v2_public_debtroute`) | LoRA v2, few-shot từ ngân hàng ví dụ train, macro công nợ, định tuyến câu nợ                                     | 821     | 656 / 165              | 0.491     |
| 11-04 | `sub_qty_unans`                | `qty` của view theo đơn vị cơ bản, trả lời `unanswerable` khi đa số                                              |         | 675 / 159              | 0.516     |
| 11-05 | `sub_v4_final`                 | Sinh mẫu mới, chuẩn hóa cách viết tên, hệ số đơn vị đọc từ giá                                                   | 837     | 719 / 118              | 0.60      |
| 11-05 | `sub_v6`                       | Mẫu thường bắt đầu bằng ghi chú ngắn trước SQL. Ví dụ macro công nợ trong ngân hàng                              | 875     | 752 / 122              | 0.63      |
| 11-05 | `sub_v7`                       | v6 + ẩn bảng thô (`--hide-raw`): bỏ                                                                              |         |                        | 0.623     |
| 11-06 | `sub_v8`                       | LoRA v3 (2910 dòng, có 26 câu do Qwen3.8-27B giải làm teacher, chỉ trên train). Kiểm tra view theo giờ và đơn vị | 889     | 764 / 125              | 0.639     |
| 11-06 | **`sub_v9`**                   | v8 + nhánh `sql-hints`, bỏ 2 phần chỉ khớp public. Lớp công nợ dev_003 soạn lại                                  | 902     | 792 / 110              | **0.683** |

## Đọc các bước tăng

- **v6 → v8 (+0.9 điểm):** LoRA v3 gần như không đổi hành vi trên public: 105 câu khác v6, ròng khoảng +12 đúng / +3 sai. Dữ liệu
  SFT v3 phần lớn giống v2, và phần khó của public (4 ngành không có trong train, từ địa phương, viết tắt tiền) nằm ngoài train.
- **v8 → v9 (+4.4 điểm):** chỉ khoảng 90 câu khác v8 nhưng ròng +28 đúng / -15 sai. Phần lớn đến từ gợi ý theo dạng câu hỏi (mỗi gợi ý
  là một luật trong rulebook, đã đối chiếu với đáp án train) và lớp công nợ mới của dev_003 (lớp cũ báo 26.288 đơn còn nợ).
- **v9 so với 0.696 (-1.3 điểm):** hai phần bỏ có chủ đích (danh sách từ đồng nghĩa lấy từ tên món public, gợi ý dòng sản phẩm. Hai phần này chạm
  khoảng 10 câu public, không kiểm chứng được trên train), LoRA v3 thay v2, view khác, và dao động giữa các lần chạy.

## Holdout không phải cổng

`splits/holdout600.txt` (632 câu train của 6 shop) đo cao hơn public một khoảng lớn và không đều: LoRA v3 đã học 724 dòng của
chính các câu này, và các shop public có 4 ngành không có trong train (210/1000 câu). Dùng holdout như smoke test (code chạy,
không tụt mạnh). Thay đổi thật được đánh giá bằng lượt nộp public, so cùng câu với bản trước bằng các cờ không cần nhãn
(`analysis/flags.py`, `notebooks/03_evaluation.ipynb`).

## Rủi ro đang mở

- Mọi điểm public ở trên dùng view và lớp công nợ đã lưu từ các lượt trước (`--views-from`). Bản Docker cho private soạn lại
  tất cả: `FRESH=1 bash scripts/run_public.sh` đo điểm khi soạn mới.
- View của dev_014 không cố định giữa các lần chạy: số khách khác nhau (`customer_key`) đổi 2292-2299 giữa hai lần dựng cùng
  một câu SQL, do code nạp dữ liệu sẵn có, không do bước tách module. Ảnh hưởng các câu về khách hàng của shop này.
- Câu về tồn kho theo lô (khoảng 31 câu public, chủ yếu dev_006) chưa có lớp riêng.
