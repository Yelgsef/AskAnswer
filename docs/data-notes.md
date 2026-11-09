# Data notes (from the 2026-10-26 release)

Layout after unzip into `data/` (gitignored, original zips in `data/raw/`):

```
data/train/        questions.jsonl (3000)  answers.jsonl  rulebook.md  calendar.csv  data/biz_101..biz_130
data/public_test/  questions.jsonl (1000)                 rulebook.md  calendar.csv  data/dev_001..dev_014
```

`answers.jsonl` rows: `{"id", "type", "answer", "alias"}`. `alias` is non-empty for 167 rows
(other accepted spellings). Unanswerable rows have `answer: null`.

## Splits

- Businesses are disjoint between splits (biz_* vs dev_*). Expect private test to bring new ones.
- **The rulebook differs per split.** The public rulebook adds rules (loyal / big-spender
  customer, peak hour, main supplier, topping in the same cell…) and whole industry sections
  absent from train (Mẹ và bé, Sửa xe, Cửa hàng điện thoại). Read the rulebook at runtime.
  Do not hard-code train rules.
- `calendar.csv` is identical across splits.
- Questions per business: ~100 in train, 21-129 in public.

## Train answer types

| type | n | share |
|---|---|---|
| number | 1884 | 62.8% |
| percent | 578 | 19.3% |
| list | 193 | 6.4% |
| date | 145 | 4.8% |
| string | 88 | 2.9% |
| unanswerable | 70 | 2.3% |
| boolean | 23 | 0.8% |
| table | 19 | 0.6% |

## Files

Per business ≤ 24 MB, 1-25 files: xlsx (176), csv (30), sqlite (2). Everything fits in memory.
Each business README maps every column to a role phrase from a small shared vocabulary
("mã đơn", "ngày bán", "trạng thái đơn", "thành tiền sau giảm", "đơn vị cơ bản", …), and
states the last data date and unit conversions. That vocabulary is the natural schema-mapping key.

Messiness seen in one hand-kept ledger (`dev_004/SoBanHang.xlsx`), all of which the rulebook §8
tells you how to read:

- two header rows (group header above column header). One sheet per month (`T04-2025`)
- money as `55000`, `'55.000đ'`, `'55k'`, `'2,825tr'`, `'1tr312'`
- quantity as `2` or `'1 bình'`, dates as `'01/04/2025'` or `'1/4'` (year = sheet year)
- order-level cells filled only on the first line of an order (forward-fill within order)
- exact duplicate lines inside one order (export error → count once)
- status spelled many ways (`Hoàn thành`, `Đã hoàn thành`, `đã xong`), walk-in customer as
  `lẻ`, `kl`, `Khách lẻ`

## Questions

- About 26% have no diacritics in both splits. Public test adds heavier teencode and emoji
  ("kó nhìu", "cko a xin", "o đờ", "🤔").
- 15 question strings appear in both train and public (template overlap only, different businesses).

## EDA findings (notebooks/01_data.ipynb Part B, 2026-10-28, data cards added 2026-10-30)

**Train does not cover the public test's industries.** Train has no phone, motorbike-repair or
mother-&-baby shop, its rulebook has only the Cà phê, Nhà thuốc, VLXD sections. Four public shops
(dev_011 and dev_014 phone, dev_012 repair, dev_013 mẹ và bé) are new industries: 210 / 1000 public
questions. Expect validation on train to overestimate public PA on that slice.

**Rulebook drift:** train 70 rules, public 92: +25 added (new rankings like "khách sộp nhất",
"giờ cao điểm", "Sổ không có cột trạng thái", 4 industry sections), 3 removed, 3 reworded.

**Questions**
- Answer type follows the question shape closely: percent ≈ "% / tỉ lệ / chiếm" cue (573/578).
  Date/string/table answers almost always come with a ranking cue ("nhất", "top", "liệt kê").
- 95.8% of train questions (90.9% public) contain a time expression. Explicit month ("th3/2026",
  "5/25", "1-2026") is the most common. Public has fewer item-ranking questions, more
  customer/debt/staff/count-with-threshold questions than train.
- as_of_date equals the shop's last data date for 79% (train) / 75% (public) of questions,
  never after it.

**Answers (train)**
- Every `number` is an integer, 49% are ≥ 1e6 (VND). 23.5% of percents are negative.
- Lists have length 2, 3 or 5. 45% are numeric (multi-quantity questions → list in question order).
- `alias` is filled only for list / string / table answers.

**Unanswerable (70 train rows, hand-sorted)**
- ~1/3 period fully outside the data range, including "vs previous month" asked on the first
  month (spot-checked: biz_121 starts 03/2025, biz_106 02/2025, biz_103 04/2025).
- ~1/2 the shop does not record that kind of information (street/address, gender, stock,
  purchases, debt, sales channel, payment method, cancel/return reason).
- ~1/10 subjective or outside knowledge (skill, politeness, drug dosage).
All three are checkable before answering: data date range, field availability, subjectivity.

**Files**
- 7 of 44 READMEs describe **no columns** (biz_106, biz_109, biz_123, biz_124, dev_003, dev_006,
  dev_014): POS / sqlite exports with conventions only. README-role mapping cannot be the only
  schema mapping.
- Where README describes columns, the header row is found with 100% column coverage. The header row
  index is 0 (213 tables), 1 (80), 2 (37), 3 (40).
- 80 / 260 order tables leave the order id blank on follow-up lines (fill-down). 96 / 260 contain
  exact duplicate rows. That count is taken **before** fill-down, so it is inflated: follow-up lines
  of two different orders with the same item look identical while the id is blank. The rulebook
  says duplicates within one order count once and identical lines of different orders count
  twice, so **fill down first, then dedupe within order**. The reverse order silently drops sales.
- Order status has 34 spellings (Vietnamese and English: "xong", "ok", "completed", "da lay",
  "khong giao duoc", "hang tra ve", "shipping"...) mapping to 4 states. Walk-in customer:
  "khách lẻ", "kl", "lẻ".
- Trap: ISO datetimes ('2025-02-01 06:03:42') parsed with `dayfirst=True` swap day and month.

**Data cards (01_data §B10, normalized data)**
- README quality: train 26 full / 4 none. Public 8 full / 3 partial / 3 none.
- Sale dates from parsed columns: every shop's last sale date equals the README's "đến hết" date.
  First sale dates range 2025-01-01 to 2025-07-01 (repair shop dev_012 starts in July 2025).
- Share of shops recording each entity (column names + README meanings), public vs train: staff /
  technicians 21% vs 3%, sales channel 36% vs 20%, price history 43% vs 27%, purchases 43% vs 30%,
  returns 50% vs 37%, warranty 7% vs 0%. Fewer product catalogs (64% vs 93%), inventory (43% vs 63%),
  unit conversions (57% vs 70%), batches / expiry (7% vs 17%). Questions about staff, channels and
  price changes have little train coverage.
- Fill-down needs tolerance: in dev_004, 8,914 orders had a few stray names on follow-up lines,
  which blocked a "blank on every follow-up line" rule. The fill now applies to columns blank on
  ≥ 90% of follow-up lines.

