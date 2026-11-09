# DataMate: challenge spec (Zalo AI Challenge 2026, track 2)

Source: https://challenge.zalo.ai/portal/datamate, fetched 2026-10-28 from
`api-challenge.zalo.ai/fe/api/config-detail?title=2`. The Data tab needs a login
and is not mirrored here. See the README.md shipped inside the data zips.

## Task

Answer everyday Vietnamese questions over one business's tabular data
(.csv / .xlsx / .sqlite + a short README.md), or say the data cannot answer.

Input per question (one line of `questions.jsonl`):

```json
{"id": "q_06de33fa21cc", "business_id": "biz_118", "question": "sp bán dc nhiều nhất tuần rồi là gì", "as_of_date": "2026-03-04"}
```

- `as_of_date` is "today" for relative time ("tháng này", "tuần rồi", "cùng kỳ năm ngoái").
- A rulebook and a solar-lunar calendar are shared by every business in the split.

Output per question (one line of `submission.jsonl`):

```json
{"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg"}
```

## Answer types and matching

An answer is correct only if `type` equals the reference type **and** the value matches.

| type | content | matching |
|---|---|---|
| number | integer (VND, counts, qty) or 2-decimal | exact |
| percent | e.g. `-12.45` | exact after rounding to 2 dp, sign must be right |
| string | name of product/customer/branch/weekday… | normalized exact: NFC → lowercase → non letter/digit/space → space → collapse spaces. Diacritics kept. Accepted forms = every spelling in that business's data **plus its code** (e.g. `sp0042`) |
| date | `yyyy-mm-dd`, or `yyyy-mm` for month questions | exact |
| boolean | `true` / `false` | exact |
| list | ordered names or numbers | position by position, string/number rule per element |
| table | `{"columns": [...], "rows": [[...], ...]}` | columns by normalized name, else by position, rows as a set |
| unanswerable | data does not contain what is needed | correct iff reference is unanswerable |
| abstain | agent declines | scores 0 regardless |

Ranking questions: use the criterion in the question, or the rulebook default,
descending, **valid orders only**. Ties broken by code ascending. Questions with a
tie at the k boundary (or at rank 1 for "… nhất") are excluded. Fewer than k
candidates → answer is all of them.

## Metric

Penalized Accuracy: `PA = (1/N) Σ s_i`, with `s_i = +1` correct, `0` abstain
(or missing), `−1` wrong. Any non-unanswerable answer to an unanswerable question
is wrong. Final-ranking ties broken by total inference time in the organizers'
re-run, then by earlier submission.

Consequence: answer only when P(correct) > 0.5. Abstaining is free. A wrong answer
costs as much as a right one earns.

## Rules that constrain the design

1. Open-weight models released before 2026-10-01, fine-tuning allowed.
2. **≤ 15B parameters total across every model used at inference** (LLM + embedder +
   reranker + classifier, MoE counted in full, loaded together or not). Any quantization.
3. Fully offline in Docker, networking disabled.
4. No human in the loop. No hard-coding keyed on question or business ids.
5. The agent may generate and execute code. `/data` is mounted read-only.
6. Train on the provided train data, own synthetic data, and public data. Never on the test sets.
7. Hardware: 2 × RTX 5090 (32 GB each), 512 GB RAM. **Whole private test in ≤ 6 h
   wall-clock, model loading included.** CUDA ≥ 12.8. Base image given in the Docker guideline.
8. Final score = organizers' re-run of the Docker image. Reproducible iff
   |PA(uploaded) - PA(re-run)| ≤ 0.03. The rule does **not** prescribe temperature or seeds. It
   only bounds the PA difference. Our uploaded file comes from our hardware and the re-run from
   2 × RTX 5090, so even greedy decoding is not bit-identical across them: measure run-to-run PA
   drift (e.g. different batch sizes) and keep it well under 0.03. Write logs. On timeout or crash, answers already written to
   `/result/submission.jsonl` count and the rest are abstain → write incrementally.

Docker I/O: test set mounted at `/data`, output collected from `/result/submission.jsonl`.

## From the general regulations (zalo-ai-challenge-regulations-en.pdf)

- §3: on Oct 26 the organizers release "the development kit, data (training & public test), and
  **evaluation software**". The evaluator, the Docker guideline and the log format referred to by
  Rule 8 are not on the public pages. They are presumably behind the login-only Data tab.
- §4.2: extra annotation is allowed on **train only**. No manual labels or human predictions on
  validation/test data to create a submission → the public test cannot be hand-labelled for local eval.
- §4.5: open-source code/libraries must carry an OSI-approved licence without commercial
  restrictions. For models, the track rule asks for a licence that permits the use. Apache-2.0 / MIT
  weights are the safe choice, custom community licences need checking.
- §5.1: max 5 submissions/day. §5.2: the frozen solution includes training code, testing code,
  model, **method description and run instructions**.
- §6.1: public-leaderboard ties go to the earlier submission. Final ties go to faster total
  inference time (track rule), then earlier submission.
- Teams ≤ 5, member changes locked from Nov 17.
- Prizes per track: 1st VND 90,412,000, 2nd VND 64,580,000.

## Timeline (GMT+7)

| date | event |
|---|---|
| Oct 26 | train + public test released |
| Oct 26 to Nov 16 | public leaderboard, 5 submissions/day |
| **Nov 16 23:59** | solution freeze: Docker image + training/inference code (Drive link + checksum, max 2 uploads, last counts) |
| Nov 20 09:00 | private test released |
| Nov 23 23:59 | private-test submission.jsonl |
| Dec 2 | top 2 per track announced |
| Dec 12 | final standing |
