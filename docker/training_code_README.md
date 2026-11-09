# DataMate: training code and evidence

The inference adapter `/code/models/lora` is LoRA v3, the last of three QLoRA rounds on Qwen3.5-9B. Each round continues from
the previous adapter: v1, then v2, then v3. All training data come from the provided **train** split. No public or private
test data was used for training, and no test answers were collected.

## What is here

| Path | Content |
| --- | --- |
| `train.sh` | Retrains v1, v2 and v3 from the packaged SFT data, with the exact settings below |
| `requirements.txt` | Pinned training stack. The image installs it in `/opt/venv-train`, plus `fla-core==0.6.0 --no-deps` |
| `src/` | `train_lora.py` (Unsloth QLoRA, loss on the reply only), `check_fla.py` (fla kernel vs PyTorch reference), `merge_lora.py`, `no_fla/` |
| `data_generation/` | `build_sft.py` (selects correct samples as SFT rows), `add_notes.py` (notes before the SQL), `build_fewshot_bank.py`, `build_pool.py`, and the `datamate` package that produced the samples |
| `configs/` | `flags.sh` (sampling settings of the generation runs, the same as inference), `sql_rft_v{1,2,3}.json` (training settings), `splits/` |
| `data/competition_train/` | The provided train split, unchanged: 30 businesses, 3,000 questions with answers, rulebook, calendar |
| `data/sft/` | The SFT data of each round. Every row holds the full prompt (`system`, `user`), the reply and the thinking flag |
| `data/fewshot_bank.jsonl` | The example bank used at inference (built from train questions only) |
| `logs/` | Training logs and `trainer_state.json` (loss per step) of the three rounds |
| `checkpoints/` | Final adapters v1 and v2, to check the chain. The v3 adapter is `/code/models/lora` |
| `data_manifest.json` | Every file with its size and SHA-256, dataset counts, and the sources and checksums of the base and generator models |

## How the data was made

1. **Sampling on train.** We ran the inference pipeline (`data_generation/datamate`, settings in `configs/flags.sh`, with `--save-prompts`)
   on train questions, with Qwen3.5-9B and the adapter of the previous round. This gives 8 plain samples per question, plus 4
   thinking-mode samples on hard and debt questions.
   - v1 and v2 sampled the 24 shops outside the holdout (`configs/splits/rft_train24.txt`).
   - v3 added all 30 train shops, including 390 questions of the 6 holdout shops.
2. **Rejection sampling.** `build_sft.py` keeps only the samples whose executed answer matches `train/answers.jsonl` (scored with
   `datamate/scoring.py`). Easy questions (all samples right) are capped, and each question keeps its solutions on the best route that
   reached the gold answer (debt macros, then canonical views, then raw tables).
3. **Teacher, train only.** For 26 train questions that the 9B never solved, a local open model generated the replies: avyukth/Qwen3.8-27B-AWQ-INT4
   (Apache-2.0), run through the same pipeline on train. Those replies were kept only when they matched the train answer. This model
   is not used at inference.
4. **Notes.** `add_notes.py` adds short notes (period, filters, measure) before the SQL of the plain rows, so the replies look like the
   inference format (`--notes`). The rows are stored as generated, with prompts and replies, because sampling is not bit-for-bit
   reproducible across GPUs.

| Round | Data file | Rows (file) | Rows trained | Questions | Thinking rows | Init | max-len | lr | Steps | Train loss | Time |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | `data/sft/sql_rft_v1.jsonl` | 1,800 | 1,636 | 1,800 (24 shops) | 0 | Qwen3.5-9B | 16,384 | 1e-4 | 205 | 0.118 | 3.8 h |
| v2 | `data/sft/sql_rft_v2.jsonl` | 2,400 | 2,398 | 1,266 (24 shops) | 818 | v1 | 20,000 | 5e-5 | 300 | 0.176 | 6.9 h |
| v3 | `data/sft/sql_rft_v3_notes.jsonl` | 2,910 | 2,909 | 1,765 (30 shops) | 1,138 | v2 | 32,768 | 5e-5 | 364 | 0.181 | 8.2 h |

Rows longer than max-len were dropped by `train_lora.py`, which is why the trained counts are smaller. `sql_rft_v3.jsonl` is
the v3 data before the notes pass.

**Shared settings.** QLoRA (4-bit base), r 16, alpha 32, dropout 0 on attention and MLP projections (43,278,336 parameters).
1 epoch, batch 1, gradient accumulation 8, cosine schedule with 10 warmup steps, adamw_8bit, weight decay 0, bf16, seed 3407.
The prompts are rendered with the model's own chat template, exactly as vLLM receives them at inference.

**Validation.** `configs/splits/holdout600.txt` holds 632 questions from 6 train shops that v1 and v2 never saw. v3 was trained on
390 of them, so for v3 the holdout was only a quick check. The released configuration was chosen on the public leaderboard (aggregate score only).

## Reproduce

```bash
# inside the image (GPU, network not needed: the base model is /code/models/Qwen3.5-9B)
bash /code/training_code/train.sh                 # writes /tmp/lora/sql_rft_v{1,2,3}/adapter
# rebuild SFT rows from your own sampling runs on train (needs vLLM, which the image has):
cd /code/training_code && PYTHONPATH=data_generation:/code/src python3 data_generation/build_sft.py RUN [RUN ...] \
    --fewshot data/fewshot_bank.jsonl --out /tmp/sql_rft.jsonl --easy 800 --max 4000
PYTHONPATH=data_generation:/code/src python3 data_generation/add_notes.py /tmp/sql_rft.jsonl /tmp/sql_rft_notes.jsonl
```

**Hardware and nondeterminism.**
- All rounds ran on 1x NVIDIA RTX PRO 4000 Blackwell 24 GB. Peak memory was 15.6 GiB for v3.
- v3 used the fla Gated DeltaNet kernels (fla-core 0.6.0, checked by `check_fla.py`) and `--completion-logits`. Its throughput,
  1,301 tokens/s, matches that speed check. The flag computes the same loss, only on fewer positions.
- GPU kernels and 4-bit quantization make a retrained adapter close to the packaged one but not bit-identical.
