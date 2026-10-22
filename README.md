# LLM Fine-Tuning for Financial Question Answering

QLoRA fine-tuning of a small open LLM (**Qwen2.5-1.5B-Instruct**) to answer questions about
company financial filings, grounded in a provided context passage. Includes a reproducible
data pipeline, completion-only training, a base-vs-fine-tuned evaluation harness with
finance-aware metrics, and a FastAPI inference service.

```
 HF dataset / JSONL ──► clean + dedupe ──► deterministic train/val/test split
                                               │
                         chat-template prompt + completion-only labels
                                               │
                 4-bit NF4 base model + LoRA adapters (PEFT) ──► Trainer
                                               │
                    adapter/ ──► evaluate (EM, token-F1, numeric match)
                                               │
                                 FastAPI /ask  (or merge → standalone model)
```

## Why this design

| Decision | Reason |
|---|---|
| **QLoRA (4-bit NF4 + LoRA r=16)** | Trains ~1% of parameters; fits on one 16 GB GPU (free Colab T4) instead of needing multi-GPU full fine-tuning. |
| **Completion-only loss** | Prompt/context tokens are masked to `-100`, so the model learns to *answer*, not to memorize filing text. |
| **One prompt template** (`prompts.py`) | Training, evaluation and serving use identical formatting → no train/serve skew. |
| **Drop over-length examples** instead of truncating | A truncated answer is a corrupted training target. |
| **Numeric-match metric** | Exact match is too strict for finance ("$4,200 million" = "$4.2 billion"). Numbers are parsed with scale words, commas, %, and parenthetical negatives, compared at 1% tolerance. |
| **Held-out test split saved to disk** | Base and fine-tuned models are scored on the exact same rows. |
| **Abstention target** | System prompt teaches a fixed reply when the context lacks the answer, reducing hallucinated figures. |

## Project structure

```
configs/qlora_qwen2.5_1.5b.yaml   all hyperparameters in one place
data/sample/                       12 fictional examples for offline smoke tests
src/finqa_ft/
  config.py      typed YAML config
  prompts.py     system prompt + chat message builder
  data.py        load, clean, dedupe, split, tokenize (label masking), collator
  modeling.py    4-bit / full-precision model loading, adapter attach
  train.py       QLoRA training entry point
  inference.py   batched greedy generation
  evaluate.py    scores a model on the test split, writes predictions + metrics
  metrics.py     exact match, token F1, numeric match
app/main.py      FastAPI service (/health, /ask)
scripts/         merge_adapter.py, compare_runs.py
tests/           unit tests (no GPU needed) — run in GitHub Actions
```

## Quickstart

### 1. Train on a GPU (Google Colab is easiest)

Runtime → Change runtime type → **T4 GPU**, then:

```bash
!git clone https://github.com/ketankj/finqa-llm-finetune.git
%cd finqa-llm-finetune
!pip install -q -r requirements.txt && pip install -q -e .

# Smoke run (~2 min): proves the pipeline end-to-end
!python -m finqa_ft.train --config configs/qlora_qwen2.5_1.5b.yaml --max-steps 20

# Full run
!python -m finqa_ft.train --config configs/qlora_qwen2.5_1.5b.yaml
```

To train on your own data, set `data.local_path` to a `.jsonl` file with
`question`, `context`, `answer` fields.

### 2. Evaluate base vs fine-tuned

```bash
python -m finqa_ft.evaluate --config configs/qlora_qwen2.5_1.5b.yaml --tag base
python -m finqa_ft.evaluate --config configs/qlora_qwen2.5_1.5b.yaml \
    --adapter outputs/qwen2.5-1.5b-finqa-lora/adapter --tag finetuned
python scripts/compare_runs.py outputs/qwen2.5-1.5b-finqa-lora/eval
```

### 3. Serve

```bash
export FINQA_ADAPTER=outputs/qwen2.5-1.5b-finqa-lora/adapter
uvicorn app.main:app --port 8000

curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{
  "question": "What was free cash flow?",
  "context": "Operating cash flow was $1,240 million and capex was $910 million, resulting in free cash flow of $330 million."
}'
```

Docker: `docker build -t finqa . && docker run -p 8000:8000 -v $PWD/outputs/qwen2.5-1.5b-finqa-lora/adapter:/models/adapter finqa`

### 4. Tests (CPU only)

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Results

> Fill this in from `scripts/compare_runs.py` after your run. Do not publish numbers you haven't produced.

| Model | n | Exact match | Token F1 | Numeric match | s / example |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | | | | | |
| + QLoRA fine-tune | | | | | |

Training details: dataset size after cleaning, epochs, GPU, wall-clock time, final train/eval loss
(from `outputs/.../train_summary.json`).

## Limitations & next steps

- Single-passage QA; no multi-hop reasoning across tables (FinQA/TAT-QA style program generation is a natural extension).
- Metrics are lexical/numeric; an LLM-as-judge pass would capture paraphrase-correct answers better.
- Pair with retrieval (RAG over full 10-K filings) so the model receives the right context at inference time.
- Push the adapter to the Hugging Face Hub instead of committing weights to git.

## License

MIT
