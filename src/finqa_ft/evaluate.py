"""Evaluate base vs fine-tuned model on the held-out test split.

Usage:
    # fine-tuned
    python -m finqa_ft.evaluate --config configs/qlora_qwen2.5_1.5b.yaml \
        --adapter outputs/qwen2.5-1.5b-finqa-lora/adapter --tag finetuned
    # baseline (same prompts, no adapter)
    python -m finqa_ft.evaluate --config configs/qlora_qwen2.5_1.5b.yaml --tag base
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .config import load_config
from .data import read_jsonl, write_jsonl
from .metrics import exact_match, numeric_match, score, token_f1

log = logging.getLogger("finqa_ft.evaluate")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter; omit for base model")
    ap.add_argument("--test-file", default=None, help="defaults to <output_dir>/splits/test.jsonl")
    ap.add_argument("--tag", default=None, help="name for this run, e.g. base / finetuned")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    out_dir = Path(cfg.training.output_dir)
    test_file = Path(args.test_file or out_dir / "splits" / "test.jsonl")
    rows = read_jsonl(test_file)
    if cfg.evaluation.num_samples:
        rows = rows[: cfg.evaluation.num_samples]
    tag = args.tag or ("finetuned" if args.adapter else "base")
    log.info("Evaluating %s on %d rows from %s", tag, len(rows), test_file)

    from .inference import FinQAModel

    model = FinQAModel.load(cfg.model.base_model, args.adapter, cfg.model.load_in_4bit,
                            cfg.evaluation.max_new_tokens)

    preds, start = [], time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        preds += model.answer_batch([r["question"] for r in chunk], [r["context"] for r in chunk])
        log.info("%d/%d", len(preds), len(rows))
    elapsed = time.time() - start

    refs = [r["answer"] for r in rows]
    tol = cfg.evaluation.numeric_rel_tol
    metrics = score(preds, refs, rel_tol=tol)
    per_example = round(elapsed / max(len(rows), 1), 3)
    metrics.update({"tag": tag, "adapter": args.adapter, "seconds_per_example": per_example})

    eval_dir = out_dir / "eval"
    write_jsonl(
        [
            {**r, "prediction": p, "exact_match": exact_match(p, r["answer"]),
             "token_f1": round(token_f1(p, r["answer"]), 4),
             "numeric_match": numeric_match(p, r["answer"], tol)}
            for r, p in zip(rows, preds)
        ],
        eval_dir / f"predictions_{tag}.jsonl",
    )
    (eval_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
