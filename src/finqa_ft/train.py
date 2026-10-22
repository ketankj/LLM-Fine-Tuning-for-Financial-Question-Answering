"""QLoRA supervised fine-tuning.

Usage:
    python -m finqa_ft.train --config configs/qlora_qwen2.5_1.5b.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .data import CompletionCollator, clean, load_raw, split_records, tokenize_example, write_jsonl

log = logging.getLogger("finqa_ft.train")


def prepare_splits(cfg) -> dict[str, list[dict]]:
    raw = load_raw(cfg.data.hf_dataset, cfg.data.hf_split, cfg.data.local_path)
    rows = clean(raw)
    if cfg.data.max_samples:
        rows = rows[: cfg.data.max_samples]
    splits = split_records(rows, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.seed)
    log.info("Rows after cleaning: %d | %s", len(rows), {k: len(v) for k, v in splits.items()})
    return splits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=-1, help="override for smoke tests")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data
    splits = prepare_splits(cfg)
    for name, rows in splits.items():  # persist exact splits for reproducible eval
        write_jsonl(rows, out_dir / "splits" / f"{name}.jsonl")

    from .modeling import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(cfg.model.base_model)

    def to_dataset(rows):
        toks = [tokenize_example(r, tokenizer, cfg.model.max_seq_length) for r in rows]
        kept = [t for t in toks if t is not None]
        dropped = len(rows) - len(kept)
        log.info("Tokenized %d/%d rows (dropped %d over max_seq_length)", len(kept), len(rows), dropped)
        return Dataset.from_list(kept)

    train_ds = to_dataset(splits["train"])
    eval_ds = to_dataset(splits["validation"]) if splits["validation"] else None

    # ---- model
    model = load_base_model(cfg.model.base_model, cfg.model.load_in_4bit)
    if cfg.model.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    t = cfg.training
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=t.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        weight_decay=t.weight_decay,
        logging_steps=t.logging_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=t.gradient_checkpointing,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to=[] if t.report_to == "none" else [t.report_to],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CompletionCollator(pad_token_id=tokenizer.pad_token_id),
    )
    result = trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    summary = {"train": result.metrics}
    if eval_ds is not None:
        summary["eval"] = trainer.evaluate()
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved adapter to %s", adapter_dir)


if __name__ == "__main__":
    main()
