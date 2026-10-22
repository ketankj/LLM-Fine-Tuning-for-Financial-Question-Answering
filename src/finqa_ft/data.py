"""Data loading, cleaning, splitting and tokenization.

Tokenization uses *completion-only loss*: prompt tokens get label -100 so the
model is only trained to produce the answer, not to reproduce the context.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import build_messages

QUESTION_KEYS = ("question", "query", "instruction")
CONTEXT_KEYS = ("context", "input", "passage", "text")
ANSWER_KEYS = ("answer", "output", "response")

IGNORE_INDEX = -100


# ---------------------------------------------------------------- loading
def _first(record: dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        v = record.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def normalize_record(record: dict[str, Any]) -> dict[str, str]:
    """Map heterogeneous column names onto question / context / answer."""
    return {
        "question": _first(record, QUESTION_KEYS),
        "context": _first(record, CONTEXT_KEYS),
        "answer": _first(record, ANSWER_KEYS),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_raw(hf_dataset: str | None, hf_split: str = "train", local_path: str | None = None):
    if local_path:
        return read_jsonl(local_path)
    from datasets import load_dataset  # lazy import: keeps tests lightweight

    return list(load_dataset(hf_dataset, split=hf_split))


# ---------------------------------------------------------------- cleaning
def is_valid(rec: dict[str, str], max_context_chars: int = 6000) -> bool:
    return (
        len(rec["question"]) >= 5
        and len(rec["answer"]) >= 1
        and len(rec["context"]) <= max_context_chars
    )


def _key(rec: dict[str, str]) -> str:
    text = (rec["question"].lower().strip() + "||" + rec["context"][:200].lower())
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean(records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize, drop invalid rows and exact duplicates."""
    seen, out = set(), []
    for r in records:
        rec = normalize_record(r)
        if not is_valid(rec):
            continue
        k = _key(rec)
        if k in seen:
            continue
        seen.add(k)
        out.append(rec)
    return out


def split_records(
    records: list[dict[str, str]],
    val_fraction: float = 0.05,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Deterministic shuffle + split. Test set is held out from training entirely."""
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be in [0, 1)")
    rows = list(records)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    return {
        "test": rows[:n_test],
        "validation": rows[n_test : n_test + n_val],
        "train": rows[n_test + n_val :],
    }


# ---------------------------------------------------------------- tokenizing
def tokenize_example(rec: dict[str, str], tokenizer, max_length: int) -> dict[str, list[int]] | None:
    """Build input_ids/labels with prompt tokens masked out.

    Returns None if the answer would be truncated (better to drop than to
    train on a cut-off target).
    """
    prompt_text = tokenizer.apply_chat_template(
        build_messages(rec["question"], rec["context"]),
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_text = rec["answer"].strip() + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    if len(prompt_ids) + len(answer_ids) > max_length:
        return None

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class CompletionCollator:
    """Right-pads a batch; label padding uses -100 so it is ignored by the loss."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
