from pathlib import Path

import pytest

from finqa_ft.config import load_config
from finqa_ft.data import IGNORE_INDEX, clean, normalize_record, read_jsonl, split_records, tokenize_example
from finqa_ft.prompts import SYSTEM_PROMPT, build_messages

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "financial_qa_sample.jsonl"


class FakeTokenizer:
    """Whitespace tokenizer with a ChatML-like template — enough to test masking."""

    eos_token = "<eos>"

    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = " ".join(f"<{m['role']}> {m['content']}" for m in messages)
        return text + (" <assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(w) for w in text.replace("<eos>", " <eos>").split()]}


def test_normalize_record_maps_alternate_keys():
    rec = normalize_record({"instruction": "What is X?", "input": "ctx", "output": "42"})
    assert rec == {"question": "What is X?", "context": "ctx", "answer": "42"}


def test_clean_drops_invalid_and_duplicates():
    rows = [
        {"question": "What was revenue?", "context": "c", "answer": "1"},
        {"question": "what was revenue? ", "context": "c", "answer": "1"},  # dup
        {"question": "Hi", "context": "c", "answer": "1"},                  # too short
        {"question": "No answer here?", "context": "c", "answer": ""},      # empty answer
    ]
    assert len(clean(rows)) == 1


def test_split_is_deterministic_and_disjoint():
    rows = [{"question": f"q{i}?", "context": "", "answer": str(i)} for i in range(100)]
    a = split_records(rows, 0.1, 0.2, seed=7)
    b = split_records(rows, 0.1, 0.2, seed=7)
    assert a == b
    assert (len(a["train"]), len(a["validation"]), len(a["test"])) == (70, 10, 20)
    ids = [r["answer"] for part in a.values() for r in part]
    assert len(ids) == len(set(ids)) == 100


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_records([], 0.6, 0.5)


def test_tokenize_masks_prompt_tokens():
    tok = FakeTokenizer()
    rec = {"question": "What was revenue?", "context": "Revenue was 5.", "answer": "Revenue was 5."}
    out = tokenize_example(rec, tok, max_length=512)
    n_answer = len("Revenue was 5.".split()) + 1  # + eos
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
    assert out["labels"][-n_answer:] == out["input_ids"][-n_answer:]
    assert all(lab == IGNORE_INDEX for lab in out["labels"][:-n_answer])


def test_tokenize_drops_overlong_examples():
    tok = FakeTokenizer()
    rec = {"question": "What was revenue?", "context": "word " * 500, "answer": "5"}
    assert tokenize_example(rec, tok, max_length=64) is None


def test_prompt_contains_context_and_system():
    msgs = build_messages("Q?", "some context", "A")
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "some context" in msgs[1]["content"] and msgs[2]["content"] == "A"


def test_sample_data_is_clean():
    rows = clean(read_jsonl(SAMPLE))
    assert len(rows) == 12


def test_repo_config_loads():
    cfg = load_config(ROOT / "configs" / "qlora_qwen2.5_1.5b.yaml")
    assert cfg.lora.r == 16 and cfg.model.load_in_4bit
