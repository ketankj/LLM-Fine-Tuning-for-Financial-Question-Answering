"""Evaluation metrics for financial QA.

Plain exact match is too strict for generated text ("$4.2 billion" vs
"4.2 billion dollars"), so we report three complementary metrics:

* exact_match   – normalized string equality
* token_f1      – SQuAD-style bag-of-tokens overlap
* numeric_match – every number in the reference appears in the prediction
                  (within a relative tolerance, with scale words like
                  "million"/"billion" and % handled)
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_NUM = re.compile(
    r"(?P<neg>\(|-)?\$?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\)?"
    r"\s*(?P<pct>%|percent)?\s*(?P<scale>thousand|million|billion|trillion)?",
    re.IGNORECASE,
)


def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {"."})
    s = _ARTICLES.sub(" ", s)
    s = s.replace(". ", " ").rstrip(".")
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize_text(pred).split(), normalize_text(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_numbers(s: str) -> list[float]:
    """Pull numeric values out of text, applying sign and scale words."""
    values = []
    for m in _NUM.finditer(s):
        v = float(m.group("num").replace(",", ""))
        scale = m.group("scale")
        if scale:
            v *= _SCALES[scale.lower()]
        if m.group("neg"):
            v = -v
        values.append(v)
    return values


def _close(a: float, b: float, rel_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def numeric_match(pred: str, ref: str, rel_tol: float = 0.01) -> float | None:
    """1.0 if every reference number is found in the prediction.

    Returns None when the reference has no numbers (metric not applicable),
    so it doesn't dilute the average.
    """
    ref_nums = extract_numbers(ref)
    if not ref_nums:
        return None
    pred_nums = extract_numbers(pred)
    for rn in ref_nums:
        if not any(_close(abs(rn), abs(pn), rel_tol) for pn in pred_nums):
            return 0.0
    return 1.0


def score(preds: Iterable[str], refs: Iterable[str], rel_tol: float = 0.01) -> dict[str, float]:
    preds, refs = list(preds), list(refs)
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    if not preds:
        return {"n": 0, "exact_match": 0.0, "token_f1": 0.0, "numeric_match": 0.0, "numeric_n": 0}
    em = [exact_match(p, r) for p, r in zip(preds, refs)]
    f1 = [token_f1(p, r) for p, r in zip(preds, refs)]
    nm = [x for x in (numeric_match(p, r, rel_tol) for p, r in zip(preds, refs)) if x is not None]
    return {
        "n": len(preds),
        "exact_match": round(sum(em) / len(em), 4),
        "token_f1": round(sum(f1) / len(f1), 4),
        "numeric_match": round(sum(nm) / len(nm), 4) if nm else 0.0,
        "numeric_n": len(nm),
    }
