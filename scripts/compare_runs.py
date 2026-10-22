"""Print a markdown table comparing metrics_*.json files (paste into the README).

Usage:
    python scripts/compare_runs.py outputs/qwen2.5-1.5b-finqa-lora/eval
"""
import json
import sys
from pathlib import Path

COLS = ["exact_match", "token_f1", "numeric_match", "seconds_per_example"]


def main(eval_dir: str):
    files = sorted(Path(eval_dir).glob("metrics_*.json"))
    if not files:
        sys.exit(f"No metrics_*.json in {eval_dir}")
    print("| Model | n | " + " | ".join(COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for f in files:
        m = json.loads(f.read_text())
        print(f"| {m['tag']} | {m['n']} | " + " | ".join(str(m.get(c, "")) for c in COLS) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
