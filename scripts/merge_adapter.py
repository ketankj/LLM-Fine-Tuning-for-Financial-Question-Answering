"""Merge a LoRA adapter into the base weights for standalone deployment.

Usage:
    python scripts/merge_adapter.py --base Qwen/Qwen2.5-1.5B-Instruct \
        --adapter outputs/qwen2.5-1.5b-finqa-lora/adapter --out outputs/merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Merge into full-precision weights (merging into 4-bit weights loses accuracy).
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
