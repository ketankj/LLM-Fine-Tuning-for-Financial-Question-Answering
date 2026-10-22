"""Model / tokenizer loading (4-bit QLoRA or full precision)."""
from __future__ import annotations


def _compute_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_name: str, load_in_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {"torch_dtype": _compute_dtype()}
    if load_in_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_compute_dtype(),
        )
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_for_inference(model_name: str, adapter_path: str | None = None, load_in_4bit: bool = True):
    """Base model, optionally with a trained LoRA adapter attached."""
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, load_in_4bit=load_in_4bit)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer
