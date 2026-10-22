"""Answer generation shared by evaluation and the API."""
from __future__ import annotations

from .prompts import build_messages


class FinQAModel:
    def __init__(self, model, tokenizer, max_new_tokens: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @classmethod
    def load(cls, base_model: str, adapter_path: str | None = None, load_in_4bit: bool = True,
             max_new_tokens: int = 128) -> FinQAModel:
        from .modeling import load_for_inference

        model, tok = load_for_inference(base_model, adapter_path, load_in_4bit)
        return cls(model, tok, max_new_tokens)

    def answer(self, question: str, context: str | None = None, max_new_tokens: int | None = None) -> str:
        return self.answer_batch([question], [context], max_new_tokens)[0]

    def answer_batch(self, questions: list[str], contexts: list[str | None],
                     max_new_tokens: int | None = None) -> list[str]:
        import torch

        prompts = [
            self.tokenizer.apply_chat_template(
                build_messages(q, c), tokenize=False, add_generation_prompt=True
            )
            for q, c in zip(questions, contexts)
        ]
        # Left padding so every prompt ends right where generation starts.
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,  # greedy → deterministic, comparable eval
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tokenizer.batch_decode(gen, skip_special_tokens=True)]
