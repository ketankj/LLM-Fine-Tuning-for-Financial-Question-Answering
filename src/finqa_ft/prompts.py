"""Prompt template shared by training, evaluation and serving.

Keeping one template in one place guarantees the model sees the exact same
format at train time and at inference time (train/serve skew is a common
cause of fine-tuned models underperforming in production).
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question using only the "
    "provided context from the company's filing. Be concise and precise. Keep "
    "numbers, units and periods exactly as stated. If the context does not "
    "contain the answer, reply: \"The context does not provide this information.\""
)


def build_user_message(question: str, context: str | None) -> str:
    context = (context or "").strip()
    if context:
        return f"Context:\n{context}\n\nQuestion: {question.strip()}"
    return f"Question: {question.strip()}"


def build_messages(question: str, context: str | None, answer: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, context)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages
