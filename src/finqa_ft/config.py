"""Typed config loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    max_seq_length: int = 1024


@dataclass
class LoraConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class DataConfig:
    hf_dataset: str = "virattt/financial-qa-10K"
    hf_split: str = "train"
    local_path: str | None = None
    val_fraction: float = 0.05
    test_fraction: float = 0.10
    max_samples: int | None = None
    seed: int = 42


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/finqa-lora"
    num_train_epochs: float = 2
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    report_to: str = "none"


@dataclass
class EvalConfig:
    max_new_tokens: int = 128
    num_samples: int | None = 200
    numeric_rel_tol: float = 0.01


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


def _build(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        model=_build(ModelConfig, raw.get("model")),
        lora=_build(LoraConfigSpec, raw.get("lora")),
        data=_build(DataConfig, raw.get("data")),
        training=_build(TrainingConfig, raw.get("training")),
        evaluation=_build(EvalConfig, raw.get("evaluation")),
    )
