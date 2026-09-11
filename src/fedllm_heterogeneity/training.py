"""Optional Transformers/PEFT training helpers for controlled experiments."""

from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .lora import TensorState, apply_base_residual, effective_lora_delta
from .types import CanonicalExample, ClientUpdate


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    quantization_bits: int | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    ffa_lora: bool = False
    random_init: bool = False

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.lora_rank


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 2e-4
    batch_size: int = 4
    eval_batch_size: int = 8
    max_steps: int = 10
    max_length: int = 512
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class TrainResult:
    update: ClientUpdate
    mean_loss: float


def seed_everything(seed: int) -> None:
    """Seed model initialization, data order, and dropout reproducibly."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def require_hf() -> tuple[object, object, object, object, object]:
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install training dependencies with pip install -e '.[train]'") from exc
    return (
        (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig),
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )


def build_model_and_tokenizer(config: ModelConfig, device: str | None = None):
    hf, LoraConfig, get_peft_model, _, _ = require_hf()
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = hf
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, revision=config.revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    load_kwargs: dict[str, object] = {}
    if config.quantization_bits == 4:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    if config.random_init:
        from transformers import AutoConfig

        base_config = AutoConfig.from_pretrained(
            config.model_id, revision=config.revision
        )
        base = AutoModelForCausalLM.from_config(base_config)
        if device:
            base.to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            config.model_id, revision=config.revision, **load_kwargs
        )
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(base, lora_config)
    if config.ffa_lora:
        for name, parameter in model.named_parameters():
            if "lora_A" in name:
                parameter.requires_grad_(False)
    if device and config.quantization_bits is None:
        model.to(device)
    return model, tokenizer


def get_adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    _, _, _, get_state, _ = require_hf()
    return {
        key: value.detach().cpu().clone()
        for key, value in get_state(model).items()
    }


def set_adapter_state(model: torch.nn.Module, state: TensorState) -> None:
    _, _, _, _, set_state = require_hf()
    device = next(model.parameters()).device
    set_state(model, {key: value.to(device=device) for key, value in state.items()})


def _chat_prompt(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, ValueError, TypeError):
        return f"### Instruction:\n{prompt}\n\n### Response:\n"


def tokenize_example(tokenizer, example: CanonicalExample, max_length: int) -> dict[str, list[int]]:
    prompt_ids = tokenizer.encode(
        _chat_prompt(tokenizer, example.prompt), add_special_tokens=False
    )
    target_ids = tokenizer.encode(example.target, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError(f"Prompt truncation removed target for {example.example_id}")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class TokenizedExamples(Dataset):
    def __init__(self, examples: Sequence[CanonicalExample], tokenizer, max_length: int):
        self.rows = [tokenize_example(tokenizer, item, max_length) for item in examples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def make_collator(pad_token_id: int):
    def collate(rows: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        maximum = max(len(row["input_ids"]) for row in rows)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            padding = maximum - len(row["input_ids"])
            result["input_ids"].append(list(row["input_ids"]) + [pad_token_id] * padding)
            result["attention_mask"].append(list(row["attention_mask"]) + [0] * padding)
            result["labels"].append(list(row["labels"]) + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}

    return collate


def _device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def train_client(
    model: torch.nn.Module,
    tokenizer,
    global_adapter: TensorState,
    examples: Sequence[CanonicalExample],
    client_id: str,
    round_id: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> TrainResult:
    if not examples:
        raise ValueError(f"Client {client_id} has no training examples")
    local_seed = train_config.seed + round_id
    # Reset stochastic layers per client so adding a diagnostic evaluation does
    # not silently change subsequent training randomness.
    seed_everything(local_seed)
    set_adapter_state(model, global_adapter)
    model.train()
    generator = torch.Generator().manual_seed(local_seed)
    dataset = TokenizedExamples(examples, tokenizer, train_config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=make_collator(tokenizer.pad_token_id),
    )
    iterator = itertools.cycle(loader)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    losses: list[float] = []
    input_tokens = 0
    target_tokens = 0
    for _ in range(train_config.max_steps):
        batch = {key: value.to(_device(model)) for key, value in next(iterator).items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            train_config.gradient_clip_norm,
        )
        optimizer.step()
        losses.append(float(output.loss.detach().cpu()))
        input_tokens += int(batch["attention_mask"].sum().item())
        target_tokens += int((batch["labels"] != -100).sum().item())

    state = get_adapter_state(model)
    update = ClientUpdate(
        round_id=round_id,
        client_id=client_id,
        num_examples=len(examples),
        num_input_tokens=input_tokens,
        num_target_tokens=target_tokens,
        adapter_state=state,
        effective_delta=effective_lora_delta(state, model_config.scaling),
        metrics={"train_loss": float(sum(losses) / len(losses))},
    )
    return TrainResult(update=update, mean_loss=update.metrics["train_loss"])


@torch.no_grad()
def evaluate_nll(
    model: torch.nn.Module,
    tokenizer,
    adapter_state: TensorState,
    examples: Sequence[CanonicalExample],
    max_length: int = 512,
    batch_size: int = 8,
) -> float:
    set_adapter_state(model, adapter_state)
    model.eval()
    dataset = TokenizedExamples(examples, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collator(tokenizer.pad_token_id),
    )
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(_device(model)) for key, value in batch.items()}
        output = model(**batch)
        count = int((batch["labels"] != -100).sum().item())
        total_loss += float(output.loss.detach().cpu()) * count
        total_tokens += count
    return total_loss / max(total_tokens, 1)


def common_checkpoint_gradient(
    model: torch.nn.Module,
    tokenizer,
    adapter_state: TensorState,
    examples: Sequence[CanonicalExample],
    max_length: int = 512,
    microbatch_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Token-mean gradient at a shared checkpoint, accumulated in microbatches."""

    set_adapter_state(model, adapter_state)
    model.train()
    dataset = TokenizedExamples(examples, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=microbatch_size or len(dataset),
        shuffle=False,
        collate_fn=make_collator(tokenizer.pad_token_id),
    )
    model.zero_grad(set_to_none=True)
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(_device(model)) for key, value in batch.items()}
        target_tokens = int((batch["labels"] != -100).sum().item())
        (model(**batch).loss * target_tokens).backward()
        total_tokens += target_tokens
    if total_tokens == 0:
        raise ValueError("Gradient probe contains no target tokens")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(total_tokens)
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def set_cumulative_base_residual(
    model: torch.nn.Module,
    desired: Mapping[str, torch.Tensor],
    currently_applied: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    currently_applied = currently_applied or {}
    difference = {
        key: value - currently_applied.get(key, torch.zeros_like(value))
        for key, value in desired.items()
    }
    if difference:
        apply_base_residual(model, difference)
    return {key: value.detach().cpu().clone() for key, value in desired.items()}
