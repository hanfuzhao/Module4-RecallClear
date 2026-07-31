"""LoRA fine-tuning and inference for the RecallClear plain-language model.

https://arxiv.org/abs/2106.09685) of the base model named in
https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from scripts import config
from scripts.make_dataset import read_jsonl
from scripts.prompts import build_few_shot_messages, build_zero_shot_messages

LABEL_IGNORE_INDEX = -100


def load_tokenizer(model_id: str = config.BASE_MODEL_ID) -> AutoTokenizer:
    """Load the base model's tokenizer with a padding token guaranteed."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_example(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    """Tokenise one example, masking the prompt so loss covers only the card."""
    messages = example["messages"]
    prompt_messages = [message for message in messages if message["role"] != "assistant"]
    target_text = next(message["content"] for message in messages if message["role"] == "assistant")

    prompt_ids = list(
        tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"]
    )
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    target_ids = target_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([LABEL_IGNORE_INDEX] * len(prompt_ids) + target_ids)[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_torch_dataset(
    rows: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = config.MAX_SEQUENCE_LENGTH,
    drop_overlong: bool = True,
):
    """Convert JSONL rows into a tokenised ``datasets.Dataset``."""
    from datasets import Dataset

    dataset = Dataset.from_list([{"messages": row["messages"]} for row in rows])
    encoded = dataset.map(
        lambda example: encode_example(example, tokenizer, max_length),
        remove_columns=dataset.column_names,
        desc="tokenising",
    )
    if not drop_overlong:
        return encoded

    kept = encoded.filter(lambda example: len(example["input_ids"]) < max_length)
    dropped = len(encoded) - len(kept)
    if dropped:
        print(f"  dropped {dropped} of {len(encoded)} examples longer than {max_length} tokens")
    return kept


PAD_TO_MULTIPLE_OF = 64


class ReleaseCacheCallback(TrainerCallback):
    """Periodically hand cached device memory back to the allocator."""

    def __init__(self, every_steps: int = 50) -> None:
        self.every_steps = every_steps

    def on_step_end(self, args, state, control, **kwargs):
        """Empty the MPS cache every ``every_steps`` optimizer steps."""
        if state.global_step and state.global_step % self.every_steps == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return control


def build_lora_model(model_id: str = config.BASE_MODEL_ID, device: str | None = None):
    """Load the base model and wrap it in a freshly initialised LoRA adapter."""
    device = device or config.resolve_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.LORA.rank,
        lora_alpha=config.LORA.alpha,
        lora_dropout=config.LORA.dropout,
        target_modules=config.LORA.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model.to(device)


def oversample_rare_urgency(
    rows: list[dict], factor: int = None, seed: int = None
) -> list[dict]:
    """Repeat rare-urgency examples so the model actually sees them."""
    import random

    factor = factor if factor is not None else config.RARE_CLASS_OVERSAMPLE_FACTOR
    seed = seed if seed is not None else config.RANDOM_SEED
    rare = [row for row in rows if row.get("park_it") or row.get("park_outside")]
    combined = rows + rare * (factor - 1)
    random.Random(seed).shuffle(combined)
    return combined


def train_lora(
    train_rows: list[dict],
    validation_rows: list[dict],
    output_dir: Path = config.ADAPTER_DIR,
    epochs: float | None = None,
    max_examples: int | None = None,
) -> Path:
    """Fine-tune the LoRA adapter and save it to ``output_dir``."""
    device = config.resolve_device()
    if device == "cpu":
        torch.set_num_threads(8)
    if max_examples is None:
        max_examples = config.MAX_TRAIN_EXAMPLES
    if max_examples:
        train_rows = train_rows[:max_examples]
    train_rows = oversample_rare_urgency(train_rows)
    rare_count = sum(1 for row in train_rows if row.get("park_it") or row.get("park_outside"))
    print(f"  after oversampling: {len(train_rows)} rows, {rare_count} rare-urgency")
    print(f"Training on {len(train_rows)} examples, validating on {len(validation_rows)} (device={device})")

    tokenizer = load_tokenizer()
    model = build_lora_model(device=device)

    train_dataset = build_torch_dataset(train_rows, tokenizer)
    eval_dataset = build_torch_dataset(validation_rows[: config.TRAINING.max_eval_examples], tokenizer)

    training_arguments = TrainingArguments(
        output_dir=str(config.CHECKPOINT_DIR),
        num_train_epochs=epochs if epochs is not None else config.TRAINING.epochs,
        learning_rate=config.TRAINING.learning_rate,
        per_device_train_batch_size=config.TRAINING.batch_size,
        per_device_eval_batch_size=config.TRAINING.batch_size,
        gradient_accumulation_steps=config.TRAINING.gradient_accumulation_steps,
        warmup_ratio=config.TRAINING.warmup_ratio,
        weight_decay=config.TRAINING.weight_decay,
        lr_scheduler_type=config.TRAINING.lr_scheduler_type,
        logging_steps=config.TRAINING.logging_steps,
        eval_strategy="steps",
        eval_steps=config.TRAINING.eval_steps,
        save_strategy="steps",
        save_steps=config.TRAINING.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,


        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=config.RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
            label_pad_token_id=LABEL_IGNORE_INDEX,
        ),
        callbacks=[ReleaseCacheCallback()],
    )

    result = trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "base_model": config.BASE_MODEL_ID,
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "metrics": result.metrics,
                "log_history": trainer.state.log_history,
                "lora": vars(config.LORA),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Adapter saved to {output_dir}")
    return output_dir


class RecallExplainer:
    """Generates plain-language recall cards, with and without the adapter."""

    MODE_BASE = "base"
    MODE_FEW_SHOT = "few_shot"
    MODE_TUNED = "tuned"
    MODES = (MODE_BASE, MODE_FEW_SHOT, MODE_TUNED)

    def __init__(
        self,
        adapter_path: Path | str | None = config.ADAPTER_DIR,
        base_model_id: str = config.BASE_MODEL_ID,
        device: str | None = None,
        quantise: bool = config.QUANTISE_ON_CPU,
    ) -> None:
        self.device = device or config.resolve_device()
        self.base_model_id = base_model_id
        self.tokenizer = load_tokenizer(base_model_id)

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype)

        self.has_adapter = adapter_path is not None and Path(adapter_path).exists()
        if self.has_adapter:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        self.model = model.to(self.device).eval()
        self.is_quantised = False
        if self.device == "cpu" and quantise and not self.has_adapter:


            self.is_quantised = self._quantise_for_cpu()

    def _quantise_for_cpu(self) -> bool:
        """Apply int8 dynamic quantisation to the linear layers, if supported."""
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            return True
        except (RuntimeError, NotImplementedError) as error:
            print(f"int8 quantisation unavailable, serving full precision ({error})")
            return False

    def _messages_for(self, notice: str, mode: str) -> list[dict]:
        """Return the chat messages appropriate to a comparison mode."""
        if mode == self.MODE_FEW_SHOT:
            return build_few_shot_messages(notice)
        return build_zero_shot_messages(notice)

    def prompt_token_count(self, notice: str, mode: str) -> int:
        """Return how many prompt tokens a mode costs for a given notice."""
        encoded = self.tokenizer.apply_chat_template(
            self._messages_for(notice, mode),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        return len(encoded["input_ids"])

    @torch.inference_mode()
    def explain(
        self,
        notice: str,
        mode: str = MODE_TUNED,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> str:
        """Generate a recall card for one notice under the given mode."""
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {self.MODES}")
        if mode == self.MODE_TUNED and not self.has_adapter:
            raise RuntimeError("no LoRA adapter is loaded; train one first or pass adapter_path")

        messages = self._messages_for(notice, mode)
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

        with self._adapter_enabled(mode == self.MODE_TUNED):
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        completion = generated[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def explain_batch(
        self,
        notices: list[str],
        mode: str = MODE_TUNED,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        batch_size: int = 8,
    ) -> list[str]:
        """Generate cards for many notices, batching to keep evaluation tractable."""
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        outputs: list[str] = []
        try:
            for start in range(0, len(notices), batch_size):
                chunk = notices[start : start + batch_size]
                prompts = [
                    self.tokenizer.apply_chat_template(
                        self._messages_for(notice, mode), tokenize=False, add_generation_prompt=True
                    )
                    for notice in chunk
                ]
                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(self.device)

                with self._adapter_enabled(mode == self.MODE_TUNED):
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                prompt_length = inputs["input_ids"].shape[-1]
                outputs.extend(
                    self.tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
                    for row in generated
                )
        finally:
            self.tokenizer.padding_side = original_side
        return outputs

    def _adapter_enabled(self, enabled: bool):
        """Context manager toggling the LoRA adapter for a single generation."""
        if not self.has_adapter or enabled:
            return _NullContext()
        return self.model.disable_adapter()


class _NullContext:
    """No-op context manager used when the adapter needs no toggling."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exception_details) -> bool:
        return False


def merge_adapter(
    adapter_path: Path = config.ADAPTER_DIR, output_dir: Path = config.MERGED_DIR
) -> Path:
    """Merge LoRA weights into the base model and save a standalone checkpoint."""
    tokenizer = load_tokenizer()
    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_ID, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Merged model saved to {output_dir}")
    return output_dir


def main() -> None:
    """Command-line entry point for training and ad-hoc generation."""
    parser = argparse.ArgumentParser(description="Train or run the RecallClear model.")
    parser.add_argument("--train", action="store_true", help="Run LoRA fine-tuning.")
    parser.add_argument("--merge", action="store_true", help="Merge the adapter into the base model.")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    config.ensure_directories()

    if args.train:
        train_rows = read_jsonl(config.SPLIT_PATHS["train"])
        validation_rows = read_jsonl(config.SPLIT_PATHS["validation"])
        train_lora(train_rows, validation_rows, epochs=args.epochs, max_examples=args.max_examples)
    if args.merge:
        merge_adapter()
    if not (args.train or args.merge):
        parser.error("nothing to do: pass --train and/or --merge")


if __name__ == "__main__":
    main()
