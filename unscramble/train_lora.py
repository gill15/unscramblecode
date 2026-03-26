from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig


@dataclass(frozen=True)
class DataRow:
    prompt: str
    response: str


def _load_jsonl(path: str):
    # datasets can read JSONL, but we store a meta row first; filter it out.
    ds = load_dataset("json", data_files=path, split="train")
    # When a JSONL has a `_meta` object in only the first row, Datasets may
    # still create a `_meta` column for all rows with null values. Filter by value.
    ds = ds.filter(lambda r: r.get("_meta") is None)
    return ds


def _format_for_sft(tokenizer, row: dict[str, Any]) -> str:
    # Prefer chat-template formatting for instruct models.
    if "messages" in row and row["messages"]:
        msgs = row["messages"]
        try:
            # include assistant turn in training text
            return tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            # fall back to plain prompt/response
            pass

    prompt = row["prompt"]
    response = row["response"]
    text = prompt + response
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token
    return text


def _to_text_only_dataset(ds, tokenizer):
    ds2 = ds.map(lambda r: {"text": _format_for_sft(tokenizer, r)})
    keep = {"text"}
    drop_cols = [c for c in ds2.column_names if c not in keep]
    if drop_cols:
        ds2 = ds2.remove_columns(drop_cols)
    return ds2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", required=True, help="train.jsonl")
    ap.add_argument("--val", required=True, help="val.jsonl")
    ap.add_argument("--out", required=True, help="Output directory for LoRA adapter")

    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_steps", type=int, default=-1, help="Override number of training steps (useful for smoke tests)")

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="Force bf16")
    ap.add_argument("--fp16", action="store_true", help="Force fp16")
    ap.add_argument("--cpu", action="store_true", help="Force CPU training (slow, low memory)")
    ap.add_argument("--load_in_4bit", action="store_true", help="Enable QLoRA (4-bit base model on GPU)")
    ap.add_argument(
        "--resume_from",
        default=None,
        help="Path to a checkpoint directory (e.g. runs/.../checkpoint-200) to resume training",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto"
    dtype: str | torch.dtype = "auto"
    if args.cpu:
        device_map = "cpu"
        dtype = torch.float32
    else:
        if args.bf16:
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.float16

    quant_cfg = None
    if args.load_in_4bit:
        if args.bf16:
            raise SystemExit("--load_in_4bit with --bf16 is not supported in this setup; use --fp16")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # when quantized, let HF decide dtype for non-quant modules
        dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )

    # Qwen2.x uses typical projection names; this target list works well for Qwen-family.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_ds = _load_jsonl(args.train)
    val_ds = _load_jsonl(args.val)
    train_ds = _to_text_only_dataset(train_ds, tokenizer)
    val_ds = _to_text_only_dataset(val_ds, tokenizer)

    training_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="none",
        seed=args.seed,
        bf16=bool(args.bf16) if args.bf16 else None,
        # For QLoRA we already set 4-bit compute dtype; avoid AMP scaler issues here.
        fp16=bool(args.fp16) if not args.load_in_4bit else False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.load_in_4bit and not args.cpu else "adamw_torch",
        max_grad_norm=0.0 if args.load_in_4bit else 1.0,
        max_length=args.max_seq_len,
        packing=False,
        dataset_text_field="text",
        use_cpu=bool(args.cpu),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Persist run info for reproducibility
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "train": os.path.abspath(args.train),
                "val": os.path.abspath(args.val),
                "max_seq_len": args.max_seq_len,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "warmup_ratio": args.warmup_ratio,
                "lora": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                },
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter to {out_dir}")


if __name__ == "__main__":
    main()

