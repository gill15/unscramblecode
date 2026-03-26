from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from unscramble.prompting import make_messages
from unscramble.lang_plugins import get_plugin, guess_language
from unscramble.gating import passes_minimal_edit


@dataclass(frozen=True)
class Chunk:
    start: int  # inclusive line index
    end: int  # exclusive line index


def _chunk_ranges(n_lines: int, *, chunk_lines: int, overlap: int) -> list[Chunk]:
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be > 0")
    if overlap < 0 or overlap >= chunk_lines:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_lines")
    out: list[Chunk] = []
    step = chunk_lines - overlap
    s = 0
    while s < n_lines:
        e = min(n_lines, s + chunk_lines)
        out.append(Chunk(s, e))
        if e == n_lines:
            break
        s += step
    return out


@torch.inference_mode()
def _generate(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.95,
        # For code restoration, these anti-repetition knobs can backfire by
        # forcing token escapes / mangled output on small models.
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return gen.strip("\n")


def _merge_chunk_replace(lines: list[str], chunk: Chunk, new_text: str) -> list[str]:
    new_lines = new_text.splitlines()
    # Preserve trailing newline behavior by keeping list-of-lines; writing later adds a final newline.
    return lines[: chunk.start] + new_lines + lines[chunk.end :]


def _looks_like_code(text: str) -> bool:
    """
    Guardrail: refuse obviously bad generations (e.g., gibberish / numeric spam).
    """
    t = text.strip()
    if len(t) < 10:
        return False
    # if it's mostly digits/punctuation, it's likely garbage
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    if letters == 0:
        return False
    if digits > letters * 3:
        return False
    # require some structure
    if "\n" not in t and len(t) > 200:
        return False
    return True


def _iter_chunks(lines: list[str], *, chunk_lines: int, overlap: int) -> Iterable[tuple[Chunk, str]]:
    for ch in _chunk_ranges(len(lines), chunk_lines=chunk_lines, overlap=overlap):
        txt = "\n".join(lines[ch.start : ch.end])
        yield ch, txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora", default=None, help="Path to LoRA adapter dir (optional)")
    ap.add_argument("--path", required=True, help="File to unscramble")
    ap.add_argument("--passes", type=int, default=3, help="How many full-file passes")
    ap.add_argument("--chunk_lines", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--backup", action="store_true", help="Create <file>.bak before overwriting")
    ap.add_argument(
        "--max_change_ratio",
        type=float,
        default=0.65,
        help="Reject candidates that change too many original lines (minimal-edit gate).",
    )
    args = ap.parse_args()

    path = Path(args.path)
    orig_text = path.read_text(encoding="utf-8", errors="replace")
    if args.backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_text(orig_text, encoding="utf-8")
    lines = orig_text.splitlines()

    language = guess_language(path)
    plugin = get_plugin(language)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    total_steps = args.passes * max(1, len(_chunk_ranges(len(lines), chunk_lines=args.chunk_lines, overlap=args.overlap)))
    step_idx = 0
    print(f"PROGRESS|0|{total_steps}|0")

    for p in range(args.passes):
        changed_any = False
        # iterate chunks on current lines
        for ch, chunk_text in _iter_chunks(lines, chunk_lines=args.chunk_lines, overlap=args.overlap):
            step_idx += 1
            pct = int((step_idx * 100) / max(1, total_steps))
            print(f"PROGRESS|{step_idx}|{total_steps}|{pct}")
            user_prefix = plugin.user_prefix()
            msgs = make_messages(language, user_prefix + chunk_text, clean=None)
            try:
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                # fallback to raw text if chat template unavailable
                prompt = (
                    f"Language: {language}\n\n{user_prefix}SCRAMBLED:\n{chunk_text}\n\nUNSCRAMBLED:\n"
                )
            applied = False
            # Try a few attempts to get a valid, non-gibberish candidate.
            for temp in (args.temperature, 0.2, 0.35):
                suggestion = _generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temp,
                )
                suggestion = plugin.clean_model_output(suggestion)
                suggestion = plugin.strip_noise(suggestion)
                suggestion = plugin.postprocess_candidate(suggestion)
                if not suggestion.strip():
                    continue
                if not _looks_like_code(suggestion):
                    continue
                if not plugin.validate_candidate(original_chunk=chunk_text, candidate=suggestion):
                    continue
                if suggestion.strip() == chunk_text.strip():
                    applied = True  # treat as success but no-op
                    break
                if not passes_minimal_edit(chunk_text, suggestion, max_change_ratio=args.max_change_ratio):
                    continue
                tentative_lines = _merge_chunk_replace(lines, ch, suggestion)
                tentative_text = "\n".join(tentative_lines) + "\n"
                tentative_text = plugin.finalize_file_text(original_text=orig_text, new_text=tentative_text)
                v = plugin.validate_file_path(path=path, original_text=orig_text, new_text=tentative_text)
                if not v.ok:
                    continue
                lines = tentative_lines
                changed_any = True
                applied = True
                break
            if not applied:
                continue

        if not changed_any:
            break

    # Write back, ensure file ends with newline
    new_text = "\n".join(lines) + "\n"
    new_text = plugin.finalize_file_text(original_text=orig_text, new_text=new_text)

    # Fallback for badly broken Python files:
    # if chunk mode produced no valid change and syntax is still broken, try one whole-file rescue generation.
    if language.lower() == "python":
        try:
            ast.parse(new_text)
        except SyntaxError:
            rescue_prefix = (
                plugin.user_prefix()
                + "This file is heavily scrambled and may not parse.\n"
                + "Reconstruct a single coherent valid Python file with minimal behavior changes.\n\n"
            )
            rescue_msgs = make_messages(language, rescue_prefix + orig_text, clean=None)
            try:
                rescue_prompt = tokenizer.apply_chat_template(rescue_msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                rescue_prompt = f"Language: {language}\n\n{rescue_prefix}SCRAMBLED:\n{orig_text}\n\nUNSCRAMBLED:\n"
            rescue = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=rescue_prompt,
                max_new_tokens=max(args.max_new_tokens, 1200),
                temperature=max(args.temperature, 0.2),
            )
            rescue = plugin.clean_model_output(rescue)
            rescue = plugin.strip_noise(rescue)
            rescue = plugin.postprocess_candidate(rescue)
            if rescue.strip():
                candidate_text = rescue.strip("\n") + "\n"
                candidate_text = plugin.finalize_file_text(original_text=orig_text, new_text=candidate_text)
                rv = plugin.validate_file_path(path=path, original_text=orig_text, new_text=candidate_text)
                if rv.ok:
                    try:
                        ast.parse(candidate_text)
                    except SyntaxError:
                        pass
                    else:
                        new_text = candidate_text
    print("PROGRESS|1|1|100")
    if new_text != orig_text:
        path.write_text(new_text, encoding="utf-8")
        print(f"Updated {path}")
    else:
        print("No changes")


if __name__ == "__main__":
    main()

