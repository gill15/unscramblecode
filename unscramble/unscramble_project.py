from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from unscramble.lang_plugins import get_plugin, guess_language
from unscramble.gating import passes_minimal_edit


@dataclass(frozen=True)
class FileJob:
    path: Path
    language: str


def _sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()


def _scan_files(root: Path, *, include_globs: list[str], exclude_dirs: set[str]) -> list[FileJob]:
    out: list[FileJob] = []
    for pat in include_globs:
        for p in root.rglob(pat):
            if not p.is_file():
                continue
            if any(part in exclude_dirs for part in p.parts):
                continue
            lang = guess_language(p)
            out.append(FileJob(path=p, language=lang))
    # de-dupe + stable order
    seen: set[Path] = set()
    uniq: list[FileJob] = []
    for j in sorted(out, key=lambda x: str(x.path)):
        if j.path not in seen:
            uniq.append(j)
            seen.add(j.path)
    return uniq


def _ensure_backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
    return bak


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"files": {}}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _run_gate(cmd: str, *, cwd: Path) -> tuple[bool, str]:
    if not cmd.strip():
        return True, "skipped (no gate)"
    p = subprocess.run(["/bin/bash", "-lc", cmd], cwd=str(cwd), capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode == 0:
        return True, out
    return False, out


def _run_unscramble_file(
    *,
    model: str,
    lora: str | None,
    path: Path,
    passes: int,
    chunk_lines: int,
    overlap: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[int, str]:
    args = [
        sys.executable,
        "-m",
        "unscramble.unscramble_file",
        "--model",
        model,
        "--path",
        str(path),
        "--passes",
        str(passes),
        "--chunk_lines",
        str(chunk_lines),
        "--overlap",
        str(overlap),
        "--max_new_tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--backup",
    ]
    if lora:
        args += ["--lora", lora]
    p = subprocess.run(args, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Project root to scan")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--include", nargs="+", default=["**/*.*"], help="Glob patterns to include")
    ap.add_argument(
        "--exclude_dirs",
        nargs="+",
        default=[".git", ".venv", "node_modules", "dist", "build", "__pycache__", ".mypy_cache"],
    )
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--chunk_lines", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--state", default=".unscramble_state.json")
    ap.add_argument("--gate_cmd", default="", help="Shell command to run as build/test gate (run from root)")
    ap.add_argument(
        "--max_change_ratio",
        type=float,
        default=0.65,
        help="Reject changes that modify too many original lines (minimal-edit gate).",
    )
    ap.add_argument("--max_files", type=int, default=0, help="0 = no limit")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    state_path = root / args.state
    state = _load_state(state_path)
    files_state: dict = state.setdefault("files", {})

    jobs = _scan_files(root, include_globs=args.include, exclude_dirs=set(args.exclude_dirs))
    if args.max_files and args.max_files > 0:
        jobs = jobs[: args.max_files]

    print(f"Found {len(jobs)} files")

    updated = 0
    skipped = 0
    failed = 0

    for j in jobs:
        rel = str(j.path.relative_to(root))
        orig_text = j.path.read_text(encoding="utf-8", errors="replace")
        orig_hash = _sha256_text(orig_text)

        prev = files_state.get(rel)
        if prev and prev.get("orig_hash") == orig_hash and prev.get("status") in {"ok", "skipped"}:
            skipped += 1
            continue

        plugin = get_plugin(j.language)
        if j.language == "text":
            files_state[rel] = {"status": "skipped", "reason": "unknown language", "orig_hash": orig_hash}
            _save_state(state_path, state)
            skipped += 1
            continue

        if args.dry_run:
            files_state[rel] = {"status": "skipped", "reason": "dry_run", "orig_hash": orig_hash}
            _save_state(state_path, state)
            skipped += 1
            continue

        _ensure_backup(j.path)

        rc, log = _run_unscramble_file(
            model=args.model,
            lora=args.lora,
            path=j.path,
            passes=args.passes,
            chunk_lines=args.chunk_lines,
            overlap=args.overlap,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        new_text = j.path.read_text(encoding="utf-8", errors="replace")
        new_hash = _sha256_text(new_text)

        if not passes_minimal_edit(orig_text, new_text, max_change_ratio=args.max_change_ratio):
            bak = j.path.with_suffix(j.path.suffix + ".bak")
            if bak.exists():
                shutil.copy2(bak, j.path)
            files_state[rel] = {
                "status": "failed",
                "reason": f"minimal-edit gate failed (>{args.max_change_ratio:.2f})",
                "orig_hash": orig_hash,
            }
            _save_state(state_path, state)
            failed += 1
            continue

        # Validate file with plugin (parse/compile checks if toolchain exists)
        v = plugin.validate_file_path(path=j.path, original_text=orig_text, new_text=new_text)
        if not v.ok:
            # rollback
            bak = j.path.with_suffix(j.path.suffix + ".bak")
            if bak.exists():
                shutil.copy2(bak, j.path)
            files_state[rel] = {
                "status": "failed",
                "reason": v.reason,
                "orig_hash": orig_hash,
            }
            _save_state(state_path, state)
            failed += 1
            continue

        # Optional project gate
        ok_gate, gate_out = _run_gate(args.gate_cmd, cwd=root)
        if not ok_gate:
            bak = j.path.with_suffix(j.path.suffix + ".bak")
            if bak.exists():
                shutil.copy2(bak, j.path)
            files_state[rel] = {"status": "failed", "reason": "gate failed", "orig_hash": orig_hash}
            _save_state(state_path, state)
            failed += 1
            continue

        files_state[rel] = {
            "status": "ok",
            "orig_hash": orig_hash,
            "new_hash": new_hash,
            "changed": new_hash != orig_hash,
            "rc": rc,
            "validator": v.reason,
        }
        _save_state(state_path, state)
        if new_hash != orig_hash:
            updated += 1
        else:
            skipped += 1

    print(f"Done. updated={updated} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()

