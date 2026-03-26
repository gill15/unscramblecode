from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    path: Path
    tmp: Path
    rc: int
    updated: bool
    python_parse_ok: bool | None
    guard_ok: bool | None
    guard_reason: str | None


_INPUT_RE = re.compile(r"\binput\s*\(")


def _has_input(text: str) -> bool:
    return _INPUT_RE.search(text) is not None


def _has_main_guard(text: str) -> bool:
    return ("if __name__" in text) and ("__main__" in text)


def _run_unscramble(
    *,
    model: str,
    lora: str | None,
    src: Path,
    tmp: Path,
    passes: int,
    chunk_lines: int,
    overlap: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "unscramble.unscramble_file",
        "--model",
        model,
        "--path",
        str(tmp),
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
        cmd.extend(["--lora", lora])

    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def _validate_python(*, orig: str, new: str) -> tuple[bool, bool, str | None]:
    # parse
    ast.parse(new)

    # guardrails
    if (not _has_input(orig)) and _has_input(new):
        return True, False, "introduced input()"
    if _has_main_guard(orig) and (not _has_main_guard(new)):
        return True, False, "dropped __main__ guard"
    return True, True, None


def _iter_files(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(root.rglob(pat)))
    # de-dupe
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in out:
        if p.is_file() and p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tests_scrambled", help="Folder to scan")
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=["*.py", "*.js", "*.ts", "*.tsx", "*.sh", "*.bash"],
        help="Glob patterns to include (space-separated)",
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--chunk_lines", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tmp_dir", default="/tmp/unscramble_batch", help="Where to write temp outputs")
    args = ap.parse_args()

    root = Path(args.root)
    tmp_root = Path(args.tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    files = _iter_files(root, args.patterns)
    if not files:
        print(f"No files found under {root} for patterns: {args.patterns}")
        raise SystemExit(2)

    results: list[Result] = []
    for src in files:
        tmp = tmp_root / src.name
        shutil.copy2(src, tmp)
        orig = src.read_text(encoding="utf-8", errors="replace")
        rc, out = _run_unscramble(
            model=args.model,
            lora=args.lora,
            src=src,
            tmp=tmp,
            passes=args.passes,
            chunk_lines=args.chunk_lines,
            overlap=args.overlap,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        updated = "Updated " in out
        new = tmp.read_text(encoding="utf-8", errors="replace")

        python_parse_ok: bool | None = None
        guard_ok: bool | None = None
        guard_reason: str | None = None
        if src.suffix == ".py":
            try:
                python_parse_ok, guard_ok, guard_reason = _validate_python(orig=orig, new=new)
            except Exception as e:
                python_parse_ok = False
                guard_ok = None
                guard_reason = f"{type(e).__name__}: {e}"

        results.append(
            Result(
                path=src,
                tmp=tmp,
                rc=rc,
                updated=updated,
                python_parse_ok=python_parse_ok,
                guard_ok=guard_ok,
                guard_reason=guard_reason,
            )
        )

    # report
    py_total = sum(1 for r in results if r.path.suffix == ".py")
    py_ok = sum(1 for r in results if r.path.suffix == ".py" and r.python_parse_ok)
    updated_n = sum(1 for r in results if r.updated)

    print(f"Files: {len(results)} | Updated: {updated_n} | Python parse OK: {py_ok}/{py_total}")
    for r in results:
        extra = ""
        if r.path.suffix == ".py":
            extra = f" | py_parse={r.python_parse_ok} guards={r.guard_ok}"
            if r.guard_reason:
                extra += f" ({r.guard_reason})"
        print(f"- {r.path} -> {r.tmp} | rc={r.rc} updated={r.updated}{extra}")


if __name__ == "__main__":
    main()

