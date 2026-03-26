from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable


_IDENT_RE = re.compile(r"\b[_a-zA-Z][_a-zA-Z0-9]*\b")


@dataclass(frozen=True)
class ScrambleConfig:
    rename_idents: bool = True
    max_renames: int = 30
    whitespace_damage: bool = True
    duplicate_lines: bool = True
    drop_lines: bool = True
    shuffle_blocks: bool = True
    insert_noise_comments: bool = True

    # aggressiveness
    p_duplicate: float = 0.06
    p_drop: float = 0.05
    p_whitespace: float = 0.35
    p_noise: float = 0.08


PYTHON_KEYWORDS = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}


def _choose_ident_candidates(code: str, *, language: str) -> list[str]:
    idents = _IDENT_RE.findall(code)
    if language.lower() == "python":
        idents = [x for x in idents if x not in PYTHON_KEYWORDS]
    # avoid renaming dunder / common builtins-ish
    idents = [x for x in idents if not (x.startswith("__") and x.endswith("__"))]
    idents = [x for x in idents if x not in {"print", "len", "range", "int", "str", "float", "dict", "list", "set"}]
    # keep only ones that appear multiple times to make rename meaningful
    counts: dict[str, int] = {}
    for x in idents:
        counts[x] = counts.get(x, 0) + 1
    return [k for k, v in counts.items() if v >= 2 and len(k) >= 2]


def _rename_identifiers(rng: random.Random, code: str, *, language: str, max_renames: int) -> str:
    candidates = _choose_ident_candidates(code, language=language)
    rng.shuffle(candidates)
    candidates = candidates[: max_renames]
    mapping: dict[str, str] = {}
    for i, name in enumerate(candidates):
        # generate consistent but ugly new names
        mapping[name] = f"v_{i}_{rng.randint(10, 999)}"
    if not mapping:
        return code

    # replace as whole-word tokens
    def repl(m: re.Match[str]) -> str:
        t = m.group(0)
        return mapping.get(t, t)

    return _IDENT_RE.sub(repl, code)


def _damage_whitespace(rng: random.Random, code: str, *, p: float) -> str:
    lines = code.splitlines()
    out: list[str] = []
    for ln in lines:
        if rng.random() < p:
            # randomize indentation & collapse spaces
            ln2 = ln.replace("\t", "    ")
            ln2 = re.sub(r"[ ]{2,}", " ", ln2)
            if ln2.strip() and rng.random() < 0.5:
                # add or remove indentation
                if rng.random() < 0.5:
                    ln2 = (" " * rng.choice([0, 2, 4, 6, 8])) + ln2.lstrip()
                else:
                    ln2 = ln2.lstrip()
            out.append(ln2)
        else:
            out.append(ln)
    # sometimes remove blank lines
    if rng.random() < 0.35:
        out = [x for x in out if x.strip() != "" or rng.random() < 0.25]
    return "\n".join(out) + ("\n" if code.endswith("\n") else "")


def _duplicate_and_drop_lines(rng: random.Random, code: str, *, p_dup: float, p_drop: float) -> str:
    lines = code.splitlines()
    out: list[str] = []
    for ln in lines:
        if ln.strip() and rng.random() < p_drop:
            continue
        out.append(ln)
        if ln.strip() and rng.random() < p_dup:
            out.append(ln)
    return "\n".join(out) + ("\n" if code.endswith("\n") else "")


def _insert_noise(rng: random.Random, code: str, *, language: str, p: float) -> str:
    comment = "#" if language.lower() == "python" else "//"
    lines = code.splitlines()
    out: list[str] = []
    for ln in lines:
        out.append(ln)
        if rng.random() < p:
            out.append(f"{comment} TODO: fix {rng.choice(['naming', 'ordering', 'imports', 'formatting'])}")
    return "\n".join(out) + ("\n" if code.endswith("\n") else "")


def _split_top_level_blocks(code: str) -> list[str]:
    """
    Heuristic: split code into blocks separated by 1+ blank lines.
    Works acceptably for many small synthetic snippets.
    """
    parts = re.split(r"\n\s*\n", code.strip("\n"))
    return [p.strip("\n") for p in parts if p.strip("\n")]


def _shuffle_blocks(rng: random.Random, code: str) -> str:
    blocks = _split_top_level_blocks(code)
    if len(blocks) < 3:
        return code
    rng.shuffle(blocks)
    return "\n\n".join(blocks) + ("\n" if code.endswith("\n") else "\n")


def scramble_code(
    clean_code: str,
    *,
    language: str,
    rng: random.Random,
    cfg: ScrambleConfig | None = None,
) -> str:
    cfg = cfg or ScrambleConfig()
    code = clean_code

    transforms: list[Callable[[str], str]] = []
    if cfg.shuffle_blocks:
        transforms.append(lambda c: _shuffle_blocks(rng, c))
    if cfg.rename_idents:
        transforms.append(lambda c: _rename_identifiers(rng, c, language=language, max_renames=cfg.max_renames))
    if cfg.drop_lines or cfg.duplicate_lines:
        transforms.append(lambda c: _duplicate_and_drop_lines(rng, c, p_dup=cfg.p_duplicate, p_drop=cfg.p_drop))
    if cfg.insert_noise_comments:
        transforms.append(lambda c: _insert_noise(rng, c, language=language, p=cfg.p_noise))
    if cfg.whitespace_damage:
        transforms.append(lambda c: _damage_whitespace(rng, c, p=cfg.p_whitespace))

    rng.shuffle(transforms)
    for t in transforms:
        code = t(code)

    # ensure it's not identical too often
    if code.strip() == clean_code.strip():
        code = _damage_whitespace(rng, code, p=min(0.9, cfg.p_whitespace + 0.2))
    return code

