from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from unscramble.prompting import make_training_example
from unscramble.scramble import ScrambleConfig, scramble_code
from unscramble.templates import LANGUAGES, sample_clean_code


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, required=True, help="Number of examples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_chars", type=int, default=120)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--languages", default=",".join(LANGUAGES), help="Comma-separated list")
    ap.add_argument("--aggressive", action="store_true", help="More corruption (harder)")
    ap.add_argument("--allow_drop_lines", action="store_true", help="Allow dropping lines (harder, less learnable)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    languages = [x.strip().lower() for x in args.languages.split(",") if x.strip()]

    cfg = ScrambleConfig()
    if args.aggressive:
        cfg = ScrambleConfig(
            rename_idents=True,
            max_renames=60,
            whitespace_damage=True,
            duplicate_lines=True,
            drop_lines=bool(args.allow_drop_lines),
            shuffle_blocks=True,
            insert_noise_comments=True,
            p_duplicate=0.09,
            p_drop=0.08 if args.allow_drop_lines else 0.0,
            p_whitespace=0.55,
            p_noise=0.13,
        )
    else:
        # By default, keep the task learnable by not deleting lines unless explicitly enabled.
        cfg = ScrambleConfig(
            rename_idents=True,
            max_renames=30,
            whitespace_damage=True,
            duplicate_lines=True,
            drop_lines=bool(args.allow_drop_lines),
            shuffle_blocks=True,
            insert_noise_comments=True,
            p_duplicate=0.06,
            p_drop=0.05 if args.allow_drop_lines else 0.0,
            p_whitespace=0.35,
            p_noise=0.08,
        )

    out_rows: list[dict] = []
    meta = {"scramble_config": asdict(cfg), "seed": args.seed, "n": args.n}
    out_rows.append({"_meta": meta})

    for _ in tqdm(range(args.n), desc="Generating"):
        lang = rng.choice(languages)
        clean = sample_clean_code(rng, language=lang)
        if not (args.min_chars <= len(clean) <= args.max_chars):
            # templates are small, but keep guardrails
            continue
        scrambled = scramble_code(clean, language=lang, rng=rng, cfg=cfg)
        ex = make_training_example(lang, scrambled, clean)
        out_rows.append(ex)

    # shuffle, keeping meta first
    rows = out_rows[1:]
    rng.shuffle(rows)
    final_rows = [out_rows[0], *rows]

    _write_jsonl(Path(args.out), final_rows)
    print(f"Wrote {len(final_rows)-1} examples to {args.out}")
    print(f"Tip: set HF_HOME to control cache dir. Current HF_HOME={os.environ.get('HF_HOME','(default)')}")


if __name__ == "__main__":
    main()

