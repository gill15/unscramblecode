from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--in", dest="inputs", required=True, nargs="+")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = {"_meta": {"merged_from": args.inputs}}
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for p in args.inputs:
            for row in _iter_jsonl(Path(p)):
                if "_meta" in row and isinstance(row["_meta"], dict):
                    # skip per-file meta lines
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote merged dataset to {out}")


if __name__ == "__main__":
    main()

