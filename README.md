## Unscramble Code (Qwen2.5-1.5B)

This repo generates **synthetic scrambled↔unscrambled code chunk pairs**, fine-tunes **Qwen2.5-1.5B** with **LoRA**, and provides a CLI that **iteratively unscrambles a source file** by proposing and applying patches.

### What you get

- **Synthetic dataset generator**: produces many examples across Python/JS/TS/Bash with multiple scrambling styles
- **Training**: LoRA fine-tune script for `Qwen/Qwen2.5-1.5B-Instruct` (downloaded automatically)
- **Tooling**: `unscramble_file.py` that chunks a file, runs the model multiple passes, and applies safe diffs

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Generate synthetic training data

This creates JSONL records with instruction-style prompts and targets.

```bash
python -m unscramble.generate_data \
  --out data/train.jsonl \
  --n 50000 \
  --seed 42

python -m unscramble.generate_data \
  --out data/val.jsonl \
  --n 2000 \
  --seed 43
```

### 2) Train (LoRA) on Qwen2.5-1.5B

Defaults are conservative. For best results, use a CUDA GPU.

```bash
python -m unscramble.train_lora \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --out runs/qwen2.5-1.5b-unscramble-lora \
  --max_seq_len 2048 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 16
```

### 3) Unscramble a file (iterative)

This runs the model on chunks and applies unified diffs if they apply cleanly.

```bash
python -m unscramble.unscramble_file \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --lora runs/qwen2.5-1.5b-unscramble-lora \
  --path path/to/scrambled_file.py \
  --passes 3 \
  --chunk_lines 180 \
  --overlap 40
```

### 4) Batch test on many files

Runs the unscrambler over a folder and prints a pass/fail report (Python files are validated with `ast.parse()` and basic guardrails).

```bash
python -m unscramble.batch_test --root tests_scrambled
```

### 5) Unscramble an entire project (large codebases)

This scans a folder, runs the unscrambler per file, and keeps a resume/checkpoint state in `.unscramble_state.json`.
You can optionally provide a project build/test command as a strict gate (recommended for semantic safety).

```bash
python -m unscramble.unscramble_project \
  --root . \
  --include "src/**/*.py" "src/**/*.js" "src/**/*.ts" \
  --exclude_dirs .git .venv node_modules dist build __pycache__ \
  --gate_cmd "pytest -q" \
  --passes 2 \
  --max_change_ratio 0.65
```

### Desktop GUI (Linux)

Install deps, then:

```bash
./run_gui.sh
```

If you see a Qt error like “Could not load the Qt platform plugin 'xcb'”, install the missing system libraries:

```bash
sudo apt update
sudo apt install -y libxcb-cursor0
```

On Wayland desktops you can also try:

```bash
QT_QPA_PLATFORM=wayland ./run_gui.sh
```

### Notes / expectations

- Scrambling is synthetic, so quality depends on how close your real “scrambled code” is to the synthetic corruptions.
- The CLI is intentionally cautious: it only applies patches that parse as unified diffs and match file context.
- If you want higher quality, increase dataset size and include **your real scrambled/unscrambled pairs** too.

