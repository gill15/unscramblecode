from __future__ import annotations

import os
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _repo_root() -> Path:
    # file: <root>/unscramble_gui/app.py
    return Path(__file__).resolve().parents[1]


def _load_stylesheet(name: str) -> str:
    p = Path(__file__).resolve().parent / name
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _python_exe() -> str:
    # Use the venv python if present, otherwise fallback to system python3.
    root = _repo_root()
    vpy = root / ".venv" / "bin" / "python"
    if vpy.exists():
        return str(vpy)
    return "python3"


def _ensure_backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
    return bak


@dataclass(frozen=True)
class RunSpec:
    title: str
    program: str
    args: list[str]
    cwd: str
    env: dict[str, str]


class Runner:
    def __init__(self, log: QTextEdit, status: QLabel):
        self.log = log
        self.status = status
        self.proc: QProcess | None = None
        self._on_finished = None
        self._on_progress = None
        self._title = ""
        self._progress_re = re.compile(r"^PROGRESS\|(\d+)\|(\d+)\|(\d+)$")

    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def stop(self) -> None:
        if not self.proc:
            return
        if self.proc.state() == QProcess.NotRunning:
            return
        self.proc.kill()
        self.status.setText("Stopped")

    def run(self, spec: RunSpec, on_finished=None, on_progress=None) -> None:
        if self.running():
            QMessageBox.warning(None, "Busy", "A job is already running. Stop it first.")
            return

        self.log.clear()
        self._append(f"$ {spec.program} " + " ".join(shlex.quote(a) for a in spec.args))
        self.status.setText(f"Running: {spec.title}")
        self._title = spec.title
        self._on_progress = on_progress

        p = QProcess()
        p.setWorkingDirectory(spec.cwd)
        env = os.environ.copy()
        env.update(spec.env)
        p.setProcessEnvironment(QProcessEnvironment_from_dict(env))

        p.readyReadStandardOutput.connect(lambda: self._append_bytes(p.readAllStandardOutput()))
        p.readyReadStandardError.connect(lambda: self._append_bytes(p.readAllStandardError()))
        p.finished.connect(lambda code, st: self._finished(spec.title, code, st))

        self.proc = p
        self._on_finished = on_finished
        p.start(spec.program, spec.args)

    def _append_bytes(self, b) -> None:
        try:
            s = bytes(b).decode("utf-8", errors="replace")
        except Exception:
            s = str(b)
        self._append(s.rstrip("\n"))

    def _append(self, text: str) -> None:
        if not text:
            return
        # Parse machine-readable progress lines emitted by CLI.
        lines = text.splitlines() or [text]
        visible_lines: list[str] = []
        for ln in lines:
            m = self._progress_re.match(ln.strip())
            if m:
                pct = int(m.group(3))
                self.status.setText(f"Running: {self._title} ({pct}%)")
                if self._on_progress:
                    try:
                        self._on_progress(pct)
                    except Exception:
                        pass
            else:
                visible_lines.append(ln)
        if visible_lines:
            self.log.append("\n".join(visible_lines))
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _finished(self, title: str, code: int, _status) -> None:
        self.status.setText(f"Finished: {title} (exit {code})")
        self._append(f"\n[exit_code={code}]")
        cb = self._on_finished
        self.proc = None
        self._on_finished = None
        self._on_progress = None
        if cb:
            try:
                cb(code)
            except Exception as e:
                self._append(f"[callback_error] {e}")


def QProcessEnvironment_from_dict(env: dict[str, str]):
    # tiny helper to avoid importing QProcessEnvironment directly in type checkers
    from PySide6.QtCore import QProcessEnvironment

    pe = QProcessEnvironment.systemEnvironment()
    for k, v in env.items():
        pe.insert(k, v)
    return pe


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unscramble Code")
        self.resize(1200, 820)

        root = _repo_root()

        self.status = QLabel("Idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Monospace"))
        self.log.setLineWrapMode(QTextEdit.NoWrap)

        self.runner = Runner(self.log, self.status)

        primary = self._tab_paste_unscramble(root)

        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.runner.stop)

        bottom = QHBoxLayout()
        bottom.addWidget(self.progress, 2)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(stop_btn)

        layout = QVBoxLayout()
        layout.addWidget(primary, 5)
        layout.addWidget(QLabel("Logs (behind the scenes)"))
        layout.addWidget(self.log, 2)

        bottom_w = QWidget()
        bottom_w.setLayout(bottom)
        layout.addWidget(bottom_w)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

    def _tab_dataset(self, root: Path) -> QWidget:
        w = QWidget()
        form = QFormLayout()

        out_train = QLineEdit(str(root / "data" / "train.jsonl"))
        out_val = QLineEdit(str(root / "data" / "val.jsonl"))
        n_train = QSpinBox()
        n_train.setRange(10, 2_000_000)
        n_train.setValue(50_000)
        n_val = QSpinBox()
        n_val.setRange(10, 200_000)
        n_val.setValue(2_000)
        seed = QSpinBox()
        seed.setRange(0, 2_000_000_000)
        seed.setValue(42)

        aggressive = QPushButton("Toggle aggressive: OFF")
        aggressive.setCheckable(True)
        aggressive.clicked.connect(lambda: aggressive.setText(f"Toggle aggressive: {'ON' if aggressive.isChecked() else 'OFF'}"))

        def browse_train():
            p, _ = QFileDialog.getSaveFileName(self, "Train JSONL", out_train.text(), "JSONL (*.jsonl)")
            if p:
                out_train.setText(p)

        def browse_val():
            p, _ = QFileDialog.getSaveFileName(self, "Val JSONL", out_val.text(), "JSONL (*.jsonl)")
            if p:
                out_val.setText(p)

        btn_train_path = QPushButton("Browse…")
        btn_train_path.clicked.connect(browse_train)
        btn_val_path = QPushButton("Browse…")
        btn_val_path.clicked.connect(browse_val)

        row_train = QHBoxLayout()
        row_train.addWidget(out_train, 1)
        row_train.addWidget(btn_train_path)
        row_val = QHBoxLayout()
        row_val.addWidget(out_val, 1)
        row_val.addWidget(btn_val_path)

        form.addRow("Train output", _wrap(row_train))
        form.addRow("Val output", _wrap(row_val))
        form.addRow("Train examples", n_train)
        form.addRow("Val examples", n_val)
        form.addRow("Seed", seed)
        form.addRow("Mode", aggressive)

        run_btn = QPushButton("Generate Dataset")

        def run():
            py = _python_exe()
            env = {}
            args_train = [
                "-m",
                "unscramble.generate_data",
                "--out",
                out_train.text(),
                "--n",
                str(n_train.value()),
                "--seed",
                str(seed.value()),
            ]
            if aggressive.isChecked():
                args_train.append("--aggressive")
            # Use a different seed for val deterministically.
            args_val = [
                "-m",
                "unscramble.generate_data",
                "--out",
                out_val.text(),
                "--n",
                str(n_val.value()),
                "--seed",
                str(seed.value() + 1),
            ]
            if aggressive.isChecked():
                args_val.append("--aggressive")

            # Chain via shell to run both sequentially; keep it simple.
            cmd = f"{shlex.quote(py)} " + " ".join(shlex.quote(a) for a in args_train) + " && " + f"{shlex.quote(py)} " + " ".join(
                shlex.quote(a) for a in args_val
            )
            self.runner.run(
                RunSpec(
                    title="Generate dataset",
                    program="/bin/bash",
                    args=["-lc", cmd],
                    cwd=str(root),
                    env=env,
                )
            )

        run_btn.clicked.connect(run)

        box = QGroupBox("Synthetic dataset generation")
        vb = QVBoxLayout()
        vb.addLayout(form)
        vb.addWidget(run_btn)
        box.setLayout(vb)

        lay = QVBoxLayout()
        lay.addWidget(box)
        lay.addStretch(1)
        w.setLayout(lay)
        return w

    def _tab_train(self, root: Path) -> QWidget:
        w = QWidget()
        form = QFormLayout()

        model = QLineEdit("Qwen/Qwen2.5-1.5B-Instruct")
        train = QLineEdit(str(root / "data" / "train.jsonl"))
        val = QLineEdit(str(root / "data" / "val.jsonl"))
        out = QLineEdit(str(root / "runs" / "qwen2.5-1.5b-unscramble-qlora"))

        max_seq = QSpinBox()
        max_seq.setRange(64, 8192)
        max_seq.setValue(512)
        steps = QSpinBox()
        steps.setRange(1, 2_000_000)
        steps.setValue(300)
        grad_acc = QSpinBox()
        grad_acc.setRange(1, 256)
        grad_acc.setValue(16)

        qlora = QPushButton("QLoRA (4-bit): ON")
        qlora.setCheckable(True)
        qlora.setChecked(True)
        qlora.clicked.connect(lambda: qlora.setText(f"QLoRA (4-bit): {'ON' if qlora.isChecked() else 'OFF'}"))

        def browse_any(line: QLineEdit, title: str):
            p, _ = QFileDialog.getOpenFileName(self, title, line.text(), "JSONL (*.jsonl);;All (*.*)")
            if p:
                line.setText(p)

        def browse_dir(line: QLineEdit, title: str):
            p = QFileDialog.getExistingDirectory(self, title, line.text())
            if p:
                line.setText(p)

        b_train = QPushButton("Browse…")
        b_train.clicked.connect(lambda: browse_any(train, "Train JSONL"))
        b_val = QPushButton("Browse…")
        b_val.clicked.connect(lambda: browse_any(val, "Val JSONL"))
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(lambda: browse_dir(out, "Output folder"))

        row_train = QHBoxLayout()
        row_train.addWidget(train, 1)
        row_train.addWidget(b_train)
        row_val = QHBoxLayout()
        row_val.addWidget(val, 1)
        row_val.addWidget(b_val)
        row_out = QHBoxLayout()
        row_out.addWidget(out, 1)
        row_out.addWidget(b_out)

        form.addRow("Base model", model)
        form.addRow("Train dataset", _wrap(row_train))
        form.addRow("Val dataset", _wrap(row_val))
        form.addRow("Output (adapter)", _wrap(row_out))
        form.addRow("Max seq len", max_seq)
        form.addRow("Max steps", steps)
        form.addRow("Grad accum", grad_acc)
        form.addRow("Mode", qlora)

        run_btn = QPushButton("Start Training")

        def run():
            py = _python_exe()
            args = [
                "-m",
                "unscramble.train_lora",
                "--model",
                model.text().strip(),
                "--train",
                train.text().strip(),
                "--val",
                val.text().strip(),
                "--out",
                out.text().strip(),
                "--max_seq_len",
                str(max_seq.value()),
                "--epochs",
                "1",
                "--max_steps",
                str(steps.value()),
                "--batch_size",
                "1",
                "--grad_accum",
                str(grad_acc.value()),
                "--lr",
                "1e-4",
            ]
            if qlora.isChecked():
                args.append("--load_in_4bit")
            self.runner.run(
                RunSpec(
                    title="Train",
                    program=py,
                    args=args,
                    cwd=str(root),
                    env={},
                )
            )

        run_btn.clicked.connect(run)

        box = QGroupBox("Training (LoRA / QLoRA)")
        vb = QVBoxLayout()
        vb.addLayout(form)
        vb.addWidget(run_btn)
        box.setLayout(vb)

        lay = QVBoxLayout()
        lay.addWidget(box)
        lay.addStretch(1)
        w.setLayout(lay)
        return w

    def _tab_unscramble(self, root: Path) -> QWidget:
        w = QWidget()
        form = QFormLayout()

        model = QLineEdit("Qwen/Qwen2.5-1.5B-Instruct")
        lora = QLineEdit(str(root / "runs" / "qwen2.5-1.5b-unscramble-qlora-long"))
        path = QLineEdit("")
        passes = QSpinBox()
        passes.setRange(1, 50)
        passes.setValue(3)
        chunk_lines = QSpinBox()
        chunk_lines.setRange(40, 2000)
        chunk_lines.setValue(200)
        overlap = QSpinBox()
        overlap.setRange(0, 500)
        overlap.setValue(50)
        max_new_tokens = QSpinBox()
        max_new_tokens.setRange(32, 4000)
        max_new_tokens.setValue(700)
        temperature = QDoubleSpinBox()
        temperature.setRange(0.0, 2.0)
        temperature.setSingleStep(0.05)
        temperature.setValue(0.0)

        def browse_file():
            p, _ = QFileDialog.getOpenFileName(self, "Scrambled file", str(root), "All (*.*)")
            if p:
                path.setText(p)

        b_file = QPushButton("Browse…")
        b_file.clicked.connect(browse_file)

        row_path = QHBoxLayout()
        row_path.addWidget(path, 1)
        row_path.addWidget(b_file)

        form.addRow("Base model", model)
        form.addRow("LoRA adapter folder (optional)", lora)
        form.addRow("File to unscramble", _wrap(row_path))
        form.addRow("Passes", passes)
        form.addRow("Chunk lines", chunk_lines)
        form.addRow("Overlap", overlap)
        form.addRow("Max new tokens", max_new_tokens)
        form.addRow("Temperature", temperature)

        run_btn = QPushButton("Unscramble File (creates .bak)")

        def run():
            p = path.text().strip()
            if not p:
                QMessageBox.warning(self, "Missing file", "Pick a file to unscramble.")
                return
            file_path = Path(p)
            if not file_path.exists():
                QMessageBox.warning(self, "Not found", f"File does not exist: {file_path}")
                return

            bak = _ensure_backup(file_path)
            QMessageBox.information(self, "Backup created", f"Backup: {bak}")

            py = _python_exe()
            args = [
                "-m",
                "unscramble.unscramble_file",
                "--model",
                model.text().strip(),
                "--path",
                str(file_path),
                "--passes",
                str(passes.value()),
                "--chunk_lines",
                str(chunk_lines.value()),
                "--overlap",
                str(overlap.value()),
                "--max_new_tokens",
                str(max_new_tokens.value()),
                "--temperature",
                str(temperature.value()),
                "--backup",
            ]
            lora_path = lora.text().strip()
            if lora_path:
                args += ["--lora", lora_path]

            self.runner.run(
                RunSpec(
                    title="Unscramble file",
                    program=py,
                    args=args,
                    cwd=str(root),
                    env={},
                )
            )

        run_btn.clicked.connect(run)

        box = QGroupBox("Iterative unscramble")
        vb = QVBoxLayout()
        vb.addLayout(form)
        vb.addWidget(run_btn)
        box.setLayout(vb)

        lay = QVBoxLayout()
        lay.addWidget(box)
        lay.addStretch(1)
        w.setLayout(lay)
        return w

    def _tab_batch_test(self, root: Path) -> QWidget:
        w = QWidget()
        form = QFormLayout()

        model = QLineEdit("Qwen/Qwen2.5-1.5B-Instruct")
        lora = QLineEdit(str(root / "runs" / "qwen2.5-1.5b-unscramble-qlora-long"))

        scan_root = QLineEdit(str(root / "tests_scrambled"))
        tmp_dir = QLineEdit("/tmp/unscramble_batch")

        patterns = QLineEdit("*.py *.js *.ts *.tsx *.sh *.bash")

        passes = QSpinBox()
        passes.setRange(1, 50)
        passes.setValue(1)
        chunk_lines = QSpinBox()
        chunk_lines.setRange(40, 2000)
        chunk_lines.setValue(180)
        overlap = QSpinBox()
        overlap.setRange(0, 500)
        overlap.setValue(40)
        max_new_tokens = QSpinBox()
        max_new_tokens.setRange(32, 4000)
        max_new_tokens.setValue(700)
        temperature = QDoubleSpinBox()
        temperature.setRange(0.0, 2.0)
        temperature.setSingleStep(0.05)
        temperature.setValue(0.0)

        include_lora = QCheckBox("Use LoRA adapter")
        include_lora.setChecked(True)

        def browse_root():
            p = QFileDialog.getExistingDirectory(self, "Folder to scan", scan_root.text())
            if p:
                scan_root.setText(p)

        b_root = QPushButton("Browse…")
        b_root.clicked.connect(browse_root)
        row_root = QHBoxLayout()
        row_root.addWidget(scan_root, 1)
        row_root.addWidget(b_root)

        form.addRow("Base model", model)
        form.addRow("LoRA adapter folder", lora)
        form.addRow("", include_lora)
        form.addRow("Folder to scan", _wrap(row_root))
        form.addRow("Patterns", patterns)
        form.addRow("Temp output dir", tmp_dir)
        form.addRow("Passes", passes)
        form.addRow("Chunk lines", chunk_lines)
        form.addRow("Overlap", overlap)
        form.addRow("Max new tokens", max_new_tokens)
        form.addRow("Temperature", temperature)

        run_btn = QPushButton("Run Batch Test")

        def run():
            py = _python_exe()
            args = [
                "-m",
                "unscramble.batch_test",
                "--root",
                scan_root.text().strip(),
                "--patterns",
                *patterns.text().split(),
                "--model",
                model.text().strip(),
                "--passes",
                str(passes.value()),
                "--chunk_lines",
                str(chunk_lines.value()),
                "--overlap",
                str(overlap.value()),
                "--max_new_tokens",
                str(max_new_tokens.value()),
                "--temperature",
                str(temperature.value()),
                "--tmp_dir",
                tmp_dir.text().strip(),
            ]
            if include_lora.isChecked():
                lp = lora.text().strip()
                if lp:
                    args += ["--lora", lp]
            self.runner.run(
                RunSpec(
                    title="Batch test",
                    program=py,
                    args=args,
                    cwd=str(root),
                    env={},
                )
            )

        run_btn.clicked.connect(run)

        box = QGroupBox("Batch testing (folder scan)")
        vb = QVBoxLayout()
        vb.addLayout(form)
        vb.addWidget(run_btn)
        box.setLayout(vb)

        lay = QVBoxLayout()
        lay.addWidget(box)
        lay.addStretch(1)
        w.setLayout(lay)
        return w

    def _tab_paste_unscramble(self, root: Path) -> QWidget:
        w = QWidget()
        # Keep UI minimal; these backend defaults are fixed for one-click flow.
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        lora_path = str(root / "runs" / "qwen2.5-1.5b-unscramble-qlora-long")
        passes = 2
        chunk_lines = 180
        overlap = 40
        max_new_tokens = 700
        temperature = 0.0

        in_edit = QTextEdit()
        in_edit.setPlaceholderText("Paste scrambled code here...")
        in_edit.setFont(QFont("Monospace"))
        in_edit.setLineWrapMode(QTextEdit.NoWrap)
        out_edit = QTextEdit()
        out_edit.setReadOnly(True)
        out_edit.setFont(QFont("Monospace"))
        out_edit.setLineWrapMode(QTextEdit.NoWrap)
        out_edit.setPlaceholderText("Unscrambled code will appear here...")

        btn_uns = QPushButton("Unscramble")

        def run():
            text = in_edit.toPlainText()
            if not text.strip():
                QMessageBox.warning(self, "Missing input", "Paste scrambled code first.")
                return
            py = _python_exe()
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
                tf.write(text)
                if not text.endswith("\n"):
                    tf.write("\n")
                tmp_path = tf.name
            out_edit.clear()
            self.progress.setValue(0)
            args = [
                "-m",
                "unscramble.unscramble_file",
                "--model",
                model_id,
                "--path",
                tmp_path,
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
            if lora_path:
                args += ["--lora", lora_path]

            def done(exit_code: int) -> None:
                self.progress.setValue(100 if exit_code == 0 else 0)
                if exit_code != 0:
                    return
                try:
                    out_text = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
                    out_edit.setPlainText(out_text)
                except Exception as e:
                    QMessageBox.warning(self, "Read error", f"Could not read output: {e}")

            self.runner.run(
                RunSpec(
                    title="Unscramble pasted code",
                    program=py,
                    args=args,
                    cwd=str(root),
                    env={},
                ),
                on_finished=done,
                on_progress=lambda pct: self.progress.setValue(max(0, min(100, pct))),
            )

        btn_uns.clicked.connect(run)

        editors = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Scrambled input"))
        left.addWidget(in_edit)
        right = QVBoxLayout()
        right.addWidget(QLabel("Unscrambled output"))
        right.addWidget(out_edit)
        editors.addLayout(left, 1)
        editors.addLayout(right, 1)

        box = QGroupBox("Unscramble")
        vb = QVBoxLayout()
        vb.addWidget(btn_uns)
        vb.addLayout(editors)
        box.setLayout(vb)

        lay = QVBoxLayout()
        lay.addWidget(box)
        w.setLayout(lay)
        return w

    def _tab_project(self, root: Path) -> QWidget:
        w = QWidget()
        form = QFormLayout()

        model = QLineEdit("Qwen/Qwen2.5-1.5B-Instruct")
        lora = QLineEdit(str(root / "runs" / "qwen2.5-1.5b-unscramble-qlora-long"))
        include_lora = QCheckBox("Use LoRA adapter")
        include_lora.setChecked(True)

        project_root = QLineEdit(str(root))
        include = QLineEdit("tests_scrambled/*.py tests_scrambled/*.js tests_scrambled/*.ts tests_scrambled/*.sh")
        exclude_dirs = QLineEdit(".git .venv node_modules dist build __pycache__ .mypy_cache")
        state = QLineEdit(".unscramble_state.json")
        gate_cmd = QLineEdit("")

        passes = QSpinBox()
        passes.setRange(1, 50)
        passes.setValue(2)
        chunk_lines = QSpinBox()
        chunk_lines.setRange(40, 2000)
        chunk_lines.setValue(180)
        overlap = QSpinBox()
        overlap.setRange(0, 500)
        overlap.setValue(40)
        max_new_tokens = QSpinBox()
        max_new_tokens.setRange(32, 4000)
        max_new_tokens.setValue(700)
        temperature = QDoubleSpinBox()
        temperature.setRange(0.0, 2.0)
        temperature.setSingleStep(0.05)
        temperature.setValue(0.0)
        max_change_ratio = QDoubleSpinBox()
        max_change_ratio.setRange(0.0, 1.0)
        max_change_ratio.setSingleStep(0.05)
        max_change_ratio.setValue(0.65)
        max_files = QSpinBox()
        max_files.setRange(0, 10_000_000)
        max_files.setValue(0)

        dry_run = QCheckBox("Dry run (no changes)")
        dry_run.setChecked(False)

        def browse_root():
            p = QFileDialog.getExistingDirectory(self, "Project root", project_root.text())
            if p:
                project_root.setText(p)

        b_root = QPushButton("Browse…")
        b_root.clicked.connect(browse_root)
        row_root = QHBoxLayout()
        row_root.addWidget(project_root, 1)
        row_root.addWidget(b_root)

        form.addRow("Base model", model)
        form.addRow("LoRA adapter folder", lora)
        form.addRow("", include_lora)
        form.addRow("Project root", _wrap(row_root))
        form.addRow("Include globs", include)
        form.addRow("Exclude dirs", exclude_dirs)
        form.addRow("State file", state)
        form.addRow("Gate command (optional)", gate_cmd)
        form.addRow("Passes", passes)
        form.addRow("Chunk lines", chunk_lines)
        form.addRow("Overlap", overlap)
        form.addRow("Max new tokens", max_new_tokens)
        form.addRow("Temperature", temperature)
        form.addRow("Max change ratio", max_change_ratio)
        form.addRow("Max files (0=all)", max_files)
        form.addRow("", dry_run)

        run_btn = QPushButton("Run Project Unscramble")

        def run():
            py = _python_exe()
            args = [
                "-m",
                "unscramble.unscramble_project",
                "--root",
                project_root.text().strip(),
                "--model",
                model.text().strip(),
                "--include",
                *include.text().split(),
                "--exclude_dirs",
                *exclude_dirs.text().split(),
                "--state",
                state.text().strip(),
                "--gate_cmd",
                gate_cmd.text(),
                "--passes",
                str(passes.value()),
                "--chunk_lines",
                str(chunk_lines.value()),
                "--overlap",
                str(overlap.value()),
                "--max_new_tokens",
                str(max_new_tokens.value()),
                "--temperature",
                str(temperature.value()),
                "--max_change_ratio",
                str(max_change_ratio.value()),
                "--max_files",
                str(max_files.value()),
            ]
            if include_lora.isChecked():
                lp = lora.text().strip()
                if lp:
                    args += ["--lora", lp]
            if dry_run.isChecked():
                args.append("--dry_run")

            self.runner.run(
                RunSpec(
                    title="Project unscramble",
                    program=py,
                    args=args,
                    cwd=str(root),
                    env={},
                )
            )

        run_btn.clicked.connect(run)

        box = QGroupBox("Project unscramble (large codebases)")
        vb = QVBoxLayout()
        vb.addLayout(form)
        vb.addWidget(run_btn)
        box.setLayout(vb)

        lay = QVBoxLayout()
        lay.addWidget(box)
        lay.addStretch(1)
        w.setLayout(lay)
        return w


def _wrap(layout: QHBoxLayout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


def main() -> None:
    app = QApplication([])
    app.setApplicationDisplayName("Unscramble Code")
    qss = _load_stylesheet("vscode_dark.qss")
    if qss:
        app.setStyleSheet(qss)
    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()

