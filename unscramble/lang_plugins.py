from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FileValidationResult:
    ok: bool
    reason: str | None = None


class LanguagePlugin(Protocol):
    name: str
    extensions: set[str]

    def user_prefix(self) -> str: ...

    def clean_model_output(self, text: str) -> str: ...

    def strip_noise(self, text: str) -> str: ...

    def postprocess_candidate(self, text: str) -> str: ...

    def validate_candidate(self, *, original_chunk: str, candidate: str) -> bool: ...

    def validate_file_path(
        self, *, path: Path, original_text: str, new_text: str
    ) -> FileValidationResult: ...

    def validate_file_text(self, *, original_text: str, new_text: str) -> FileValidationResult: ...

    def finalize_file_text(self, *, original_text: str, new_text: str) -> str: ...


def guess_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".py"}:
        return "python"
    if ext in {".js", ".mjs", ".cjs"}:
        return "javascript"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    if ext in {".sh", ".bash"}:
        return "bash"
    if ext in {".java"}:
        return "java"
    if ext in {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"}:
        return "cpp"
    if ext in {".go"}:
        return "go"
    if ext in {".rs"}:
        return "rust"
    if ext in {".php"}:
        return "php"
    return "text"


def _clean_model_output_generic(text: str) -> str:
    t = text.strip()
    for marker in ("\nHuman:", "\nAssistant:", "\nUser:", "\nSystem:"):
        if marker in t:
            t = t.split(marker, 1)[0].strip()

    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 2:
            inside = parts[1]
            inside_lines = inside.splitlines()
            if inside_lines and inside_lines[0].strip().lower() in {
                "python",
                "py",
                "javascript",
                "js",
                "typescript",
                "ts",
                "bash",
                "sh",
                "java",
                "c",
                "cpp",
                "c++",
                "go",
                "rust",
                "rs",
                "php",
            }:
                inside_lines = inside_lines[1:]
            t = "\n".join(inside_lines).strip()

    if "\nOutput:" in t:
        t = t.split("\nOutput:", 1)[0].strip()

    return t.strip("\n")


class DefaultPlugin:
    name = "text"
    extensions: set[str] = set()

    def user_prefix(self) -> str:
        return ""

    def clean_model_output(self, text: str) -> str:
        return _clean_model_output_generic(text)

    def strip_noise(self, text: str) -> str:
        return text.strip("\n")

    def postprocess_candidate(self, text: str) -> str:
        return text.replace("\t", "    ").strip("\n")

    def validate_candidate(self, *, original_chunk: str, candidate: str) -> bool:
        return bool(candidate.strip())

    def validate_file_text(self, *, original_text: str, new_text: str) -> FileValidationResult:
        return FileValidationResult(ok=True)

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        return self.validate_file_text(original_text=original_text, new_text=new_text)

    def finalize_file_text(self, *, original_text: str, new_text: str) -> str:
        return new_text if new_text.endswith("\n") else new_text + "\n"


_PY_ARROW_BREAK_RE = re.compile(r"->\s*\n\s*([_a-zA-Z][_a-zA-Z0-9]*)\s*:")
_PY_RAISE_RE = re.compile(r"\braise\s+([_a-zA-Z][_a-zA-Z0-9]*)")
_PY_INPUT_RE = re.compile(r"\binput\s*\(")


def _has_imports_py(text: str) -> bool:
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("import ") or s.startswith("from "):
            return True
    return False


def _has_main_guard_py(text: str) -> bool:
    t = re.sub(r"\s+", "", text)
    return "if__name__=='__main__':" in t or 'if__name__=="__main__":' in t


def _has_input_py(text: str) -> bool:
    return _PY_INPUT_RE.search(text) is not None


def _extract_main_guard_block_py(text: str) -> str | None:
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if "__name__" in ln and "__main__" in ln and ln.lstrip().startswith("if "):
            start = i
            break
    if start is None:
        return None
    guard_ln = lines[start].rstrip("\n")
    guard_indent = len(guard_ln) - len(guard_ln.lstrip(" "))
    block_lines = [guard_ln]
    for ln in lines[start + 1 :]:
        if not ln.strip():
            block_lines.append(ln.rstrip("\n"))
            continue
        indent = len(ln) - len(ln.lstrip(" "))
        if indent > guard_indent:
            block_lines.append(ln.rstrip("\n"))
            continue
        break
    block = "\n".join(block_lines).rstrip("\n")
    return block if block.strip() else None


def _strip_input_lines_py(text: str) -> str:
    out: list[str] = []
    for ln in text.splitlines():
        if _has_input_py(ln):
            continue
        out.append(ln)
    return "\n".join(out).rstrip("\n")


def _restore_python_shape(*, original_text: str, new_text: str) -> str:
    t = new_text.rstrip("\n")
    if (not _has_input_py(original_text)) and _has_input_py(t):
        t = _strip_input_lines_py(t)

    if _has_main_guard_py(original_text) and (not _has_main_guard_py(t)):
        block = _extract_main_guard_block_py(original_text)
        if block:
            t = t.rstrip() + "\n\n" + block
    return t.rstrip("\n") + "\n"


class PythonPlugin(DefaultPlugin):
    name = "python"
    extensions = {".py"}

    def user_prefix(self) -> str:
        return (
            "Extra rules for Python:\n"
            "- Output must be valid Python 3 code (it should parse).\n"
            "- Do not add any imports unless they already appear in the scrambled chunk.\n"
            "- Remove any synthetic noise like '# TODO: fix ...' comments.\n"
            "- Fix indentation.\n"
            "- Preserve program I/O: do not add input() calls unless the scrambled chunk already had them.\n"
            "- Preserve entrypoint structure when present: if the chunk defines main(), keep main() and call it from if __name__ == '__main__'.\n"
            "- Preserve exception types used in raise statements; do not introduce new exception classes.\n"
            "- Output ONLY code.\n\n"
        )

    def strip_noise(self, text: str) -> str:
        lines = text.splitlines()
        out: list[str] = []
        for ln in lines:
            s = ln.lstrip()
            if s.startswith("# TODO: fix "):
                continue
            out.append(ln.rstrip())
        return "\n".join(out).strip("\n")

    def postprocess_candidate(self, text: str) -> str:
        t = text.replace("\t", "    ").strip("\n")
        lines = t.splitlines()
        out_lines: list[str] = []
        for ln in lines:
            if "TODO" in ln:
                if "#" in ln and "TODO" in ln.split("#", 1)[1]:
                    ln = ln.split("#", 1)[0].rstrip()
                else:
                    continue
            out_lines.append(ln.rstrip())
        t = "\n".join(out_lines).strip("\n")
        t = _PY_ARROW_BREAK_RE.sub(r"-> \1:", t)
        return t.strip("\n")

    def validate_candidate(self, *, original_chunk: str, candidate: str) -> bool:
        cand = candidate.strip()
        if not cand:
            return False
        if "//" in cand or "/*" in cand or "*/" in cand:
            return False
        if not _has_imports_py(original_chunk) and _has_imports_py(cand):
            return False
        return True

    def validate_file_text(self, *, original_text: str, new_text: str) -> FileValidationResult:
        try:
            ast.parse(new_text)
        except SyntaxError as e:
            return FileValidationResult(ok=False, reason=f"SyntaxError: {e}")
        return FileValidationResult(ok=True)

    def finalize_file_text(self, *, original_text: str, new_text: str) -> str:
        shaped = _restore_python_shape(original_text=original_text, new_text=new_text)
        try:
            ast.parse(shaped)
        except SyntaxError:
            return new_text if new_text.endswith("\n") else new_text + "\n"
        return shaped


def _tool_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run_tool(args: list[str], *, cwd: Path | None = None, timeout_s: float = 20.0) -> FileValidationResult:
    if not args:
        return FileValidationResult(ok=True)
    if not _tool_exists(args[0]):
        return FileValidationResult(ok=True, reason=f"skipped (missing tool: {args[0]})")
    try:
        p = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return FileValidationResult(ok=False, reason=f"timeout ({args[0]})")
    if p.returncode == 0:
        return FileValidationResult(ok=True)
    msg = (p.stderr or p.stdout or "").strip()
    if len(msg) > 4000:
        msg = msg[:4000] + "\n...truncated..."
    return FileValidationResult(ok=False, reason=f"{args[0]} failed: {msg}")


class JavaScriptPlugin(DefaultPlugin):
    name = "javascript"
    extensions = {".js", ".mjs", ".cjs"}

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        with tempfile.NamedTemporaryFile("w", suffix=path.suffix, delete=True) as f:
            f.write(new_text)
            f.flush()
            return _run_tool(["node", "--check", f.name], cwd=path.parent)


class TypeScriptPlugin(DefaultPlugin):
    name = "typescript"
    extensions = {".ts", ".tsx"}

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        # Prefer project-aware tsc when available; otherwise do a best-effort file check.
        if not _tool_exists("tsc"):
            return FileValidationResult(ok=True, reason="skipped (missing tool: tsc)")
        with tempfile.NamedTemporaryFile("w", suffix=path.suffix, delete=True) as f:
            f.write(new_text)
            f.flush()
            # `tsc` doesn't reliably accept single-file without config; try anyway.
            return _run_tool(["tsc", "--noEmit", "--pretty", "false", f.name], cwd=path.parent, timeout_s=60.0)


class BashPlugin(DefaultPlugin):
    name = "bash"
    extensions = {".sh", ".bash"}

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        with tempfile.NamedTemporaryFile("w", suffix=path.suffix, delete=True) as f:
            f.write(new_text)
            f.flush()
            return _run_tool(["bash", "-n", f.name], cwd=path.parent)


class PhpPlugin(DefaultPlugin):
    name = "php"
    extensions = {".php"}

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        with tempfile.NamedTemporaryFile("w", suffix=path.suffix, delete=True) as f:
            f.write(new_text)
            f.flush()
            return _run_tool(["php", "-l", f.name], cwd=path.parent)


class JavaPlugin(DefaultPlugin):
    name = "java"
    extensions = {".java"}

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        # Best-effort: compile the single file. Classpath/project builds are handled at project gate level.
        if not _tool_exists("javac"):
            return FileValidationResult(ok=True, reason="skipped (missing tool: javac)")
        with tempfile.TemporaryDirectory() as d:
            tmp_dir = Path(d)
            tmp_java = tmp_dir / path.name
            tmp_java.write_text(new_text, encoding="utf-8")
            return _run_tool(["javac", "-d", str(tmp_dir), str(tmp_java)], cwd=path.parent, timeout_s=60.0)


class CppPlugin(DefaultPlugin):
    name = "cpp"
    extensions = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"}

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        # Best-effort syntax-only. Prefer clang++ if available.
        compiler = "clang++" if _tool_exists("clang++") else ("g++" if _tool_exists("g++") else None)
        if not compiler:
            return FileValidationResult(ok=True, reason="skipped (missing tool: clang++/g++)")
        with tempfile.NamedTemporaryFile("w", suffix=path.suffix, delete=True) as f:
            f.write(new_text)
            f.flush()
            return _run_tool([compiler, "-fsyntax-only", f.name], cwd=path.parent, timeout_s=60.0)


class GoPlugin(DefaultPlugin):
    name = "go"
    extensions = {".go"}

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        if not _tool_exists("gofmt"):
            return FileValidationResult(ok=True, reason="skipped (missing tool: gofmt)")
        # gofmt parses; if it fails it's syntactically invalid.
        with tempfile.NamedTemporaryFile("w", suffix=path.suffix, delete=True) as f:
            f.write(new_text)
            f.flush()
            return _run_tool(["gofmt", f.name], cwd=path.parent)


class RustPlugin(DefaultPlugin):
    name = "rust"
    extensions = {".rs"}

    def validate_file_path(self, *, path: Path, original_text: str, new_text: str) -> FileValidationResult:
        # Rust parsing without cargo is hard; keep this as a graceful no-op.
        return FileValidationResult(ok=True, reason="skipped (rust requires project-level cargo check)")


_PLUGINS: dict[str, LanguagePlugin] = {
    "python": PythonPlugin(),
    "javascript": JavaScriptPlugin(),
    "typescript": TypeScriptPlugin(),
    "bash": BashPlugin(),
    "php": PhpPlugin(),
    "java": JavaPlugin(),
    "cpp": CppPlugin(),
    "go": GoPlugin(),
    "rust": RustPlugin(),
    "text": DefaultPlugin(),
}


def get_plugin(language: str) -> LanguagePlugin:
    return _PLUGINS.get(language.lower(), _PLUGINS["text"])

