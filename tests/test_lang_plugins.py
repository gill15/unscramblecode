from __future__ import annotations

from pathlib import Path

from unscramble.lang_plugins import get_plugin


def test_python_plugin_parses() -> None:
    p = get_plugin("python")
    src = "def f(x: int) -> int:\n    return x + 1\n"
    v = p.validate_file_text(original_text=src, new_text=src)
    assert v.ok


def test_js_plugin_skips_without_node(tmp_path: Path) -> None:
    # This test is intentionally weak: if node exists it may actually validate.
    p = get_plugin("javascript")
    js = "export function f(x){ return x+1; }\\n"
    fake = tmp_path / "x.js"
    v = p.validate_file_path(path=fake, original_text=js, new_text=js)
    assert v.ok

