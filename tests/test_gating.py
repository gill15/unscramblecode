from __future__ import annotations

from unscramble.gating import change_ratio, passes_minimal_edit


def test_change_ratio_basic() -> None:
    a = "x\\ny\\nz\\n"
    b = "x\\ny\\nq\\n"
    r = change_ratio(a, b)
    assert 0.0 < r <= 1.0


def test_passes_minimal_edit_threshold() -> None:
    a = "a\\nb\\nc\\n"
    b = "a\\nX\\nY\\n"
    assert passes_minimal_edit(a, b, max_change_ratio=1.0)
    assert not passes_minimal_edit(a, b, max_change_ratio=0.0)

