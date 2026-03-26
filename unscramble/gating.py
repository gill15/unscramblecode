from __future__ import annotations

import difflib


def change_ratio(original: str, new: str) -> float:
    """
    Rough minimal-edit proxy.
    Returns fraction of changed lines in a unified diff-like sense.
    """
    o = original.splitlines()
    n = new.splitlines()
    sm = difflib.SequenceMatcher(a=o, b=n)
    changed = 0
    total = max(1, len(o))
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changed += (i2 - i1)
    return changed / total


def passes_minimal_edit(original: str, new: str, *, max_change_ratio: float) -> bool:
    if original == new:
        return True
    return change_ratio(original, new) <= max_change_ratio

