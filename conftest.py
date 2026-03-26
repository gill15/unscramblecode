from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable as a module during tests (no packaging step required).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

