#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${WAYLAND_DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  # Prefer Wayland when available; avoids missing xcb deps on some systems.
  export QT_QPA_PLATFORM=wayland
fi

if [[ -x ".venv/bin/python" ]]; then
  exec ".venv/bin/python" -m unscramble_gui.app
fi

exec python3 -m unscramble_gui.app

