#!/usr/bin/env bash
# Kinetrace launcher for Linux (Ubuntu 22.04 or newer) and macOS (Apple Silicon,
# macOS 14 or newer). The first run creates a self-contained .venv inside this
# folder and installs everything into it; deleting the folder uninstalls it.
set -e
cd "$(dirname "$0")"

pick_python() {
    for p in python3.12 python3.13 python3.11 python3.10 python3.14 python3; do
        if command -v "$p" >/dev/null 2>&1 &&
           "$p" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)' 2>/dev/null; then
            echo "$p"; return 0
        fi
    done
    return 1
}

if [ ! -x ".venv/bin/python" ] || [ ! -f ".venv/.kinetrace-installed" ]; then
    echo "============================================================"
    echo " Kinetrace - first run: creating a self-contained environment in .venv"
    echo " This downloads up to ~4 GB (PyTorch) and can take a while."
    echo " Everything installs INSIDE this folder."
    echo "============================================================"
    PY="$(pick_python)" || {
        echo "ERROR: Python 3.10 - 3.14 not found."
        if [ "$(uname)" = "Darwin" ]; then
            echo "Install it from https://www.python.org/downloads/ (or: brew install python@3.12), then run ./run.sh again."
        else
            echo "Ubuntu: sudo apt install python3 python3-venv   then run ./run.sh again."
        fi
        exit 1
    }
    if [ ! -x ".venv/bin/python" ]; then
        "$PY" -m venv .venv || {
            echo "ERROR: could not create the environment."
            echo "Ubuntu: sudo apt install python3-venv   (or python3.12-venv), then run ./run.sh again."
            rm -rf .venv
            exit 1
        }
    fi
    .venv/bin/python install.py
    touch .venv/.kinetrace-installed
fi

if [ "$(uname)" = "Linux" ] && command -v ldconfig >/dev/null 2>&1; then
    # Qt 6 on X11 / Wayland needs a few system libraries that minimal installs lack
    missing=""
    for lib in libxcb-cursor.so.0 libEGL.so.1 libxkbcommon-x11.so.0 libGL.so.1; do
        ldconfig -p | grep -q "$lib" || missing="$missing $lib"
    done
    if [ -n "$missing" ]; then
        echo "Note: missing system libraries for the window:$missing"
        echo "Ubuntu: sudo apt install libxcb-cursor0 libegl1 libxkbcommon-x11-0 libgl1"
    fi
fi

exec .venv/bin/python -m cotracker_app "$@"
