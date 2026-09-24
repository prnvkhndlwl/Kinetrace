#!/usr/bin/env bash
# =============================================================================
#  Kinetrace launcher for Linux (Ubuntu 22.04 or newer, x86-64 or ARM) and
#  macOS (Apple Silicon, macOS 14 or newer).   ./run.sh        (Mac: Kinetrace.command)
#
#  First run: everything is set up INSIDE this folder without any questions -
#  a private Python if the computer has none that can make virtual environments
#  (.venv/base), then the packages (.venv). Later runs start the app at once.
#  Deleting the folder uninstalls everything. "./run.sh --check" prints what
#  this computer can run and exits.
# =============================================================================
set -e
cd "$(dirname "$0")"

OS="$(uname -s)"
ARCH="$(uname -m)"
PYCHECK='import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1); import venv, ensurepip'

say() { printf '%s\n' "$*"; }

# ---- a usable Python already on the computer? -------------------------------
pick_python() {
    if [ -x ".venv/base/bin/python3" ] && ".venv/base/bin/python3" -c "$PYCHECK" >/dev/null 2>&1; then
        echo ".venv/base/bin/python3"; return 0
    fi
    for p in python3.12 python3.13 python3.11 python3.10 python3.14 python3; do
        exe="$(command -v "$p" 2>/dev/null)" || continue
        # macOS: /usr/bin/python3 is a stub that pops up "install the developer
        # tools?" when Xcode's Command Line Tools are missing - never poke it then
        if [ "$OS" = "Darwin" ] && [ "$exe" = "/usr/bin/python3" ] && ! xcode-select -p >/dev/null 2>&1; then
            continue
        fi
        # the version AND the venv + ensurepip modules (Ubuntu ships python3
        # without them: python3-venv is a separate package)
        if "$exe" -c "$PYCHECK" >/dev/null 2>&1; then
            echo "$exe"; return 0
        fi
    done
    return 1
}

download() {   # url, destination
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=5 -O "$2" "$1"
    else
        say "Neither curl nor wget is available to download files."; return 1
    fi
}

# ---- none: fetch a private, relocatable CPython into .venv/base -------------
# (python-build-standalone by Astral, the builds behind uv; PSF licence,
# nothing is installed into the operating system)
bootstrap_python() {
    TAG=20260901
    VER=3.12.14
    case "$OS-$ARCH" in
        Linux-x86_64)               TRIPLE=x86_64-unknown-linux-gnu ;;
        Linux-aarch64|Linux-arm64)  TRIPLE=aarch64-unknown-linux-gnu ;;
        Darwin-arm64)               TRIPLE=aarch64-apple-darwin ;;
        *) say "No private Python build exists for $OS on $ARCH."; return 1 ;;
    esac
    URL="https://github.com/astral-sh/python-build-standalone/releases/download/$TAG/cpython-$VER+$TAG-$TRIPLE-install_only_stripped.tar.gz"
    say "No usable Python found on this computer - downloading a private copy (about 30 MB) into .venv/base ..."
    rm -rf .venv/base .venv/_python_dl
    mkdir -p .venv/_python_dl
    download "$URL" .venv/_python_dl/python.tar.gz || { rm -rf .venv/_python_dl; return 1; }
    tar -xzf .venv/_python_dl/python.tar.gz -C .venv/_python_dl
    mv .venv/_python_dl/python .venv/base
    rm -rf .venv/_python_dl
    [ -x ".venv/base/bin/python3" ]
}

# ---- Qt needs a few system libraries on Linux -------------------------------
linux_libs() {
    command -v ldconfig >/dev/null 2>&1 || return 0
    missing=""
    pkgs=""
    for pair in libxcb-cursor.so.0:libxcb-cursor0 libEGL.so.1:libegl1 libxkbcommon-x11.so.0:libxkbcommon-x11-0 \
                libGL.so.1:libgl1 libxkbcommon.so.0:libxkbcommon0 libdbus-1.so.3:libdbus-1-3 libfontconfig.so.1:libfontconfig1; do
        lib="${pair%%:*}"; pkg="${pair##*:}"
        ldconfig -p | grep -q "$lib" || { missing="$missing $lib"; pkgs="$pkgs $pkg"; }
    done
    [ -n "$missing" ] || return 0
    say "The window needs system libraries that are not installed:$missing"
    if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1 && [ -t 0 ]; then
        say "Installing them with apt (this is the ONE step that needs your password):"
        say "    sudo apt-get install -y$pkgs"
        sudo apt-get install -y $pkgs || say "apt could not install them; the app may not open a window until they are installed."
    else
        say "Install them, then start ./run.sh again. On Ubuntu / Debian:"
        say "    sudo apt-get install -y$pkgs"
    fi
}

if [ ! -f ".venv/kinetrace-install.json" ]; then
    if [ -x ".venv/bin/python" ]; then
        say "Kinetrace - checking the environment in .venv ..."
    else
        say "============================================================"
        say " Kinetrace - first run: setting up a self-contained environment in .venv"
        say " This downloads 1-4 GB (PyTorch; the CUDA build with an NVIDIA graphics"
        say " card is the largest) and can take a while. Everything installs INSIDE"
        say " this folder - deleting the folder removes the tool completely."
        say "============================================================"
    fi
    if [ "$OS" = "Darwin" ] && [ "$ARCH" != "arm64" ]; then
        say "ERROR: Kinetrace needs a Mac with Apple Silicon (M1 or later); PyTorch no longer builds for Intel Macs."
        exit 1
    fi
    if [ ! -x ".venv/bin/python" ]; then
        # KINETRACE_BOOTSTRAP_PYTHON=1 skips the search and always uses a private
        # Python (a broken system Python; the CI's bootstrap job)
        if [ "${KINETRACE_BOOTSTRAP_PYTHON:-}" = "1" ]; then
            PY=""
        else
            PY="$(pick_python)" || PY=""
        fi
        if [ -z "$PY" ]; then
            bootstrap_python || {
                say ""
                say "ERROR: could not download Python. Check the internet connection and run ./run.sh again."
                if [ "$OS" = "Darwin" ]; then
                    say "Or install Python 3.12 from https://www.python.org/downloads/ and run it again."
                else
                    say "Or:  sudo apt install python3 python3-venv   and run it again."
                fi
                exit 1
            }
            PY=".venv/base/bin/python3"
        fi
        say "Creating the environment with $PY ..."
        "$PY" -m venv .venv || {
            say "ERROR: could not create the environment. Delete the .venv folder and run ./run.sh again."
            exit 1
        }
    fi
    # install.py picks the PyTorch build, installs requirements.txt, fetches
    # AllTracker, verifies every import and writes .venv/kinetrace-install.json;
    # safe to re-run, it resumes where it stopped
    .venv/bin/python install.py || {
        say ""
        say "ERROR: the installation did not finish. Check your internet connection and run ./run.sh"
        say "again (it resumes). If it fails twice, send the lines above with your question."
        exit 1
    }
    [ "$OS" = "Linux" ] && linux_libs
    [ "$1" = "--check" ] && exit 0
fi

# The private interpreter (if any) lives inside the folder. A moved or renamed
# folder leaves .venv/bin/python pointing at the old path: re-link it in place
# (python -m venv on an existing venv only rewrites the links and pyvenv.cfg).
if ! .venv/bin/python -c "pass" >/dev/null 2>&1 && [ -x ".venv/base/bin/python3" ]; then
    say "Folder was moved - relinking .venv"
    .venv/base/bin/python3 -m venv .venv
fi

[ -f models/alltracker/nets/alltracker.py ] || .venv/bin/python install.py --alltracker-only

exec .venv/bin/python -m kinetrace "$@"
