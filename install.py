"""Install Kinetrace's Python packages into the running interpreter (the
folder's own .venv - run.bat / run.sh create it and call this script; they
also fetch a private Python into .venv/base first when the computer has none).

Runs without any input. Safe to run again at any time: a finished environment
is recognised in a second, an interrupted one resumes where it stopped.

The only platform-specific part is PyTorch:

    Windows or Linux with an NVIDIA GPU  -> CUDA 13 build (download.pytorch.org/whl/cu130)
    Windows or Linux without one         -> CPU build     (download.pytorch.org/whl/cpu)
    macOS on Apple Silicon               -> the PyPI build (runs on the GPU through Metal / MPS)
    macOS on Intel                       -> not supported (PyTorch no longer builds for it)

Override the choice with KINETRACE_TORCH=cuda | cpu | mps, e.g. to install the
CPU build on a machine whose NVIDIA driver is too old for CUDA 13 (R580+).
Everything else comes from requirements.txt.

Then the source code of AllTracker (Harley et al., ICCV 2025, MIT licence),
the default point model, is fetched into models/alltracker at a pinned commit
(277 KB); its 63 MB checkpoint downloads the first time you press Track.
Without it the app falls back to CoTracker3.

Finally the environment is verified (every package imports) and the hardware
report is printed: which GPU was found, what runs on it, what is slower or
switched off on this computer (`python -m kinetrace --check` prints it again).
A marker file, .venv/kinetrace-install.json, records a finished install; the
launchers start the app only when it is there. Standard library only.

    install.py                   install (or verify) everything
    install.py --force           reinstall the packages even if they import
    install.py --alltracker-only fetch AllTracker's code only (the launchers retry a missed fetch)
"""
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

TORCH = "2.12.1"
TORCHVISION = "0.27.1"
DRIVER_MIN = 580            # NVIDIA driver branch the CUDA 13 build needs
HERE = os.path.dirname(os.path.abspath(__file__))
# the AllTracker commit Kinetrace's alltracker_backend.py was verified against
ALLTRACKER_SHA = "e7553135e7b361590dbccd10e2b274b024f41cd6"
ALLTRACKER_DIR = os.path.join(HERE, "models", "alltracker")
MARKER = os.path.join(sys.prefix, "kinetrace-install.json")
# what a working environment must be able to import (checked in a fresh process)
IMPORTS = "import PySide6.QtWidgets, cv2, numpy, scipy, torch, torchvision, transformers, PIL, imageio_ffmpeg"


def say(msg: str) -> None:
    print(msg, flush=True)


def nvidia_driver():
    """(driver version text, major) from nvidia-smi, or None without an NVIDIA driver."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    ver = r.stdout.strip().splitlines()[0].strip()
    try:
        return ver, int(ver.split(".")[0])
    except ValueError:
        return ver, 0


def has_nvidia() -> bool:
    return nvidia_driver() is not None


def torch_choice() -> str:
    forced = os.environ.get("KINETRACE_TORCH", "").strip().lower()
    if forced in ("cuda", "cpu", "mps"):
        return forced
    if sys.platform == "darwin":
        return "mps"
    return "cuda" if has_nvidia() else "cpu"


def pip(*args: str) -> None:
    # retries and a generous timeout: the PyTorch wheel is 2-3 GB and home
    # connections drop; --no-cache-dir keeps the folder from doubling in size
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--retries", "5",
           "--timeout", "90", *args]
    say("+ " + " ".join(cmd))
    subprocess.check_call(cmd)


def imports_ok() -> tuple[bool, str]:
    """Does a fresh interpreter import every package? (the test the launchers
    trust: a half-finished install crashed on `import torch` at start-up)."""
    r = subprocess.run([sys.executable, "-c", IMPORTS], capture_output=True, text=True)
    if r.returncode == 0:
        return True, ""
    lines = [ln for ln in (r.stderr or r.stdout or "").strip().splitlines() if ln.strip()]
    return False, (lines[-1] if lines else f"exit code {r.returncode}")


def torch_matches() -> bool:
    """Is the installed torch the version this install.py pins? (a newer
    Kinetrace that moves the pin must reinstall it, I142)"""
    try:
        from importlib.metadata import version
        return version("torch").split("+")[0] == TORCH
    except Exception:
        return False


def write_marker(choice: str) -> None:
    info = {"torch_build": choice, "torch": TORCH, "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "installed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(MARKER, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)


def fetch_alltracker() -> None:
    """AllTracker's code into models/alltracker (skipped when already there)."""
    if os.path.exists(os.path.join(ALLTRACKER_DIR, "nets", "alltracker.py")):
        return
    url = f"https://github.com/aharley/alltracker/archive/{ALLTRACKER_SHA}.zip"
    say(f"+ fetching AllTracker (MIT licence) from {url}")
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        prefix = z.namelist()[0]
        for name in z.namelist():
            rel = name[len(prefix):]
            if not rel or name.endswith("/"):
                continue
            dest = os.path.join(ALLTRACKER_DIR, *rel.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(z.read(name))
    except Exception as e:  # noqa: BLE001 - the app still works with CoTracker3
        say(f"  could not fetch AllTracker ({e}); the app uses CoTracker3 for now and tries again "
            "the next time it is started with an internet connection.")
        shutil.rmtree(ALLTRACKER_DIR, ignore_errors=True)


def hardware_report() -> None:
    """The same text as `python -m kinetrace --check` (no Qt is imported)."""
    try:
        sys.path.insert(0, HERE)
        from kinetrace.device import describe
        say("")
        say(describe())
    except Exception as e:      # noqa: BLE001 - a report must never fail an install
        say(f"(hardware report unavailable: {e})")


def main(force: bool = False) -> int:
    if sys.version_info < (3, 10) or sys.version_info >= (3, 15):
        say(f"Kinetrace needs Python 3.10 - 3.14 (3.12 recommended); this is {platform.python_version()}.")
        return 1
    if sys.platform == "darwin" and platform.machine() != "arm64":
        say("Kinetrace needs a Mac with Apple Silicon (M1 or later): PyTorch no longer builds for Intel Macs.")
        return 1
    if sys.prefix == sys.base_prefix:
        say("Note: installing into the interpreter itself, not a virtual environment. run.bat / run.sh "
            "make one in .venv; this is fine for a CI machine.")

    choice = torch_choice()
    if not force:
        ok, _ = imports_ok()
        if ok and torch_matches():
            # the launchers only come here when the marker is missing -- a fresh
            # folder, or an update that changed requirements.txt / install.py and
            # deleted it (I142): install what the new list adds (a no-op, and no
            # network, when everything is already there) instead of stopping at
            # "every package imports"
            pip("-r", os.path.join(HERE, "requirements.txt"))
            fetch_alltracker()
            write_marker(choice)
            say("Kinetrace install: already complete (every package imports). "
                "Run with --force to reinstall the packages.")
            hardware_report()
            return 0

    say(f"Kinetrace install: Python {platform.python_version()} on {platform.system()} {platform.machine()}, "
        f"PyTorch build: {choice}")
    drv = nvidia_driver()
    if choice == "cuda" and drv is not None and drv[1] and drv[1] < DRIVER_MIN:
        say("")
        say(f"WARNING: the NVIDIA driver on this computer is {drv[0]}; the CUDA 13 build of PyTorch needs "
            f"{DRIVER_MIN} or newer. The install goes ahead, but the graphics card will not be used until the "
            "driver is updated (nvidia.com/drivers) - Kinetrace then runs on the CPU and says so. To install "
            "the CPU build instead, delete the .venv folder, set KINETRACE_TORCH=cpu and start the launcher again.")
        say("")
    if sys.platform == "win32" and platform.machine().upper() in ("ARM64", "AARCH64"):
        say("Note: Windows on ARM - PyTorch and Qt for this machine come from PyPI (CPU only).")
    pip("--upgrade", "pip")
    wheels = [f"torch=={TORCH}", f"torchvision=={TORCHVISION}"]
    if choice == "cuda":
        pip(*wheels, "--index-url", "https://download.pytorch.org/whl/cu130")
    elif choice == "cpu" and sys.platform != "darwin" and platform.machine().upper() not in ("ARM64", "AARCH64"):
        pip(*wheels, "--index-url", "https://download.pytorch.org/whl/cpu")
    else:
        pip(*wheels)
    pip("-r", os.path.join(HERE, "requirements.txt"))
    fetch_alltracker()

    ok, why = imports_ok()
    if not ok:
        say("")
        say(f"Installation finished but a package does not import: {why}")
        say("Delete the .venv folder and start the launcher again; if it happens twice, send the lines above "
            "with your question.")
        return 1
    write_marker(choice)
    say("Kinetrace install: done.")
    hardware_report()
    return 0


if __name__ == "__main__":
    if "--alltracker-only" in sys.argv:        # the launchers retry a missed fetch on start
        fetch_alltracker()
        sys.exit(0)
    try:
        sys.exit(main(force="--force" in sys.argv))
    except subprocess.CalledProcessError as e:
        say(f"\nInstallation failed ({e}). Check the internet connection and run the launcher again: "
            "it resumes where it stopped.")
        sys.exit(1)
    except KeyboardInterrupt:
        say("\nInstallation interrupted. Run the launcher again: it resumes where it stopped.")
        sys.exit(1)
