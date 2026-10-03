"""Install Kinetrace's Python packages into the running interpreter (the
folder's own .venv - run.bat / run.sh create it and call this script; they
also fetch a private Python into .venv/base first when the computer has none).

Runs without any input. Safe to run again at any time: a finished environment
is recognised in a second, an interrupted one keeps what it installed and goes on.

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
(about 1 MB); its 66 MB checkpoint downloads the first time you press Track.
Without it the app falls back to CoTracker3.

Finally the environment is verified (every package imports) and the hardware
report is printed: which GPU was found, what runs on it, what is slower or
switched off on this computer (`python -m kinetrace --check` prints it again).
A marker file, .venv/kinetrace-install.json, records a finished install; the
launchers start the app only when it is there. Standard library only.

    install.py                   install (or verify) everything
    install.py --force           reinstall the packages even if they import (PyTorch with
                                 --force-reinstall, the others with --upgrade)
    install.py --alltracker-only fetch AllTracker's code only (the launchers retry a missed fetch)
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

TORCH = "2.12.1"
TORCHVISION = "0.27.1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# the app's own modules that need nothing installed (standard library only at import): the driver
# rule and the pinned AllTracker code live there once, not copied here (R21)
from kinetrace import device, downloads  # noqa: E402

DRIVER_MIN = device.DRIVER_MIN              # NVIDIA driver branch the CUDA 13 build needs
# the AllTracker commit Kinetrace's alltracker_backend.py was verified against
ALLTRACKER_SHA = downloads.CODE["alltracker"].commit
ALLTRACKER_DIR = str(downloads.CODE["alltracker"].dest)
MARKER = os.path.join(sys.prefix, "kinetrace-install.json")
# what a working environment must be able to import (checked in a fresh process)
IMPORTS = "import PySide6.QtWidgets, cv2, numpy, scipy, torch, torchvision, transformers, PIL, imageio_ffmpeg"


AGAIN = ("run the launcher again: what is already installed is kept and the install goes on from "
         "there (a package cut off half-way downloads again).")


def say(msg: str) -> None:
    print(msg, flush=True)


nvidia_driver = device.nvidia_driver        # (driver version text, major) or None without an NVIDIA driver


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


# Qt's system libraries on Linux: the file Qt fails to open -> the Ubuntu / Debian package (the
# list run.sh checks with ldconfig)
LINUX_LIBS = {"libxcb-cursor.so.0": "libxcb-cursor0", "libEGL.so.1": "libegl1",
              "libxkbcommon-x11.so.0": "libxkbcommon-x11-0", "libGL.so.1": "libgl1",
              "libxkbcommon.so.0": "libxkbcommon0", "libdbus-1.so.3": "libdbus-1-3",
              "libfontconfig.so.1": "libfontconfig1"}


def system_libs_hint(why: str) -> str | None:
    """When an import failed because a Linux system library is missing ("libEGL.so.1: cannot open
    shared object file"), the sentence + the apt-get line that fixes it; else None (G98). Reinstalling
    the Python packages cannot help with this, so it must not be advised."""
    import re
    if "cannot open shared object file" not in str(why):
        return None
    m = re.search(r"(lib[\w.+-]+\.so[.\d]*): cannot open shared object file", str(why))
    named = m.group(1) if m else ""
    pkgs = [LINUX_LIBS[named]] if named in LINUX_LIBS else []
    if shutil.which("ldconfig"):
        try:
            have = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=20).stdout
            pkgs += [pkg for lib, pkg in LINUX_LIBS.items() if lib not in have and pkg not in pkgs]
        except (OSError, subprocess.SubprocessError):
            pass
    if not pkgs:
        pkgs = list(LINUX_LIBS.values())
    return ("The window needs system libraries that are not installed on this computer" +
            (f" ({named})" if named else "") + ". The Python packages themselves are fine: install the "
            "libraries, then start the launcher again (nothing is downloaded again). On Ubuntu / Debian:\n"
            "    sudo apt-get install -y " + " ".join(pkgs))


EXIT_SYSTEM_LIBS = 3        # install.py's exit code for "a system library is missing"; run.sh words it


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
    """AllTracker's code into models/alltracker (skipped when already there):
    the pinned commit's zip, used only when its Python files have the pinned
    digest and every member stays inside the folder (kinetrace/downloads.py, I153)."""
    if downloads.code_present("alltracker"):
        return
    c = downloads.CODE["alltracker"]
    say(f"+ fetching AllTracker (MIT licence), commit {c.commit[:12]}, from github.com/{c.repo}")
    try:
        downloads.ensure_code("alltracker")
    except Exception as e:  # noqa: BLE001 - the app still works with CoTracker3
        say(f"  could not fetch AllTracker ({e}); the app uses CoTracker3 for now and tries again "
            "the next time it is started with an internet connection.")
        shutil.rmtree(ALLTRACKER_DIR, ignore_errors=True)


def hardware_report() -> None:
    """The same text as `python -m kinetrace --check` (no Qt is imported)."""
    try:
        say("")
        say(device.describe())
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
        ok, why = imports_ok()
        hint = None if ok else system_libs_hint(why)
        if hint:                                    # not a missing package: no reinstall (G98)
            say(hint)
            return EXIT_SYSTEM_LIBS
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
    say("Step 1 of 4: the package installer (pip)")
    pip("--upgrade", "pip")
    wheels = [f"torch=={TORCH}", f"torchvision=={TORCHVISION}"]
    # --force must really reinstall: pip treats "==2.12.1" as met by an installed 2.12.1+cpu, so the
    # wrong build (CPU on a GPU machine) would stay without --force-reinstall (I238)
    again = ["--force-reinstall"] if force else []
    say("Step 2 of 4: PyTorch, the deep-learning engine (" + ("about 3 GB with the CUDA libraries: the longest "
        "step" if choice == "cuda" else "a few hundred MB") + ")")
    if choice == "cuda":
        pip(*wheels, *again, "--index-url", "https://download.pytorch.org/whl/cu130")
    elif choice == "cpu" and sys.platform != "darwin" and platform.machine().upper() not in ("ARM64", "AARCH64"):
        pip(*wheels, *again, "--index-url", "https://download.pytorch.org/whl/cpu")
    else:
        pip(*wheels, *again)
    say("Step 3 of 4: the other packages (Qt for the window, OpenCV, transformers, ...: about 500 MB)")
    pip(*(["--upgrade"] if force else []), "-r", os.path.join(HERE, "requirements.txt"))
    say("Step 4 of 4: AllTracker's code (about 1 MB)")
    fetch_alltracker()

    ok, why = imports_ok()
    if not ok:
        say("")
        say(f"Installation finished but a package does not import: {why}")
        hint = system_libs_hint(why)
        if hint:                                    # (G98) a missing Linux library, not a broken environment
            say(hint)
            return EXIT_SYSTEM_LIBS
        say("Delete the .venv folder and start the launcher again; if it happens twice, send the lines above "
            "with your question.")
        return 1
    write_marker(choice)
    say("Kinetrace install: done. The tracking and segmentation models (66 MB - 620 MB each) are downloaded "
        "the first time each is used, with a progress window in the app; they stay in models/.")
    hardware_report()
    return 0


if __name__ == "__main__":
    if "--alltracker-only" in sys.argv:        # the launchers retry a missed fetch on start
        fetch_alltracker()
        sys.exit(0)
    try:
        sys.exit(main(force="--force" in sys.argv))
    except subprocess.CalledProcessError as e:
        say(f"\nInstallation failed ({e}). Check the internet connection and " + AGAIN)
        sys.exit(1)
    except KeyboardInterrupt:
        say("\nInstallation interrupted. To finish it, " + AGAIN)
        sys.exit(1)
