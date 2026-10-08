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
ISSUES = "https://github.com/prnvkhndlwl/Kinetrace/issues"


def _new_log() -> str | None:
    """logs/install-<date>.log: everything this run says plus pip's own full log (pip --log), one
    file a support request can ask for (Mac install audit P2-8 / P2-11). None if logs/ cannot be
    written: the install goes on without it."""
    try:
        d = os.path.join(HERE, "logs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, time.strftime("install-%Y-%m-%d-%H%M%S.log"))
        with open(path, "a", encoding="utf-8"):
            pass
        return path
    except OSError:
        return None


LOG = None if __name__ != "__main__" else _new_log()


def say(msg: str) -> None:
    print(msg, flush=True)
    if LOG:
        try:
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass


def step(msg: str) -> None:
    """A step line, set apart from pip's output so it is not buried."""
    say("")
    say("==== " + msg)


def where_to_ask() -> str:
    try:
        log = f" and attach {os.path.relpath(LOG, HERE)}" if LOG else ""
    except ValueError:                  # another drive (Windows): the full path then
        log = f" and attach {LOG}"
    return f"If it happens twice, open an issue at {ISSUES}{log}."


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
    if LOG:
        cmd += ["--log", LOG]       # pip's full log goes to the file; the console keeps its progress bars
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


SAM3_NOTE = """\
SAM 3 - the best silhouettes (optional). Its weights are GATED: Meta must say yes first.

The easy way (no Terminal):
  1. Make a free account at https://huggingface.co and open https://huggingface.co/facebook/sam3 :
     "Agree and access repository", then wait until that page says you have access.
  2. Make a READ token: https://huggingface.co/settings/tokens -> Create new token -> Read.
     (A token is a password for downloads only; keep it to yourself.)
  3. In Kinetrace: Settings (Ctrl+, ; Cmd+, on a Mac) -> Hugging Face token: paste it, Save token;
     Segmentation model: SAM 3. The first outline downloads it (3.4 GB) into models/hf with a
     progress window; it stays inside the Kinetrace folder.

Or put the files HERE yourself (this folder, models/sam3). Kinetrace uses it once it holds
config.json and model.safetensors; these are the files it reads (sam3.pt, 3.45 GB, is NOT needed):
    model.safetensors        3.44 GB
    config.json, processor_config.json, tokenizer.json, tokenizer_config.json,
    special_tokens_map.json, vocab.json, merges.txt          (small)
From a Terminal in the Kinetrace folder (Windows: .venv\\Scripts\\python.exe instead of .venv/bin/python):
    .venv/bin/python -m huggingface_hub.cli.hf download facebook/sam3 --revision {rev} --exclude "sam3.pt" --local-dir models/sam3 --token YOUR_READ_TOKEN
"""

S3DB_NOTE = """\
SAM 3D Body - 3D human joints and a body mesh (optional). It runs ONLY on an NVIDIA graphics card
(Meta's code is CUDA-only); on any other computer use ViTPose (2D joints), which needs nothing.

Two things are needed, both from Meta:
  1. The weights (GATED): request access at https://huggingface.co/facebook/sam-3d-body-dinov3 ,
     then put these files HERE (this folder, models/sam-3d-body-dinov3):
         model.ckpt                2.1 GB
         assets/mhr_model.pt       0.7 GB   (in the sub-folder assets - easy to miss)
     From a Terminal in the Kinetrace folder (Windows: .venv\\Scripts\\python.exe):
         .venv/bin/python -m huggingface_hub.cli.hf download facebook/sam-3d-body-dinov3 model.ckpt assets/mhr_model.pt --local-dir models/sam-3d-body-dinov3 --token YOUR_READ_TOKEN
  2. Meta's inference code, cloned into models/sam-3d-body (a NEW folder; it must contain
     sam_3d_body/):
         git clone https://github.com/facebookresearch/sam-3d-body models/sam-3d-body
Body -> 3D body checks both and says what is still missing.
"""


def gated_model_folders() -> None:
    """The folders the gated models go into, each with PUT_FILES_HERE.txt naming the exact files,
    their sizes and the commands (Mac install audit P1: the user had to create them with exact
    names, and a typo was ignored without a word). SAM 3D Body's only where it can run (not on a
    Mac). Never models/sam-3d-body itself: git clone refuses a folder that is not empty. A model
    folder counts only once it holds the weights, so these notes change nothing in the app."""
    notes = {"sam3": SAM3_NOTE.format(rev=downloads.HF_REVISIONS["facebook/sam3"])}
    if sys.platform != "darwin":
        notes["sam-3d-body-dinov3"] = S3DB_NOTE
    for name, text in notes.items():
        try:
            d = os.path.join(downloads.MODELS_DIR, name)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "PUT_FILES_HERE.txt"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        except OSError as e:          # a read-only models/ must not fail the install
            say(f"(could not write models/{name}/PUT_FILES_HERE.txt: {e})")


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
            gated_model_folders()
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
    step("Step 1 of 4: the package installer (pip)")
    pip("--upgrade", "pip")
    wheels = [f"torch=={TORCH}", f"torchvision=={TORCHVISION}"]
    # --force must really reinstall: pip treats "==2.12.1" as met by an installed 2.12.1+cpu, so the
    # wrong build (CPU on a GPU machine) would stay without --force-reinstall (I238)
    again = ["--force-reinstall"] if force else []
    step("Step 2 of 4: PyTorch, the deep-learning engine (" + ("about 3 GB with the CUDA libraries: the longest "
        "step" if choice == "cuda" else "a few hundred MB") + ")")
    if choice == "cuda":
        pip(*wheels, *again, "--index-url", "https://download.pytorch.org/whl/cu130")
    elif choice == "cpu" and sys.platform != "darwin" and platform.machine().upper() not in ("ARM64", "AARCH64"):
        pip(*wheels, *again, "--index-url", "https://download.pytorch.org/whl/cpu")
    else:
        pip(*wheels, *again)
    step("Step 3 of 4: the other packages (Qt for the window, OpenCV, transformers, ...: about 500 MB)")
    pip(*(["--upgrade"] if force else []), "-r", os.path.join(HERE, "requirements.txt"))
    step("Step 4 of 4: AllTracker's code (about 1 MB)")
    fetch_alltracker()
    gated_model_folders()

    ok, why = imports_ok()
    if not ok:
        say("")
        say(f"Installation finished but a package does not import: {why}")
        hint = system_libs_hint(why)
        if hint:                                    # (G98) a missing Linux library, not a broken environment
            say(hint)
            return EXIT_SYSTEM_LIBS
        say("Delete the .venv folder and start the launcher again. " + where_to_ask())
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
        say(f"\nInstallation failed ({e}). Check the internet connection and " + AGAIN + " " + where_to_ask())
        sys.exit(1)
    except KeyboardInterrupt:
        say("\nInstallation interrupted. To finish it, " + AGAIN)
        sys.exit(1)
