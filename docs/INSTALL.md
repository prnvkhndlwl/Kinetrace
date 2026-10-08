# Installing Kinetrace

**In one sentence:** put the folder somewhere sensible, double-click the
launcher, and wait once. Everything is set up **inside the folder**: no
questions, no Python or other software needed beforehand. At the end it tells
you what your computer can run. This page is in the folder you downloaded
(`docs/INSTALL.md`), so it is there even when you are offline.

- [What you need](#what-you-need)
- [1. Get the folder and put it somewhere sensible](#1-get-the-folder-and-put-it-somewhere-sensible)
- [2. Start it the first time](#2-start-it-the-first-time)
- [3. You are done: starting it again](#3-you-are-done-starting-it-again)
- [Which models run on which computer](#which-models-run-on-which-computer)
- [Optional models that need Meta's permission](#optional-models-that-need-metas-permission-sam-3-sam-3d-body)
- [Where Kinetrace keeps things](#where-kinetrace-keeps-things)
- [Uninstalling](#uninstalling)
- [Updating](#updating)
- [When something goes wrong](#when-something-goes-wrong)
- [How it works underneath](#how-it-works-underneath)
- [Words used on this page](#words-used-on-this-page)
- [Choices for experts](#choices-for-experts)

## What you need

| | Windows 10 / 11 | Ubuntu 22.04 or newer | macOS 14 or newer |
|---|---|---|---|
| Computer | 64-bit PC (Intel / AMD) | 64-bit PC (x86-64 or ARM) | a Mac with **Apple Silicon** (M1 or later). Intel Macs cannot run it: PyTorch no longer supports them |
| Software | nothing: the launcher fetches a private Python if the PC has none | nothing (a few system libraries for the window are installed with `apt` if missing; that step asks for your password) | nothing |
| Graphics | an NVIDIA GPU (driver 580 or newer) is used automatically; without one, the CPU | the same | the Mac's own GPU (Metal) |
| Memory | 16 GB recommended; 8 GB works, more slowly with 4K video | the same | the same |
| Disk for the program | about 4 GB with an NVIDIA GPU (most of it PyTorch's CUDA build), much less without | the same | about 1.5 GB |
| Disk for the models | about 0.7 GB for the ones most people use (AllTracker 66 MB, SAM 2.1 base+ 617 MB), more if you add others (table below) | the same | the same |
| Internet | for the first run and the first use of each model; afterwards only for updates, when you ask | the same | the same |

## 1. Get the folder and put it somewhere sensible

Download it from the project page with **Code → Download ZIP** and unzip it
(or `git clone` the repository). The unzipped folder is called
**`Kinetrace-main`**. Rename it to `Kinetrace` if you like: the name and the
place do not matter, and the folder can be moved or renamed later (the
launcher repairs itself).

**Where to put it.** Somewhere in your own user folder that is **not synced to
the cloud**, for example:

- **Mac:** your home folder, i.e. `Kinetrace` next to Documents (in Finder:
  **Go → Home**). Not the Desktop or Documents when *iCloud Drive → Desktop &
  Documents Folders* is on: iCloud would upload the 1.5 GB environment and the
  models, and *Optimize Mac Storage* can remove files from the disk, which
  breaks the program. Not Downloads either (easy to clean out by mistake).
  An external drive formatted for Windows (exFAT) may not work.
- **Windows:** e.g. `C:\Kinetrace` or a folder in your user folder. Avoid
  folders that OneDrive or Dropbox sync.
- **Ubuntu:** anywhere in your home folder, e.g. `~/Kinetrace`.

## 2. Start it the first time

### Windows

Double-click **`run.bat`**. If Windows shows *"Windows protected your PC"*,
click *More info → Run anyway*. It is a plain script; you can open it in
Notepad to read it.

### Mac

1. Double-click **`Kinetrace.command`**.
2. The Mac refuses the first time, because the file came from the internet
   and is not signed by Apple: *"Kinetrace.command" cannot be opened* (or
   *"Apple could not verify…"*). Click **Done** (or **OK**), **not** *Move to
   Trash*.
   - **macOS 15 and newer:** open **System Settings → Privacy & Security**,
     scroll down to the line about `Kinetrace.command`, click **Open Anyway**
     and confirm with your password or Touch ID. Then double-click
     `Kinetrace.command` again and click **Open**.
   - **macOS 14:** right-click (or Control-click) `Kinetrace.command`, choose
     **Open**, and confirm.
   - Or, in Terminal, once (adjust the path to where you put the folder):

     ```bash
     xattr -dr com.apple.quarantine ~/Kinetrace
     ```
3. A **Terminal** window opens and the setup runs in it. **Keep it open** until
   Kinetrace's window appears (a few minutes the first time). If the Mac asks
   *"Terminal would like to access files in your Documents (or Desktop,
   Downloads) folder"*, click **Allow**: the folder you put Kinetrace in is
   one of those.
4. At the end the setup makes **`Kinetrace.app`** in the folder (see step 3
   below).

### Ubuntu

Open a terminal in the folder and type `./run.sh` (or double-click `run.sh`
and choose *Run in Terminal*). If the window's system libraries are missing,
the launcher installs them with `apt`. That is the one step that asks for your
password; when it cannot ask, it prints the exact `sudo apt-get install …` line.

### What the first run does

1. Looks for a usable Python 3.10 – 3.14. On a Mac it always fetches its own
   private copy instead, so that updating Homebrew can never break Kinetrace.
   If there is none, it downloads a private copy (15 – 30 MB, checksum-checked)
   into `.venv/base` inside the folder. Nothing is installed into the
   operating system.
2. Creates the private environment `.venv` and installs the packages into it
   (most of the size above is PyTorch). The right PyTorch build is chosen for
   you: CUDA with an NVIDIA GPU, CPU without one, Metal on a Mac.
3. Fetches the code of AllTracker, the default point model, at a fixed
   version (its 66 MB weights download on the first Track).
4. Checks that everything works and prints the **system check**: what was
   found (graphics card, memory, PyTorch build) and, feature by feature, what
   runs on this computer, what runs slower, and what is switched off.

The steps are numbered (`==== Step 2 of 4 …`); the many lines in between are
the package installer's own messages. Everything is also written to
`logs/install-<date>.log` in the folder. If the download is interrupted, start
the launcher again: what is already installed is kept and it goes on from there.

## 3. You are done: starting it again

- **Windows:** double-click `run.bat`. It starts in seconds from now on.
- **Mac:** double-click **`Kinetrace.app`** in the folder. It starts in seconds
  with no Terminal window (its messages go to `logs/launcher.log`). Drag it to
  the Dock to keep it there. It is made on your Mac during the first run, so
  macOS does not ask about it. It shows as **Kinetrace** in the Dock, the menu
  bar and ⌘-Tab. (`Kinetrace.command` still works; it keeps a Terminal
  window open while Kinetrace runs, and closing that window quits Kinetrace.)
  If a later update needs to install something, `Kinetrace.app` opens the
  Terminal window for that step by itself.
- **Ubuntu:** `./run.sh`.

The first **Track** downloads the point model (66 MB) and the first outline the
silhouette model (617 MB), each with a progress window. After that Kinetrace
works offline.

**Keys on a Mac:** where this documentation says **Ctrl**, press **⌘ (Cmd)**;
**Alt** is **⌥ (Option)**. Kinetrace's own key reference and its manual (F1)
already show them that way on a Mac. Settings is **⌘,**; macOS moves it into
the application menu (the one left of **File**), as on every Mac app.

## Which models run on which computer

| Model | What for | Size on disk | NVIDIA GPU (Windows / Ubuntu) | Mac (Apple GPU) | No GPU |
|---|---|---|---|---|---|
| AllTracker | points (the default) | 66 MB + 1 MB code | ✅ | failed in 0.4.1; fixed after it (the one step Apple's GPU cannot do now runs on the CPU), not yet confirmed on a Mac. If Track fails, choose CoTracker3 in **Track ▾** | ✅ slower; works on a smaller picture with little memory |
| CoTracker3 | points (faster on sharp markers) | 102 MB + 36 MB code | ✅ | ✅ (0.5 – 1.1 px mean error, measured on an M4 Max) | ✅ slower |
| SAM 2.1 base+ | silhouettes (the default) | 617 MB | ✅ | ✅ (2.3 frames/s at 1280×720 on an M4 Max) | ✅ the practical choice |
| SAM 2.1 large | silhouettes | 1.7 GB | ✅ | not tested yet (the same code as base+) | slow |
| SAM 3 (needs Meta's permission) | the best silhouettes | 3.4 GB | ✅ | not tested yet; 32 GB of memory suggested | very slow |
| ViTPose + RT-DETR | human joints in 2D | 425 + 81 MB | ✅ | ✅ | ✅ slower |
| SAM 3D Body (needs Meta's permission) | human joints in 3D | 2.8 GB (2.1 + 0.7) | ✅ | ❌ **NVIDIA only**: Meta's code cannot run on a Mac | ❌ |

## Optional models that need Meta's permission (SAM 3, SAM 3D Body)

Neither is needed to track animals: without them Kinetrace uses SAM 2.1 for
outlines and ViTPose for human joints. Meta releases them only on request, so
they cannot be fetched automatically. The install makes the folders they go in,
each with a `PUT_FILES_HERE.txt` naming the exact files.

### SAM 3 (the best silhouettes, 3.4 GB)

The easy way, without Terminal:

1. Make a free account at [huggingface.co](https://huggingface.co), open
   [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), click
   **Agree and access repository**, and wait until that page says you have
   access.
2. Make a **Read token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   → *Create new token* → type **Read** → copy it.
3. In Kinetrace: **Ctrl+,** (**⌘,** on a Mac) opens Settings → *Hugging Face
   token*: paste it, **Save token** → *Segmentation model*: **SAM 3**.
4. The first outline downloads SAM 3 (3.4 GB) with a progress window. From then
   on it works offline, and it becomes the default.

Or, from a Terminal opened in the Kinetrace folder (on Windows write
`.venv\Scripts\python.exe` instead of `.venv/bin/python`):

```bash
.venv/bin/python -m huggingface_hub.cli.hf download facebook/sam3 --revision 3c879f39826c281e95690f02c7821c4de09afae7 --exclude "sam3.pt" --local-dir models/sam3 --token YOUR_READ_TOKEN
```

The `--exclude "sam3.pt"` leaves out a second 3.45 GB copy of the weights that
Kinetrace does not use (a plain download is 6.9 GB). The `--revision` is the
version Kinetrace was tested with.

### SAM 3D Body (3D human joints; NVIDIA GPU only)

**Mac users: skip this.** It cannot run on a Mac or on a computer without an
NVIDIA graphics card; the Body dialog offers ViTPose (2D joints) there.

1. Request access at [huggingface.co/facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3).
2. Download its two files into `models/sam-3d-body-dinov3` (from a Terminal in
   the Kinetrace folder):

   ```bash
   .venv/bin/python -m huggingface_hub.cli.hf download facebook/sam-3d-body-dinov3 model.ckpt assets/mhr_model.pt --local-dir models/sam-3d-body-dinov3 --token YOUR_READ_TOKEN
   ```

   `model.ckpt` is 2.1 GB; `assets/mhr_model.pt` (0.7 GB) sits in a
   sub-folder and is easy to miss.
3. Clone Meta's code into a new folder `models/sam-3d-body`:

   ```bash
   git clone https://github.com/facebookresearch/sam-3d-body models/sam-3d-body
   ```

Its Python packages are already installed. **Body → 3D body** says what is
still missing.

## Where Kinetrace keeps things

**Inside the Kinetrace folder** (deleting the folder removes all of these):
the environment (`.venv`), the models (`models/`), the error and install logs
(`logs/`), unsaved-work recovery copies (`recovery/`), saved skeletons
(`skeletons/`), your settings (`settings.ini`), and on a Mac `Kinetrace.app`.

**Outside it:**

- **Your projects, exports and calibration files**: wherever you save them.
  They are never deleted by an update or by uninstalling.
- Only if the Kinetrace folder cannot be written (installed read-only): the
  logs and recovery copies go to `%LOCALAPPDATA%\Kinetrace` (Windows),
  `~/Library/Application Support/Kinetrace` (Mac) or `~/.local/share/kinetrace`
  (Ubuntu).
- Kinetrace 0.4.1 and earlier kept your name for notes in the registry
  (`HKEY_CURRENT_USER\Software\Kinetrace`),
  `~/Library/Preferences/com.kinetrace.Kinetrace.plist` (Mac) or
  `~/.config/Kinetrace` (Ubuntu). Newer versions use `settings.ini` in the
  folder instead.
- PyTorch may make an empty `torchinductor_<you>` folder in the system's
  temporary folder; the system clears it.
- If you set a Hugging Face token up with Hugging Face's own `hf auth login`
  command (instead of Kinetrace's Settings), Hugging Face stores it in
  `~/.cache/huggingface`.
- macOS remembers that you allowed `Kinetrace.command` and Terminal's access to
  the folder (System Settings → Privacy & Security).

To see every one of these on your computer, with sizes: **Help → Kinetrace's
Folders…** (each one opens with **Show in Finder** / the file manager), or
`./run.sh --paths` / `run.bat --paths`.

## Uninstalling

**Mac / Ubuntu:** in a Terminal, `bash uninstall.sh` in the Kinetrace folder.
It lists what it will delete (the folder, and the outside folders above if
they exist) and asks you to type `DELETE`. It stops without deleting anything
if a project is saved inside the Kinetrace folder, and names it so you can move
it first. To do it by hand on a Mac:

```bash
rm -rf ~/Kinetrace "$HOME/Library/Application Support/Kinetrace" ~/Library/Preferences/com.kinetrace.Kinetrace.plist
```

(with your folder's path in place of `~/Kinetrace`). Also remove Kinetrace from
the Dock if you put it there.

**Windows:** close Kinetrace, then delete the Kinetrace folder, and
`%LOCALAPPDATA%\Kinetrace` if it exists. To remove the old settings of 0.4.1
and earlier, in a Command Prompt:

```bat
reg delete HKCU\Software\Kinetrace /f
```

Your projects and exports are not touched by any of this.

## Updating

**Help → Check for Updates…** asks GitHub for the newest version (one request;
nothing about you is sent, and nothing is checked unless you ask), shows what is
new, and **Update now** installs it. Only changed files are replaced; `.venv/`,
`models/`, `recovery/`, `logs/`, `skeletons/`, `settings.ini` and your footage
are never touched. When the app will not start, double-click `update.bat`
(Windows) or `Update.command` (Mac), or run `bash update.sh` (`--check` only
looks).

## When something goes wrong

- **Help → System Check…** says what this computer can run and why;
  `run.bat --check` / `./run.sh --check` prints the same without a window.
- **Help → Error Report…** shows what went wrong, with **Copy** and **Open the
  log folder**. Nothing is ever sent anywhere by Kinetrace.
- **To ask for help:** open an issue at
  [github.com/prnvkhndlwl/Kinetrace/issues](https://github.com/prnvkhndlwl/Kinetrace/issues).
  Attach the Error Report, or for an install problem the newest
  `logs/install-*.log`. Read it first: it contains folder names from your
  computer.

## How it works underneath

**Without a GPU** everything works; the models are several times slower (a 4K
video takes hours rather than minutes) and **SAM 3D Body is switched off**,
because Meta's code runs only on an NVIDIA card — the Body dialog greys it out
and says so; the 2D model (ViTPose) is offered instead. For silhouettes, SAM 2.1
base+ is the practical choice on a CPU. AllTracker works on a smaller picture on
computers with little memory (the system check says which size). **The GPU is used whenever it is faster.** The first start times a small
workload shaped like the models' inner loops on the GPU and on the CPU (a few
seconds, remembered in `models/device_benchmark.json`), and the faster one does
the work — on a normal machine the GPU wins by a wide margin, and a GPU that
cannot run the workload (a broken driver) is simply not used (and tried again at
the next start — there is no file to delete). The status bar
shows a coloured badge: green **● GPU** with the card's name, or amber **● CPU**
(with "(… available)" when a GPU exists but lost the comparison). Hover over
it for the measured times and what runs slower or is off here; while tracking,
the speed readout also starts with GPU or CPU. **Help → System Check…** has the
full report (with a **Copy** button for support requests). The same report
prints from a terminal with `run.bat --check` / `./run.sh --check`, which also
installs first if needed and never opens a window.

**When something goes wrong**, Kinetrace writes it down: every error (with
where in the program it happened and what you were doing), Qt's own warnings,
and — when the program dies of a native crash — the stack of every thread, go
to `kinetrace.log` and `kinetrace-crash.log` in the `logs/` folder inside the
Kinetrace folder (the per-user data folder when the Kinetrace folder is
read-only). A red notice says so on screen (at most once every ten seconds).
**Help → Error Report…** shows the recent errors plus the System Check, with
**Copy** for a bug report and **Open the log folder** to attach the files (two
Kinetrace windows open at once share the log safely); a
session marked "did NOT end normally" is one that closed by itself. Nothing is
ever sent anywhere.

**Models** are downloaded the first time a feature needs them, into `models/`
inside the folder: the AllTracker point model (66 MB on the first Track),
CoTracker3 (~100 MB + its code, only if you switch to it), SAM 2.1 for
silhouettes (~620 MB), ViTPose + RT-DETR for human poses (~500 MB). A window
shows what is downloading, how far it has got and about how long is left, with
**Cancel** (the next try carries on); a connection cut half way resumes where it
stopped, a stalled connection gives up after 30 s with a sentence instead of
hanging, and a full disk is reported as a full disk. **Every download is pinned and checked**:
each file is the one of a fixed upstream version (a commit), its checksum is
compared before use, and a file that differs is deleted, never run; model
weights are read as plain numbers (`weights_only`), never as code. Once a model
is in `models/`, loading it asks nothing of the internet. An internet
connection is needed only for these first downloads.

A new version is published by changing `APP_VERSION` in
`kinetrace/__init__.py` and pushing it to main: the *Release* workflow
(`.github/workflows/release.yml`) tags that commit `vX.Y.Z` and publishes a
GitHub Release with notes written from the commits. Pushes that leave the
version alone are not offered to anyone.

## Words used on this page

- **Terminal** (Mac, Ubuntu) / **Command Prompt** (Windows): the window where
  you type commands. On a Mac: Finder → Applications → Utilities → Terminal.
  "A Terminal in the Kinetrace folder" means: open Terminal, type `cd `
  (with a space), drag the Kinetrace folder onto the window, press Enter.
- **Python:** the programming language Kinetrace is written in. Kinetrace
  brings its own copy if needed.
- **Environment, `.venv`:** the folder inside Kinetrace that holds its private
  Python and the packages it needs (PyTorch, Qt, OpenCV…). Kept apart from
  anything else on the computer. Deleting `.venv` and starting the launcher
  again rebuilds it from scratch.
- **PyTorch:** the engine that runs the tracking and outline models.
- **Model / weights:** the trained network a feature uses, a file of numbers
  downloaded once into `models/`.
- **Hugging Face:** the website most models are downloaded from.
- **Gated:** a model whose authors (here Meta) must approve each person before
  they can download it.
- **Token / Read token:** a password-like code from your Hugging Face account
  that lets Kinetrace download a gated model for you. "Read" means it can only
  download, nothing else. Keep it to yourself.
- **Gatekeeper:** the part of macOS that blocks programs downloaded from the
  internet until you allow them once.

## Choices for experts

Environment variables, set before the first run or before starting:

| Variable | Effect |
|---|---|
| `KINETRACE_TORCH=cuda` / `cpu` / `mps` | which PyTorch build `install.py` installs (e.g. the CPU build on a PC whose NVIDIA driver is too old for CUDA 13) |
| `KINETRACE_DEVICE=cuda` / `mps` / `cpu` | which device the models run on (e.g. the CPU on a GPU machine, to compare) |
| `KINETRACE_BOOTSTRAP_PYTHON=1` | the launcher ignores every Python on the computer and fetches its private copy (macOS always does) |
| `KINETRACE_ALLTRACKER_MAX_DIM=N` | AllTracker's working picture size (1024 on CUDA; depends on the memory elsewhere) |
| `KINETRACE_DECODE`, `KINETRACE_CACHE_GB`, `KINETRACE_FFMPEG`, `KINETRACE_RECOVERY_DIR` | the video decoder, the frame-cache budget, an ffmpeg to use, where unsaved-work copies go |
| `KINETRACE_UPDATE_API` | where Help → Check for Updates asks (the tests point it at a local server) |

To install by hand: make a virtual environment with Python 3.10 – 3.14, then
run `python install.py` inside it (`--force` reinstalls the packages).
