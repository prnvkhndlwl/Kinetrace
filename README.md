# Kinetrace

**Track how animals move — in any video, on any computer, with no coding.**
Click the body parts you care about, watch them followed live, fix any mistake
with the mouse, and export the tracks to the tools you already use.

![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue)
![Platforms](https://img.shields.io/badge/runs%20on-Windows%20%7C%20macOS%20%7C%20Ubuntu-informational)
![GPU optional](https://img.shields.io/badge/GPU-optional-success)
[![Latest release](https://img.shields.io/github/v/release/prnvkhndlwl/Kinetrace)](https://github.com/prnvkhndlwl/Kinetrace/releases)

## What it does

- **Follows what you point at.** Click a snout, a toe, a marker on a wand; it is
  tracked frame by frame while you watch, and you can stop at any moment
  (**X**) to click a point back where it belongs.
- **Copes with hard footage.** Long recordings (40,000+ frames), 4K, 300+ fps,
  small and fast animals, thin tails and textureless bodies — the animal's
  outline supplies tail tip, midline and feet where points alone would slip;
  a target small enough to be one point — a dot over waves or sky — has a
  point model of its own, and a test on your clicks says which one to use.
- **Goes from cameras to 3D.** Line several cameras up by sound or motion,
  calibrate them with a wand and a checkerboard, and get 3D positions, speeds,
  accelerations and body volume.
- **Measures people too.** Joints and joint angles from ordinary video.
- **Explains itself.** Written for people who have never tracked anything:
  every number comes with a plain-language verdict and a way to check it by hand.

## Get started

1. **Download** — *Code → Download ZIP* (or `git clone`), and unzip it anywhere.
2. **Start** it:

   | Windows | Mac (Apple Silicon, macOS 14+) | Ubuntu 22.04+ |
   |---|---|---|
   | double-click `run.bat` | double-click `Kinetrace.command` | `./run.sh` |

3. **Wait once.** The first start sets everything up inside the folder (about
   4 GB, mostly PyTorch) without asking anything, and ends with a system check
   of what your computer can run. After that it starts in seconds; the first
   Track and the first outline each download one model (66 MB / 617 MB, with a
   progress window), and from then on it works offline.

**Optional:** the best outlines (**SAM 3**) and 3D human joints (**SAM 3D
Body**) need Meta's permission on Hugging Face first — the
[install page](https://github.com/prnvkhndlwl/Kinetrace/wiki/Install-and-Run#optional-models-that-need-metas-permission-sam-3-sam-3d-body)
has the steps. Everything else downloads by itself.

New to tracking? The **[user manual](docs/MANUAL.md)** (also **F1** in the app)
walks you through your first session.

## Works with

**DeepLabCut** · **SLEAP** · **DLTdv8** · **Anipose** · **easyWand / Argus** ·
**MATLAB** · **Blender** · plain **CSV** — tracks, calibrations and 3D points in
and out ([details](https://github.com/prnvkhndlwl/Kinetrace/wiki/Working-with-Other-Programs)).

## How accurate?

| | |
|---|---|
| Points on 4K video, against known ground truth | **0.74 px** mean error (the sub-pixel refinement takes it from 3.7 px) |
| Wand calibration of a real six-camera rig, against easyWand | 3D points agree to **0.18 cm** median |
| Triangulation, against DLTdv on the same clicks | **0.07 mm** |

More in [Accuracy](https://github.com/prnvkhndlwl/Kinetrace/wiki/Accuracy) and
[Performance](https://github.com/prnvkhndlwl/Kinetrace/wiki/Performance).

## Is it for me?

- **Yes** if you film animals (any species — lizards, gliders, birds, fish,
  mammals) or people and want positions, angles or 3D out of the video.
- **No GPU needed:** an NVIDIA card or a Mac's own GPU makes it faster;
  without one everything still works, only slower.
- **Several animals in one video**, each with its own points, skeleton and
  (optional) silhouette; identities come from your clicks (no automatic
  re-identification yet).
- **Non-commercial use only** (research, teaching, study) — see below.

## Learn more

[Install & run](https://github.com/prnvkhndlwl/Kinetrace/wiki/Install-and-Run) ·
[Your first tracking](https://github.com/prnvkhndlwl/Kinetrace/wiki/Workflow) ·
[Small, fast targets](https://github.com/prnvkhndlwl/Kinetrace/wiki/Small-Fast-Targets) ·
[Silhouettes & skeletons](https://github.com/prnvkhndlwl/Kinetrace/wiki/Segments-Silhouettes-and-Skeletons) ·
[Cameras & 3D](https://github.com/prnvkhndlwl/Kinetrace/wiki/Cameras-and-3D) ·
[Human bodies](https://github.com/prnvkhndlwl/Kinetrace/wiki/Human-Bodies) ·
[Export formats](https://github.com/prnvkhndlwl/Kinetrace/wiki/Export-Formats) ·
[Projects & autosave](https://github.com/prnvkhndlwl/Kinetrace/wiki/Projects-and-Autosave) ·
[Keyboard & mouse](https://github.com/prnvkhndlwl/Kinetrace/wiki/Keyboard-and-Mouse) ·
[Limitations](https://github.com/prnvkhndlwl/Kinetrace/wiki/Limitations) ·
[The whole wiki](https://github.com/prnvkhndlwl/Kinetrace/wiki)

**Updating:** *Help → Check for Updates…* installs a newer version and keeps
your projects and models.

## Licence & privacy

Free for any **non-commercial** use under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md) — developed at biomechLab@CMC.
Built on [AllTracker](https://github.com/aharley/alltracker),
[CoTracker3](https://github.com/facebookresearch/co-tracker) (non-commercial),
[SAM](https://github.com/facebookresearch/sam3), ViTPose and
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body); each keeps its own licence
([all of them](THIRD_PARTY_LICENSES.md)).

**Nothing leaves your computer.** Kinetrace sends no usage data; it goes online
only to download a model the first time it is needed and when you ask it to
check for updates. Everything lives in its folder — delete the folder and it is
gone.
