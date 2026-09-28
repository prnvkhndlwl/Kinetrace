# Kinetrace — animal motion tracking

<table>
<tr>
<td valign="top" width="250">

<a name="contents"></a>
**Contents** ([wiki](https://github.com/prnvkhndlwl/Kinetrace/wiki))

- [Install & run](https://github.com/prnvkhndlwl/Kinetrace/wiki/Install-and-Run)
- [Workflow: track an animal](https://github.com/prnvkhndlwl/Kinetrace/wiki/Workflow)
- [Segments, silhouettes & skeletons](https://github.com/prnvkhndlwl/Kinetrace/wiki/Segments-Silhouettes-and-Skeletons)
- [Several cameras & 3D](https://github.com/prnvkhndlwl/Kinetrace/wiki/Cameras-and-3D)
- [Human bodies](https://github.com/prnvkhndlwl/Kinetrace/wiki/Human-Bodies)
- [Export formats](https://github.com/prnvkhndlwl/Kinetrace/wiki/Export-Formats)
- [Working with other programs](https://github.com/prnvkhndlwl/Kinetrace/wiki/Working-with-Other-Programs) — DeepLabCut, SLEAP, DLTdv, Anipose, OpenCV, MATLAB, Blender; the command-line converter
- [Projects & autosave](https://github.com/prnvkhndlwl/Kinetrace/wiki/Projects-and-Autosave) — the project file format: [docs/FORMAT.md](docs/FORMAT.md)
- [Keyboard & mouse](https://github.com/prnvkhndlwl/Kinetrace/wiki/Keyboard-and-Mouse)
- [Accuracy](https://github.com/prnvkhndlwl/Kinetrace/wiki/Accuracy)
- [Performance (4K)](https://github.com/prnvkhndlwl/Kinetrace/wiki/Performance)
- [Limitations](https://github.com/prnvkhndlwl/Kinetrace/wiki/Limitations)
- [Testing](https://github.com/prnvkhndlwl/Kinetrace/wiki/Testing)
- [Folder map & design](https://github.com/prnvkhndlwl/Kinetrace/wiki/Folder-Map-and-Design)

**Also**

- [User manual](docs/MANUAL.md) — for beginners, also in the app (F1)
- [Licences of everything used](THIRD_PARTY_LICENSES.md)

</td>
<td valign="top">

**Kinetrace** is a desktop app for tracking animal movement in video: click the
body parts you care about, watch them being followed live, pause at any moment
to correct a point with the mouse, and export the tracks. Built for long
(40,000+ frame), high-resolution (4K) recordings.

- **Point tracking** with **[AllTracker](https://github.com/aharley/alltracker)**
  (the default) or Meta's **[CoTracker3](https://github.com/facebookresearch/co-tracker)**,
  plus a native-resolution sub-pixel refinement stage.
- **Silhouettes** with Meta's **[SAM 3 / SAM 2](https://github.com/facebookresearch/sam3)**:
  tail tip, midline points, feet and wing tips come from the outline, so thin
  tails and textureless bodies no longer defeat tracking.
- **Several cameras and 3D**: sync by sound or motion, wand and checkerboard
  calibration, triangulation, visual-hull volume, speeds and accelerations.
- **Human bodies**: joints and joint angles with
  **[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)** or ViTPose.
- **Exports** for DeepLabCut, DLTdv8, MATLAB and plain CSV.
- Written for users with no tracking or calibration background: every control
  explains itself, and every number comes with a plain-language verdict.

**Quick start.** Get the code (*Code → Download ZIP*, or `git clone`), then
double-click `run.bat` on Windows, `Kinetrace.command` on an Apple Silicon Mac
(macOS 14+), or run `./run.sh` on Ubuntu 22.04+. Nothing needs to be installed
first — not even Python: the first run sets up everything inside the folder
without asking anything, and ends with a **system check** saying what your
computer can run — details in
**[Install & run](https://github.com/prnvkhndlwl/Kinetrace/wiki/Install-and-Run)**. New to
tracking? Start with the **[user manual](docs/MANUAL.md)**.

**Any computer:** an NVIDIA GPU is used when present; without one (or on a Mac,
with its Metal GPU) everything still works, only slower, and the one NVIDIA-only
feature (SAM 3D Body) is greyed out with the reason. **Everything stays in the
folder:** nothing is installed into the operating system, deleting the folder
uninstalls Kinetrace, and it sends no usage data anywhere. **Updating:**
Help → Check for Updates… installs a newer version and restarts, keeping your
projects and models (or, with the app closed, double-click `update.bat` /
`Update.command`, or run `bash update.sh`).

**Licences.** Kinetrace is free for any **non-commercial** use — research,
teaching, study — under the [PolyForm Noncommercial License 1.0.0](LICENSE.md)
(developed at biomechLab@CMC). CoTracker3 is **non-commercial**
(CC BY-NC 4.0); AllTracker is MIT; SAM 3 / SAM 3D Body are under Meta's SAM
License (acknowledge them in publications). Every model and package is listed
in **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**.

</td>
</tr>
</table>

<sub>The full documentation is in the [wiki](https://github.com/prnvkhndlwl/Kinetrace/wiki);
its sidebar lists every page.</sub>
