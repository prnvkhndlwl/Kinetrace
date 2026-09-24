# Kinetrace — animal motion tracking

<table>
<tr>
<td valign="top" width="250">

<a name="contents"></a>
**Contents**

- [Install & run](docs/guide/install.md)
- [Workflow: track an animal](docs/guide/workflow.md)
- [Segments, silhouettes & skeletons](docs/guide/segments.md)
- [Several cameras & 3D](docs/guide/cameras-3d.md)
- [Human bodies](docs/guide/bodies.md)
- [Export formats](docs/guide/exports.md)
- [Projects & autosave](docs/guide/projects.md)
- [Keyboard & mouse](docs/guide/shortcuts.md)
- [Accuracy](docs/guide/accuracy.md)
- [Performance (4K)](docs/guide/performance.md)
- [Limitations](docs/guide/limitations.md)
- [Testing](docs/guide/testing.md)
- [Folder map & design](docs/guide/folder-map.md)

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
double-click `run.bat` on Windows or run `./run.sh` on Ubuntu 22.04+ or an
Apple Silicon Mac (macOS 14+). The first run installs everything inside the
folder — details in **[Install & run](docs/guide/install.md)**. New to
tracking? Start with the **[user manual](docs/MANUAL.md)**.

**Everything stays in the folder:** nothing is installed into the operating
system, deleting the folder uninstalls Kinetrace, and it sends no usage data
anywhere.

**Licences.** Kinetrace's own code has no licence chosen yet (all rights
reserved until a `LICENSE` file is added). CoTracker3 is **non-commercial**
(CC BY-NC 4.0); AllTracker is MIT; SAM 3 / SAM 3D Body are under Meta's SAM
License (acknowledge them in publications). Every model and package is listed
in **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**.

</td>
</tr>
</table>

<sub>On any page, GitHub's outline button (☰, top right of the file view) lists
that page's headings.</sub>
