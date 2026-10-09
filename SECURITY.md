# Security

## Reporting a vulnerability

Please report security problems **privately**, not in a public issue:
open the repository's **Security** tab → **Report a vulnerability**
([direct link](https://github.com/prnvkhndlwl/Kinetrace/security/advisories/new)).
Only the maintainers see the report.

Say what you found, how to reproduce it (the Kinetrace version from
*Help → About Kinetrace* and your operating system), and what an attacker could
do with it. We will acknowledge the report as soon as we can, keep you informed
while it is being fixed, and credit you in the release notes unless you would
rather not be named.

## Supported versions

Kinetrace is in beta (0.x). Only the **latest release** gets security fixes;
*Help → Check for Updates…* installs it and keeps your projects and models.

## What counts

Anything in this repository, for example:

- the installer and launchers (`install.py`, `run.bat`, `run.sh`,
  `Kinetrace.command`) and the updater (`update.*`, *Check for Updates*);
- model downloads that could fetch something other than the pinned, checksummed
  file (`kinetrace/downloads.py`);
- a project, calibration, lens or track file that makes Kinetrace run code,
  write outside the places it says it uses, or crash in a way an attacker could
  steer.

Problems in the third-party models or packages Kinetrace uses (SAM, AllTracker,
CoTracker3, PyTorch, Qt, ...) belong to their own projects: please report them
there as well.

## What Kinetrace does on your computer

Kinetrace runs locally and sends no usage data. It goes online only to download
a model the first time it is needed and when you ask it to check for updates.
