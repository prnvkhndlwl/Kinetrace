# Contributing to Kinetrace

Thank you for helping. Kinetrace is beta software (0.x), so bug reports from
real use are the most valuable contribution of all.

By taking part you agree to follow the [code of conduct](CODE_OF_CONDUCT.md).
Security problems go through a private report instead: see [SECURITY.md](SECURITY.md).

## Reporting a bug

Open an [issue](https://github.com/prnvkhndlwl/Kinetrace/issues/new/choose) and
pick **Bug report**. The form asks for what helps most:

- the version (*Help → About Kinetrace*) and your operating system;
- what you did, what you expected, and what happened instead;
- the error details: **Help → Error Report… → Copy** puts the recent errors and
  the System Check on the clipboard; paste them into the issue. They stay on
  your computer until you paste them, and they can contain file and folder
  names: read them first and remove anything private.

**Do not attach footage you are not allowed to share.** If a problem only shows
on a video, describe it (size, frame rate, camera, codec) and we will ask how it
can be reproduced.

Questions and ideas are welcome too: use **Question** or **Feature request**.

## Changing the code

1. Fork the repository and make a branch for one change (one fix or one
   feature per pull request).
2. Install as for normal use (see [docs/INSTALL.md](docs/INSTALL.md)); the
   tests run with the program's own Python.
3. **A fix comes with a test that fails without it** and, for anything in the
   window, makes the user's real gesture (keys, clicks, drops) rather than
   calling a method directly. Tests live in `tests/`; each prints `PASSED` and
   exits non-zero on a failure.
4. Run the tests for what you touched, then the CPU group:

   ```
   .venv\Scripts\python.exe tests\run_suites.py --cpu        (Windows)
   .venv/bin/python tests/run_suites.py --cpu                (macOS / Ubuntu)
   ```

   With an NVIDIA GPU, also `--gpu`. Say in the pull request what you ran.
5. Update the documentation the change makes wrong: the in-app manual
   (`docs/MANUAL.md`), the [wiki](https://github.com/prnvkhndlwl/Kinetrace/wiki)
   page, tooltips and messages.
6. Open a pull request against `main` and describe what changed and why.

### Ground rules

- **Never trade accuracy for speed.** An optimisation must produce exactly the
  same coordinates as before, or it is not merged.
- **No new dependency without asking first** (open an issue): Kinetrace installs
  into its own folder on Windows, macOS and Ubuntu, and every package has to
  work in all three.
- **No data in git:** no videos, projects, clicked points or personal paths;
  tests make their own clips.
- Messages to the user say in plain words what failed and what to do next.

## Licence of contributions

Kinetrace is released under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md). By opening a pull request
you agree that your contribution is released under the same licence.
