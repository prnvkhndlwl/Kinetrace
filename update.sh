#!/bin/bash
# Linux / macOS: update Kinetrace to the newest published version - the same as
# Help -> Check for Updates in the app, for when the app will not start.
# Close Kinetrace first; your projects, models and settings are kept.
#     bash update.sh            look, and install a newer version
#     bash update.sh --check    only look
# Afterwards start Kinetrace as usual (run.sh / Kinetrace.command): it installs
# anything the new version needs first.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "Kinetrace is not installed in this folder yet: run run.sh (or double-click Kinetrace.command) first."
    exit 1
fi
# exec on purpose: the update may replace this very file, and bash reads a
# script as it goes (I142)
exec .venv/bin/python -m kinetrace.update "$@"
