#!/bin/bash
# macOS: double-click this file in Finder to update Kinetrace to the newest
# published version (it opens a Terminal window and runs update.sh next to it).
# Close Kinetrace first. Afterwards double-click Kinetrace.command as usual.
# The first time, macOS may refuse to open it: right-click (or Control-click) it,
# choose Open, and confirm once.
cd "$(dirname "$0")" || exit 1
exec bash ./update.sh "$@"
