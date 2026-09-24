#!/bin/bash
# macOS: double-click this file in Finder to start Kinetrace (it opens a
# Terminal window and runs run.sh next to it). The first time, macOS may say the
# file "cannot be opened because it is from an unidentified developer": then
# right-click (or Control-click) it, choose Open, and confirm once.
cd "$(dirname "$0")"
chmod +x run.sh 2>/dev/null
exec ./run.sh "$@"
