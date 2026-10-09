#!/bin/bash
# macOS: double-click this file in Finder to start Kinetrace (it opens a
# Terminal window and runs run.sh next to it). The first time, macOS refuses it
# (downloaded, not signed by Apple): click Done, then System Settings -> Privacy
# & Security -> Open Anyway (macOS 15+), or right-click it -> Open (macOS 14).
# The first run makes Kinetrace.app in this folder: start with that from then on.
# Full steps: docs/INSTALL.md.
cd "$(dirname "$0")"
chmod +x run.sh 2>/dev/null
# KINETRACE_VIA=command: run.sh sets up what is needed here, then hands over to Kinetrace.app, so
# this Terminal window can be closed without closing Kinetrace
KINETRACE_VIA=command exec ./run.sh "$@"
