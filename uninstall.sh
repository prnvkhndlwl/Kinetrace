#!/usr/bin/env bash
# =============================================================================
#  Kinetrace uninstaller for macOS and Linux.        bash uninstall.sh
#
#  Removes this Kinetrace folder (the app, its environment, its models, logs,
#  recovery copies and settings) and the few things Kinetrace may have written
#  OUTSIDE it: the per-user fallback folder (used only when this folder could
#  not be written) and the settings of Kinetrace 0.4.1 and earlier. It lists
#  everything first and asks you to type DELETE. Your projects, exports and
#  calibration files are saved where you chose them and are NOT touched; if one
#  is saved INSIDE this folder, it stops and names it so you can move it first.
#  Windows: delete the Kinetrace folder and %LOCALAPPDATA%\Kinetrace (if it is there).
# =============================================================================
set -e
cd "$(dirname "$0")"
HERE="$(pwd -P)"
say() { printf '%s\n' "$*"; }

if [ ! -f "$HERE/kinetrace/__init__.py" ] || [ ! -f "$HERE/run.sh" ]; then
    say "This does not look like a Kinetrace folder ($HERE); nothing was deleted."
    exit 1
fi

# projects saved inside the folder would go with it: name them and stop
inside=$(find "$HERE" -path "$HERE/.venv" -prune -o -path "$HERE/models" -prune -o \
              -name "*.kinetrace" -type d -print 2>/dev/null | head -20)
if [ -n "$inside" ]; then
    say "These projects are saved INSIDE the Kinetrace folder and would be deleted with it:"
    say "$inside" | sed 's/^/    /'
    say "Move them somewhere else first, then run this again. Nothing was deleted."
    exit 1
fi

targets=("$HERE")
if [ "$(uname -s)" = "Darwin" ]; then
    outside=("$HOME/Library/Application Support/Kinetrace"
             "$HOME/Library/Preferences/com.kinetrace.Kinetrace.plist")
else
    outside=("${XDG_DATA_HOME:-$HOME/.local/share}/kinetrace"
             "${XDG_CONFIG_HOME:-$HOME/.config}/Kinetrace")
fi
for p in "${outside[@]}"; do
    if [ -e "$p" ]; then targets+=("$p"); fi
done
say "This removes Kinetrace completely:"
for p in "${targets[@]}"; do
    size=$(du -sh "$p" 2>/dev/null | cut -f1)
    say "    $p   ${size:+($size)}"
done
say ""
say "Your projects, exports and calibration files are NOT touched (they are where you saved them)."
if [ -n "${KINETRACE_UNINSTALL_ANSWER:-}" ]; then       # the test suite's answer (verify_install_paths)
    answer="$KINETRACE_UNINSTALL_ANSWER"
else
    printf 'Type DELETE to remove the folders above, anything else to stop: '
    read -r answer < /dev/tty || answer=""
fi
if [ "$answer" != "DELETE" ]; then
    say "Stopped; nothing was deleted."
    exit 0
fi

[ "$(uname -s)" = "Darwin" ] && defaults delete com.kinetrace.Kinetrace >/dev/null 2>&1 || true
for p in "${targets[@]}"; do
    [ "$p" = "$HERE" ] && continue
    rm -rf "$p"
done
cd /
rm -rf "$HERE"
say "Kinetrace was removed."
