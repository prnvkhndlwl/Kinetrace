#!/usr/bin/env bash
# =============================================================================
#  Kinetrace uninstaller for macOS and Linux.        bash uninstall.sh
#
#  Removes this Kinetrace folder (the app, its environment, its models, logs,
#  recovery copies and settings) after you type DELETE. Then, only if you say
#  yes, the few things Kinetrace may have written OUTSIDE it, which every
#  Kinetrace copy on the computer shares: the per-user fallback folder (used
#  only when a Kinetrace folder could not be written) and the settings of
#  Kinetrace 0.4.1 and earlier. Your projects, exports and
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

shared=()
if [ "$(uname -s)" = "Darwin" ]; then
    outside=("$HOME/Library/Application Support/Kinetrace"
             "$HOME/Library/Preferences/com.kinetrace.Kinetrace.plist")
else
    outside=("${XDG_DATA_HOME:-$HOME/.local/share}/kinetrace"
             "${XDG_CONFIG_HOME:-$HOME/.config}/Kinetrace")
fi
for p in "${outside[@]}"; do
    if [ -e "$p" ]; then shared+=("$p"); fi
done
size=$(du -sh "$HERE" 2>/dev/null | cut -f1)
say "This removes Kinetrace completely:"
say "    $HERE   ${size:+($size)}"
say ""
say "Your projects, exports and calibration files are NOT touched (they are where you saved them)."
if [ -n "${KINETRACE_UNINSTALL_ANSWER:-}" ]; then       # the test suite's answer (verify_install_paths)
    answer="$KINETRACE_UNINSTALL_ANSWER"
else
    printf 'Type DELETE to remove the folder above, anything else to stop: '
    read -r answer < /dev/tty || answer=""
fi
if [ "$answer" != "DELETE" ]; then
    say "Stopped; nothing was deleted."
    exit 0
fi

# Outside the folder: SHARED by every Kinetrace copy on this computer (unsaved-work copies of a copy
# whose folder could not be written; the settings of 0.4.1 and earlier). Asked apart, default keep:
# another copy may still need them (Mac report 2026-10-08)
if [ ${#shared[@]} -gt 0 ]; then
    say ""
    say "Kinetrace also left these OUTSIDE its folder. They are shared by every Kinetrace copy on"
    say "this computer (unsaved work of a copy that could not write its own folder, old settings):"
    for p in "${shared[@]}"; do
        s=$(du -sh "$p" 2>/dev/null | cut -f1)
        say "    $p   ${s:+($s)}"
    done
    if [ -n "${KINETRACE_UNINSTALL_SHARED:-}" ]; then   # the test suite's answer
        also="$KINETRACE_UNINSTALL_SHARED"
    else
        printf 'Remove these too? Only if no other Kinetrace copy is left. Type yes, or Enter to keep them: '
        read -r also < /dev/tty || also=""
    fi
    if [ "$also" = "yes" ]; then
        [ "$(uname -s)" = "Darwin" ] && { defaults delete com.kinetrace.Kinetrace >/dev/null 2>&1 || true; }
        for p in "${shared[@]}"; do
            rm -rf "$p"
        done
        say "Removed them."
    else
        say "Kept them."
    fi
fi
cd /
rm -rf "$HERE"
say "Kinetrace was removed."
