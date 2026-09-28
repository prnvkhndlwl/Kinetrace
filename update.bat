@echo off
rem Windows: double-click to update Kinetrace to the newest published version -
rem the same as Help -> Check for Updates in the app, for when the app will not
rem start. Close Kinetrace first. Your projects, models and settings are kept.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Kinetrace is not installed in this folder yet: double-click run.bat first.
    pause
    exit /b 1
)
echo Close Kinetrace before updating. Press a key to look for a newer version, or close this window.
pause >nul
rem ONE line on purpose: the update may replace this very file, and cmd reads a
rem batch file as it goes (I142). Start Kinetrace afterwards with run.bat - it
rem installs anything the new version needs first.
".venv\Scripts\python.exe" -m kinetrace.update & echo. & echo Start Kinetrace with run.bat. & pause & exit /b
