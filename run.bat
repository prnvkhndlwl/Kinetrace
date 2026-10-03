@echo off
setlocal
cd /d "%~dp0"
rem ============================================================================
rem  Kinetrace launcher for Windows 10 / 11. Double-click it.
rem
rem  First run: everything is set up INSIDE this folder without any questions -
rem  a private Python if the computer has none (.venv\base), then the packages
rem  (.venv). Later runs start the app at once. Deleting the folder uninstalls
rem  everything. "run.bat --check" prints what this computer can run and exits.
rem ============================================================================

if exist ".venv\kinetrace-install.json" goto :launch

if exist ".venv\Scripts\python.exe" (
    echo Kinetrace - checking the environment in .venv ...
    goto :install
)
echo ============================================================
echo  Kinetrace - first run: setting up a self-contained environment in .venv
echo  This downloads 2-4 GB (PyTorch, more with an NVIDIA graphics card)
echo  and can take a while. Everything installs INSIDE this folder -
echo  deleting the folder removes the tool completely. No questions asked.
echo ============================================================

rem --- 1. a Python 3.10 - 3.14 that can make virtual environments -------------
rem "py" is the python.org launcher; "python" may be the Microsoft Store stub,
rem which only prints an advertisement and fails - the version test filters it.
set "PYEXE="
rem the imports come BEFORE the version exit (I220): after sys.exit they never ran, so a Python without venv
rem / ensurepip was accepted and "python -m venv" failed later
set "PYCHECK=import sys, venv, ensurepip; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)"
rem KINETRACE_BOOTSTRAP_PYTHON=1 skips the search and always uses a private
rem Python (a broken system Python; the CI's bootstrap job)
if "%KINETRACE_BOOTSTRAP_PYTHON%"=="1" goto :bootstrap
if exist ".venv\base\python.exe" (
    ".venv\base\python.exe" -c "%PYCHECK%" >nul 2>nul && set "PYEXE=.venv\base\python.exe"
)
if not defined PYEXE (
    py -3.12 -c "%PYCHECK%" >nul 2>nul && set "PYEXE=py -3.12"
)
if not defined PYEXE (
    py -3 -c "%PYCHECK%" >nul 2>nul && set "PYEXE=py -3"
)
if not defined PYEXE (
    python -c "%PYCHECK%" >nul 2>nul && set "PYEXE=python"
)
if defined PYEXE goto :makevenv

:bootstrap
rem --- 2. none found: fetch a private, registry-free CPython into .venv\base ---
rem (the official NuGet build of CPython 3.12.10, 15 MB; nothing is installed
rem into Windows). PowerShell 5 is on every Windows 10 / 11.
echo No Python found on this computer - downloading a private copy (15 MB) into .venv\base ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
  "$zip = Join-Path $env:TEMP 'kinetrace-python.zip'; $tmp = Join-Path $env:TEMP 'kinetrace-python';" ^
  "if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp };" ^
  "Invoke-WebRequest -UseBasicParsing -Uri 'https://www.nuget.org/api/v2/package/python/3.12.10' -OutFile $zip;" ^
  "$want = 'bbda4dcf688a94211b62d50968a91b38f305d0b8d1ecd90269f74a86f8a0a4fcebb7ca162a0753a47691eb3df0c964009bd3d8194c6fd19afae8d5fd01e1cc0f';" ^
  "if ((Get-FileHash $zip -Algorithm SHA512).Hash -ne $want) { Remove-Item -Force $zip; throw 'The downloaded Python is not the expected file (its checksum differs); it was deleted, not used.' };" ^
  "Expand-Archive -Path $zip -DestinationPath $tmp -Force;" ^
  "New-Item -ItemType Directory -Force '.venv' | Out-Null;" ^
  "if (Test-Path '.venv\base') { Remove-Item -Recurse -Force '.venv\base' };" ^
  "Move-Item (Join-Path $tmp 'tools') '.venv\base';" ^
  "Remove-Item -Recurse -Force $tmp; Remove-Item -Force $zip"
if not exist ".venv\base\python.exe" (
    echo.
    echo ERROR: could not download Python ^(the line above says why: no connection,
    echo or a download that was not the expected file^). Check the internet connection and
    echo double-click run.bat again. ^(Or install Python 3.12 from python.org
    echo - tick "Add python.exe to PATH" - and run it again.^)
    pause
    exit /b 1
)
set "PYEXE=.venv\base\python.exe"

:makevenv
echo Creating the environment with %PYEXE% ...
%PYEXE% -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: could not create the environment. Delete the .venv folder and
    echo double-click run.bat again.
    pause
    exit /b 1
)

:install
rem install.py picks the PyTorch build (CUDA with an NVIDIA GPU, else CPU), then
rem requirements.txt, fetches AllTracker, verifies every import and writes
rem .venv\kinetrace-install.json. Safe to re-run: it resumes where it stopped.
".venv\Scripts\python.exe" install.py
if errorlevel 1 (
    echo.
    echo ERROR: the installation did not finish. Check your internet connection
    echo and double-click run.bat again ^(it resumes^). If it fails twice, send
    echo the lines above with your question.
    pause
    exit /b 1
)
if "%~1"=="--check" goto :done

:launch
rem AllTracker (the default point model) is fetched at install; retry if that was missed
if not exist "models\alltracker\nets\alltracker.py" ".venv\Scripts\python.exe" install.py --alltracker-only
rem The interpreter may live inside the folder (.venv\base). If the folder was
rem moved or renamed, .venv\pyvenv.cfg still points at the old absolute path
rem and the venv launcher exits with code 103. Detect that by trying to start
rem it (a string compare would hit findstr's 127-char limit on long paths) and
rem repoint the venv at wherever we are now.
set "VENV_HOME=%~dp0.venv\base"
if exist "%VENV_HOME%\python.exe" (
    ".venv\Scripts\python.exe" -c "pass" >nul 2>nul || (
        echo Folder was moved - relinking .venv to %VENV_HOME%
        > ".venv\pyvenv.cfg" echo home = %VENV_HOME%
        >> ".venv\pyvenv.cfg" echo include-system-site-packages = false
        >> ".venv\pyvenv.cfg" echo version = 3.12.10
        >> ".venv\pyvenv.cfg" echo executable = %VENV_HOME%\python.exe
    )
)
rem ONE line on purpose (I142): cmd reads a batch file as it goes, and Help ->
rem Check for Updates may replace this file while the app runs; the pause and
rem the exit are read before the app starts, so nothing of the new file runs.
".venv\Scripts\python.exe" -m kinetrace %* || pause & exit /b

:done
endlocal
