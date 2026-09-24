@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :launch

echo ============================================================
echo  Kinetrace - first run: creating a self-contained environment in .venv
echo  This downloads ~4 GB (PyTorch with CUDA, less without) and can take a while.
echo  Everything installs INSIDE this folder - deleting the
echo  folder removes the tool completely.
echo ============================================================
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv || goto :trypython
) else (
    goto :trypython
)
goto :install

:trypython
python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python 3.10+ not found. Install it from https://python.org and retry.
    pause
    exit /b 1
)

:install
rem install.py picks the PyTorch build (CUDA with an NVIDIA GPU, else CPU), then requirements.txt
".venv\Scripts\python.exe" install.py
if errorlevel 1 (
    echo.
    echo ERROR: dependency installation failed. Check your internet connection
    echo and re-run run.bat ^(it will resume^).
    pause
    exit /b 1
)

:launch
rem AllTracker (the default point model) is fetched at install; retry if that was missed
if not exist "models\alltracker\nets\alltracker.py" ".venv\Scripts\python.exe" install.py --alltracker-only
rem The interpreter lives inside the folder (.venv\base). If the folder was
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
".venv\Scripts\python.exe" -m cotracker_app %*
if errorlevel 1 pause
endlocal
