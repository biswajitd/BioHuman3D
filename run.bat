@echo off
REM ===========================================================================
REM  BioHuman3D - launcher
REM
REM  Double-click this file, or from a terminal:
REM      run.bat            launch the application
REM      run.bat doctor     diagnose the environment (deps, GPU, local LLMs)
REM      run.bat models     (re)generate placeholder anatomy
REM      run.bat test       run the headless UI smoke test
REM      run.bat check      verify everything without launching the GUI
REM      run.bat setup      create/refresh the .venv and install dependencies
REM
REM  Dependencies are installed in two tiers:
REM    requirements.txt        CORE   - must succeed, or the app cannot start
REM    requirements-extras.txt EXTRAS - best effort, never aborts setup
REM
REM  Exit codes: 0 = success, 1 = setup/environment error, or the app's own code
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
title BioHuman3D - Anatomy ^& Health Studio

REM UTF-8 console so model names containing non-ASCII print correctly.
chcp 65001 >nul 2>&1

REM Always operate from the folder that contains this script.
cd /d "%~dp0"

set "APP=main.py"
set "VENV_DIR=.venv"
set "PYEXE="
set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"

echo ===========================================================================
echo  BioHuman3D - Anatomy ^& Health Studio
echo ===========================================================================
echo.

REM ---------------------------------------------------------------------------
REM  1. Resolve the Python interpreter: local venv first, then launcher/PATH.
REM ---------------------------------------------------------------------------
if /i "%MODE%"=="setup" goto :do_setup

if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PYEXE=%VENV_DIR%\Scripts\python.exe"
    echo [1/6] Interpreter : %VENV_DIR%\Scripts\python.exe  [virtual environment]
    goto :have_python
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYEXE=python"
    echo [1/6] Interpreter : python  [system, from PATH]
    goto :have_python
)

where py >nul 2>&1
if not errorlevel 1 (
    set "PYEXE=py -3"
    echo [1/6] Interpreter : py -3  [Windows Python launcher]
    goto :have_python
)

echo [1/6] Interpreter : NOT FOUND
echo.
echo   [ERROR] No Python installation was found.
echo.
echo   Install Python 3.10 or newer from https://www.python.org/downloads/
echo   and be sure to tick "Add python.exe to PATH" during setup,
echo   then run this file again.
echo.
pause
exit /b 1

:have_python

REM Report the exact version, and refuse anything older than 3.10.
%PYEXE% -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [ERROR] Python 3.10 or newer is required. Found:
    %PYEXE% --version
    echo.
    pause
    exit /b 1
)
for /f "delims=" %%V in ('%PYEXE% --version 2^>^&1') do echo         version     : %%V

REM ---------------------------------------------------------------------------
REM  2. CORE dependencies - these must succeed.
REM ---------------------------------------------------------------------------
%PYEXE% -c "import PyQt6, vtk" >nul 2>&1
if errorlevel 1 (
    echo [2/6] Core deps  : missing - installing, this can take a few minutes...
    echo.
    %PYEXE% -m pip install --disable-pip-version-check --upgrade pip
    %PYEXE% -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   [ERROR] Could not install the core dependencies ^(PyQt6 and VTK^).
        echo.
        echo   Try these in order:
        echo     1. Check your internet connection / proxy, then retry: run.bat
        echo     2. Upgrade pip:      %PYEXE% -m pip install --upgrade pip
        echo     3. Use a local venv:  run.bat setup
        echo     4. Confirm your Python has wheels available for PyQt6 and VTK.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo   Core dependencies installed.
) else (
    echo [2/6] Core deps  : PyQt6 + VTK present
)

REM ---------------------------------------------------------------------------
REM  3. OPTIONAL extras - voiceover and audio playback. Never fatal.
REM     --only-binary forces wheel-only installs so anything without a prebuilt
REM     wheel fails in one clean line instead of dumping a compiler log.
REM ---------------------------------------------------------------------------
%PYEXE% -c "import pyttsx3, pygame" >nul 2>&1
if errorlevel 1 (
    echo [3/6] Extras     : installing optional voice/audio packages...
    %PYEXE% -m pip install --disable-pip-version-check --only-binary=:all: -r requirements-extras.txt
    if errorlevel 1 (
        echo.
        echo   [WARN] Some optional packages could not be installed.
        echo          BioHuman3D will still launch. Voiceover will be silent, or
        echo          run through whatever backends are already available.
        echo          See the comments in requirements-extras.txt for alternatives.
        echo.
    )
)

set "TTS=no"
set "MIXER=no"
%PYEXE% -c "import pyttsx3" >nul 2>&1
if not errorlevel 1 set "TTS=yes"
%PYEXE% -c "import pygame" >nul 2>&1
if not errorlevel 1 set "MIXER=yes"

if "%TTS%"=="yes" if "%MIXER%"=="yes" (
    echo [3/6] Extras     : system voices + audio player ready
) else (
    if "%TTS%"=="yes" (
        echo [3/6] Extras     : system voices ready, no audio player ^(cloud clips only^)
    ) else (
        echo [3/6] Extras     : not installed - voiceover will be silent
    )
)

REM ---------------------------------------------------------------------------
REM  4. Make sure there is geometry to display.
REM ---------------------------------------------------------------------------
set "MODELCOUNT=0"
for /f %%C in ('dir /b /a-d "app\assets\models" 2^>nul ^| find /c /v ""') do set "MODELCOUNT=%%C"

if "%MODELCOUNT%"=="0" (
    echo [4/6] 3D models  : none found - generating placeholder anatomy...
    %PYEXE% tools\generate_demo_models.py
    if errorlevel 1 (
        echo   [WARN] Model generation failed; the app will use simple stand-in shapes.
    )
) else (
    echo [4/6] 3D models  : %MODELCOUNT% file^(s^) in app\assets\models
)

REM ---------------------------------------------------------------------------
REM  5. Sub-commands that should not launch the GUI.
REM ---------------------------------------------------------------------------
if /i "%MODE%"=="doctor" (
    echo [5/6] Mode       : doctor
    echo.
    %PYEXE% tools\doctor.py
    echo.
    pause
    exit /b %ERRORLEVEL%
)

if /i "%MODE%"=="models" (
    echo [5/6] Mode       : regenerate models
    echo.
    %PYEXE% tools\generate_demo_models.py --force
    echo.
    pause
    exit /b %ERRORLEVEL%
)

if /i "%MODE%"=="test" (
    echo [5/6] Mode       : headless UI smoke test
    echo.
    %PYEXE% -u tools\ui_smoke_test.py --show
    echo.
    pause
    exit /b %ERRORLEVEL%
)

if /i "%MODE%"=="check" (
    echo [5/6] Mode       : check only - not launching the GUI
    echo.
    %PYEXE% tools\doctor.py
    echo.
    echo [6/6] Launch     : skipped  ^(check mode^)
    exit /b %ERRORLEVEL%
)

REM ---------------------------------------------------------------------------
REM  6. Launch the application.
REM ---------------------------------------------------------------------------
echo [5/6] Folder     : %CD%
echo [6/6] Launching BioHuman3D...
echo.

%PYEXE% "%APP%"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo ===========================================================================
    echo   BioHuman3D exited with code %RC%.
    echo.
    echo   For a full diagnosis run:   run.bat doctor
    echo   Common causes: missing/outdated GPU driver, another application holding
    echo                 the GPU, or a Python traceback printed above.
    echo ===========================================================================
    echo.
    pause
    exit /b %RC%
)

endlocal
exit /b 0

REM ---------------------------------------------------------------------------
REM  Setup path: create (or refresh) a local virtual environment.
REM ---------------------------------------------------------------------------
:do_setup
echo [setup] Creating/refreshing virtual environment in "%VENV_DIR%" ...
echo.

if not exist "%VENV_DIR%\Scripts\python.exe" (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -m venv "%VENV_DIR%"
    ) else (
        where py >nul 2>&1
        if not errorlevel 1 (
            py -3 -m venv "%VENV_DIR%"
        ) else (
            echo   [ERROR] Python was not found, so no virtual environment can be created.
            echo   Install Python from https://www.python.org/downloads/ first.
            echo.
            pause
            exit /b 1
        )
    )
    if not exist "%VENV_DIR%\Scripts\python.exe" (
        echo   [ERROR] Failed to create the virtual environment.
        echo.
        pause
        exit /b 1
    )
)

echo [setup] Installing CORE dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo   [ERROR] Core dependency installation failed.
    echo.
    pause
    exit /b 1
)

echo.
echo [setup] Installing OPTIONAL extras ^(non-fatal^)...
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check --only-binary=:all: -r requirements-extras.txt
if errorlevel 1 (
    echo   [WARN] Optional extras were skipped. The app still runs without voiceover.
)

echo.
echo [setup] Generating placeholder anatomy...
"%VENV_DIR%\Scripts\python.exe" tools\generate_demo_models.py

echo.
echo ===========================================================================
echo   Setup complete. Launch the application with:   run.bat
echo ===========================================================================
echo.
pause
exit /b 0
