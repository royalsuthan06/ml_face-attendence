@echo off
setlocal enabledelayedexpansion
title Attendify Enhanced - System Setup
color 0b

set "PYTHON_PATH=C:\Users\sutha\AppData\Local\Programs\Python\Python313\python.exe"
set "VENV_DIR=.venv"

echo ===================================================
echo   ATTENDIFY: TOTAL SYSTEM INITIALIZATION
echo ===================================================

:: PHASE 1: VENV CREATION (0% - 20%)
echo [20%%] Checking Virtual Environment...
if not exist "%VENV_DIR%" (
    echo      - Creating isolated environment...
    "%PYTHON_PATH%" -m venv "%VENV_DIR%" >nul
    echo [OK] Environment Created.
) else (
    echo [OK] Environment already exists.
)

:: PHASE 2: CORE TOOLS (20% - 40%)
echo [40%%] Installing Build Tools (CMake/Pip)...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
"%VENV_DIR%\Scripts\python.exe" -m pip install cmake setuptools >nul 2>&1

:: PHASE 3: WEB & DATABASE (40% - 70%)
echo [70%%] Syncing Web and Database Modules...
"%VENV_DIR%\Scripts\python.exe" -m pip install numpy opencv-python flask bcrypt >nul 2>&1

:: PHASE 4: AI ENGINE (70% - 100%)
echo [90%%] Finalizing AI Core (face_recognition)...
echo      - NOTE: This part is heavy, please wait...
"%VENV_DIR%\Scripts\python.exe" -m pip install face_recognition >nul 2>&1

echo [100%%] ALL SYSTEMS OPERATIONAL.
echo ===================================================
echo   LAUNCHING ATTENDIFY DASHBOARD...
echo ===================================================
"%VENV_DIR%\Scripts\python.exe" app.py

if errorlevel 1 (
    color 0c
    echo [!] CRITICAL ERROR: System failed to launch.
    echo [!] Possible reason: dlib compilation failed (Needs CMake).
)
pause