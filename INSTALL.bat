@echo off
setlocal EnableDelayedExpansion
title NeuroScan AI — One-Click Installer

echo.
echo  ================================================
echo    NeuroScan AI ^| One-Click Installer
echo  ================================================
echo.

:: ── Try to find Conda ────────────────────────────────────────────────────────
set CONDA_PATH=C:\Users\Devansh Tyagi\miniconda3
if exist "!CONDA_PATH!\Scripts\conda.exe" (
    echo [INFO] Conda detected. Activating 'ml' environment...
    call "!CONDA_PATH!\Scripts\activate.bat" ml
) else (
    echo [WARNING] Conda not found at !CONDA_PATH!. Using system Python.
)

:: ── Check Python ─────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python or Miniconda.
    pause
    exit /b 1
)

:: ── Clean up old blocked venv ────────────────────────────────────────────────
if exist "backend\venv" (
    echo [CLEAN] Removing old virtual environment folder...
    rmdir /s /q "backend\venv"
)

echo.
echo [STEP 1/2] Installing Python dependencies...
python -m pip install -r backend\requirements.txt
if errorlevel 1 (
    echo [ERROR] Python installation failed.
    pause
    exit /b 1
)

echo.
echo [STEP 2/2] Installing frontend dependencies...
if not exist "frontend\node_modules" (
    cd frontend
    call npm install
    if errorlevel 1 ( cd .. & echo [ERROR] npm install failed. & pause & exit /b 1 )
    cd ..
) else (
    echo [SKIP] node_modules already exists.
)

echo.
echo  ================================================
echo  Installation complete! 
echo  Now just double-click Run_Project.bat
echo  ================================================
echo.
pause
endlocal
