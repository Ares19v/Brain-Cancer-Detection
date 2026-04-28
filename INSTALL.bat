@echo off
setlocal EnableDelayedExpansion
title NeuroScan AI — Installer

echo.
echo  ================================================
echo    NeuroScan AI ^| Installation Script
echo  ================================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER% found.

:: ── Check Node ───────────────────────────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)
for /f %%v in ('node --version') do set NODE_VER=%%v
echo [OK] Node.js %NODE_VER% found.

echo.
echo [STEP 1/3] Setting up Python virtual environment...
if exist "backend\venv" (
    echo         Removing existing venv...
    rmdir /s /q "backend\venv"
)
python -m venv backend\venv
if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
echo [OK] Virtual environment created.

echo.
echo [STEP 2/3] Installing Python dependencies...
backend\venv\Scripts\pip install --upgrade pip -q
backend\venv\Scripts\pip install -r backend\requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )
echo [OK] Python dependencies installed.

echo.
echo [STEP 3/3] Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 ( cd .. & echo [ERROR] npm install failed. & pause & exit /b 1 )
cd ..
echo [OK] Frontend dependencies installed.

echo.
echo  ================================================

:: ── Check weights ─────────────────────────────────────────────────────────────
if exist "weights\efficientnet_b0.pth" (
    echo [OK] Model weights found.
) else (
    echo.
    echo [ACTION REQUIRED] Model weights are missing!
    echo.
    echo    weights\efficientnet_b0.pth was not found.
    echo    Please download it from the link in README.md
    echo    and place it in the weights\ directory.
    echo.
)

echo  Installation complete! Run Run_Project.bat to start.
echo  ================================================
echo.
pause
endlocal
