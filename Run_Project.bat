@echo off
setlocal EnableDelayedExpansion
title NeuroScan AI — Project Launcher

echo.
echo  ================================================
echo    NeuroScan AI ^| Brain Tumour Detection System
echo  ================================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+ and add it to PATH.
    pause
    exit /b 1
)

:: ── Check Node ───────────────────────────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 18+ and add it to PATH.
    pause
    exit /b 1
)

:: ── Check model weights ───────────────────────────────────────────────────────
if not exist "weights\efficientnet_b0.pth" (
    echo [WARNING] Model weights not found at weights\efficientnet_b0.pth
    echo           Download them from the link in README.md before running predictions.
    echo.
)

:: ── Backend venv setup ───────────────────────────────────────────────────────
if not exist "backend\venv\Scripts\python.exe" (
    echo [SETUP] Creating Python virtual environment...
    python -m venv backend\venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SETUP] Installing Python dependencies...
    backend\venv\Scripts\pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install Python dependencies.
        pause
        exit /b 1
    )
    echo [SETUP] Backend dependencies installed successfully.
) else (
    echo [OK] Backend virtual environment found.
)

:: ── Frontend node_modules ────────────────────────────────────────────────────
if not exist "frontend\node_modules" (
    echo [SETUP] Installing frontend dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        cd ..
        echo [ERROR] Failed to install frontend dependencies.
        pause
        exit /b 1
    )
    cd ..
    echo [SETUP] Frontend dependencies installed successfully.
) else (
    echo [OK] Frontend node_modules found.
)

echo.
echo  Starting services...
echo.

:: ── Launch Backend ────────────────────────────────────────────────────────────
start "NeuroScan Backend (port 8000)" cmd /k "title NeuroScan Backend && echo Starting FastAPI backend... && backend\venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --app-dir backend"

:: Small delay so backend starts first
timeout /t 3 /nobreak >nul

:: ── Launch Frontend ───────────────────────────────────────────────────────────
start "NeuroScan Frontend (port 5173)" cmd /k "title NeuroScan Frontend && cd frontend && echo Starting Vite dev server... && npm run dev"

echo.
echo  ================================================
echo    Both services are launching in new windows.
echo.
echo    Backend  : http://localhost:8000
echo    Frontend : http://localhost:5173
echo    API Docs : http://localhost:8000/docs
echo  ================================================
echo.

:: Wait a moment then open the browser
timeout /t 5 /nobreak >nul
start "" "http://localhost:5173"

echo  Press any key to close this launcher window.
pause >nul
endlocal
