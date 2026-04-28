@echo off
setlocal EnableDelayedExpansion
title NeuroScan AI — Project Launcher

echo.
echo  ================================================
echo    NeuroScan AI ^| Brain Tumour Detection System
echo  ================================================
echo.

:: ── Try to find and activate Conda 'ml' ──────────────────────────────────────
set CONDA_PATH=C:\Users\Devansh Tyagi\miniconda3
if exist "!CONDA_PATH!\Scripts\conda.exe" (
    echo [INFO] Activating 'ml' environment...
    call "!CONDA_PATH!\Scripts\activate.bat" ml
)

:: ── Verify Environment ───────────────────────────────────────────────────────
python -c "import uvicorn, fastapi, torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required Python packages are missing.
    echo Please double-click INSTALL.bat first.
    pause
    exit /b 1
)

:: ── Launch Backend ────────────────────────────────────────────────────────────
echo [OK] Starting Backend...
start "NeuroScan Backend" cmd /k "title NeuroScan Backend && echo Starting FastAPI backend... && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --app-dir backend"

:: Small delay for backend startup
timeout /t 3 /nobreak >nul

:: ── Launch Frontend ───────────────────────────────────────────────────────────
echo [OK] Starting Frontend...
start "NeuroScan Frontend" cmd /k "title NeuroScan Frontend && cd frontend && echo Starting Vite dev server... && npm run dev"

echo.
echo  ================================================
echo    Dashboard: http://localhost:5173
echo  ================================================
echo.

:: Wait and open browser
timeout /t 5 /nobreak >nul
start "" "http://localhost:5173"

echo Press any key to exit this window.
pause >nul
endlocal
