@echo off
setlocal EnableDelayedExpansion
title NeuroScan AI — Uninstaller

echo.
echo  ================================================
echo    NeuroScan AI ^| Uninstallation Script
echo  ================================================
echo.
echo  This will remove:
echo    - backend\venv\      (Python virtual environment)
echo    - frontend\node_modules\  (npm packages)
echo    - frontend\dist\     (build output)
echo.
echo  Your source code, datasets, and model weights will NOT be deleted.
echo.
set /p CONFIRM=Type YES to continue: 
if /i not "!CONFIRM!"=="YES" (
    echo Aborted.
    pause
    exit /b 0
)

echo.
if exist "backend\venv" (
    echo [REMOVE] backend\venv ...
    rmdir /s /q "backend\venv"
    echo [OK] backend\venv removed.
) else (
    echo [SKIP] backend\venv not found.
)

if exist "frontend\node_modules" (
    echo [REMOVE] frontend\node_modules ...
    rmdir /s /q "frontend\node_modules"
    echo [OK] frontend\node_modules removed.
) else (
    echo [SKIP] frontend\node_modules not found.
)

if exist "frontend\dist" (
    echo [REMOVE] frontend\dist ...
    rmdir /s /q "frontend\dist"
    echo [OK] frontend\dist removed.
) else (
    echo [SKIP] frontend\dist not found.
)

echo.
echo  Uninstall complete. Run INSTALL.bat to reinstall.
echo  ================================================
echo.
pause
endlocal
