@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title RetailPulse Dashboard

echo ==================================================
echo       RetailPulse - Dashboard Launcher
echo ==================================================
echo Project folder: %CD%
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating project virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo ERROR: Python launcher could not create .venv.
    echo Check that Python 3.13 is installed and available in PATH.
    pause
    exit /b 1
  )
)

set "PY=.venv\Scripts\python.exe"
echo [2/4] Checking Streamlit...
"%PY%" -m pip --version
if errorlevel 1 goto :error
"%PY%" -m streamlit --version >nul 2>&1
if errorlevel 1 (
  echo Streamlit is missing. Installing project dependencies; internet is required the first time...
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

if not exist "reports\summary.json" (
  echo [3/4] Preparing dashboard data for the first run...
  "%PY%" scripts\run_pipeline.py
  if errorlevel 1 goto :error
) else (
  echo [3/4] Existing dashboard reports found.
)

echo [4/4] Starting dashboard. Keep this window open.
echo If the browser does not open, visit http://localhost:8501
start "" "http://localhost:8501"
"%PY%" -m streamlit run app\streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo ==================================================
echo RetailPulse could not start. Read the error above.
echo Make sure internet is available if packages are being installed.
echo ==================================================
pause
exit /b 1
