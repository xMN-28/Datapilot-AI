@echo off
setlocal

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "FRONTEND=%ROOT%frontend"

echo Starting DataPilot AI...

if not exist "%BACKEND%\requirements.txt" (
  echo Backend folder not found.
  pause
  exit /b 1
)

if not exist "%FRONTEND%\package.json" (
  echo Frontend folder not found.
  pause
  exit /b 1
)

if not exist "%FRONTEND%\node_modules" (
  echo Installing frontend dependencies...
  pushd "%FRONTEND%"
  call npm.cmd install
  if errorlevel 1 (
    echo Frontend dependency install failed.
    pause
    exit /b 1
  )
  popd
)

start "DataPilot AI Backend" cmd /k "cd /d "%BACKEND%" && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "DataPilot AI Frontend" cmd /k "cd /d "%FRONTEND%" && npm.cmd run dev -- --host 127.0.0.1 --port 5173"

echo.
echo DataPilot AI is starting.
echo Frontend: http://127.0.0.1:5173/
echo Backend:  http://127.0.0.1:8000/api/health
echo.
timeout /t 3 >nul
start http://127.0.0.1:5173/

endlocal
