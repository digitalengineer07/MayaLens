@echo off
cd /d "%~dp0"
echo ========================================================
echo           Launching MayaLens Web Application            
echo ========================================================
echo.
echo [1/2] Starting FastAPI Backend on port 8000...
start "MayaLens Backend (FastAPI)" cmd /k "python -m uvicorn backend.main:app --reload --port 8000"

echo [2/2] Starting React Vite Frontend on port 5173...
start "MayaLens Frontend (Vite)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both servers are starting up in separate terminal windows!
echo Opening browser at http://localhost:5173 in 4 seconds...
timeout /t 4 >nul
start http://localhost:5173
echo.
echo Done! Keep the two command windows open while using MayaLens.
pause
