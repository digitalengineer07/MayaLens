@echo off
cd /d "%~dp0"
echo ========================================================
echo           Launching MayaLens Web Application            
echo ========================================================
echo.

echo [1/4] Ensuring ports 8000 and 5173 are free from previous runs...
powershell -Command "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -like '*uvicorn*' -or $_.CommandLine -like '*backend.main:app*' -or $_.CommandLine -like '*spawn_main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
powershell -Command "Get-NetTCPConnection -LocalPort 8000, 5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo [2/4] Starting FastAPI Backend on port 8000...
start "MayaLens Backend (FastAPI)" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo [3/4] Starting React Vite Frontend on port 5173...
start "MayaLens Frontend (Vite)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo [4/4] Verifying backend connectivity on http://127.0.0.1:8000...
powershell -Command "$ready = $false; for ($i=0; $i -lt 40; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { $ready = $true; break; } } catch {}; Start-Sleep -Milliseconds 300 }; if ($ready) { Write-Host 'Backend is online and listening!' -ForegroundColor Green } else { Write-Host 'Backend is still initializing...' -ForegroundColor Yellow }"

echo Opening browser at http://localhost:5173...
start http://localhost:5173
echo.
echo Done! Keep the two command windows open while using MayaLens.
pause
