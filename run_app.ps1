# MayaLens Development Runner
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    Starting MayaLens Web Application   " -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Start Backend
Write-Host "[1/2] Starting FastAPI Backend on port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python -m uvicorn backend.main:app --reload --port 8000"

# Start Frontend
Write-Host "[2/2] Starting React + Vite Frontend on port 5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"

Write-Host "MayaLens is running!" -ForegroundColor Cyan
Write-Host "Open in your browser: http://localhost:5173" -ForegroundColor Yellow

