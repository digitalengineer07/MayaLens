@echo off
echo Starting MayaLens Backend and Frontend...
start cmd /k "python -m uvicorn backend.main:app --reload --port 8000"
start cmd /k "cd frontend && npm run dev"
echo Both servers are launching!
echo Open your browser at http://localhost:5173
timeout /t 3 >nul
start http://localhost:5173

