@echo off
rem Start the app and open it in the browser (run from anywhere)
cd /d "%~dp0"
start "" cmd /c "timeout /t 8 /nobreak > nul & start http://127.0.0.1:8000"
uvicorn app:app --host 127.0.0.1 --port 8000
