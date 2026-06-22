@echo off
cd /d %~dp0
set /p GEMINI_API_KEY=Paste GEMINI_API_KEY and press Enter: 
set GEMINI_MODEL=gemini-2.5-flash-lite
set GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-flash-latest
py check_gemini.py
if errorlevel 1 (
  echo Gemini connection failed. Please renew or check the API key.
  pause
  exit /b 1
)
py app.py
pause
