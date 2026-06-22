@echo off
set /p GEMINI_API_KEY=Paste GEMINI_API_KEY and press Enter: 
set GEMINI_MODEL=gemini-2.5-flash-lite
py app.py
pause
