@echo off
REM Double-click or run from cmd: activates the venv and runs the full pipeline.
REM Extra arguments are passed through, e.g.  run_overnight.bat --smoke
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run_all.py %*
echo.
echo Finished. Open RESULTS.md
