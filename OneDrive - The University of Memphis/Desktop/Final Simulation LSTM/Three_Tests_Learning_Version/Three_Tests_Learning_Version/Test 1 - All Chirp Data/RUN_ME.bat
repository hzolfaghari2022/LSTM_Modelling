REM Hide command echoing so the terminal shows only useful progress and errors.
@echo off
REM Start in this test folder even when the launcher is opened elsewhere.
cd /d "%~dp0"
REM Activate the environment, run the simulation, and keep the window open.
call conda activate lstm-py312
REM Run the stated Python entry point with the active environment.
python main.py
REM Wait for a key press so the terminal message remains visible.
pause
