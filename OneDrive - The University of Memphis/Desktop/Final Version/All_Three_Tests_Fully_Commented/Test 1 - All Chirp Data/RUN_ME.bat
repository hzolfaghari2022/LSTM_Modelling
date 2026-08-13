@echo off
REM Start in this test folder even when the launcher is opened elsewhere.
cd /d "%~dp0"
REM Activate the environment, run the simulation, and keep the window open.
call conda activate lstm-py312
python main.py
pause
