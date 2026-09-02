@echo off
cd /d "%~dp0"
call conda activate lstm-py312
python run_all.py
pause
