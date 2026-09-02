@echo off
cd /d "%~dp0"
call conda activate lstm-py312
python check_data.py
pause
