@echo off
cd /d "%~dp0"
call conda activate lstm-py312
python trustworthy_evaluation.py --mode repeatability
pause
