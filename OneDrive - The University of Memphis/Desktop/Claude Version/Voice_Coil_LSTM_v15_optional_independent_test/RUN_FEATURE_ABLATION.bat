@echo off
call conda activate lstm-py312
python ablation_study.py --study features
pause
