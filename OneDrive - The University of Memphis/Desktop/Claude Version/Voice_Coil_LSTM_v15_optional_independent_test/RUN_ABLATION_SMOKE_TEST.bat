@echo off
call conda activate lstm-py312
set DLSTM_SKIP_GITHUB_PUSH=1
python ablation_study.py --study all --smoke-test
set DLSTM_SKIP_GITHUB_PUSH=
pause
