@echo off
setlocal
cd /d "%~dp0"
call conda activate lstm-py312
set DLSTM_SKIP_GITHUB_PUSH=1
python trustworthy_evaluation.py --mode all --smoke-test
pause
endlocal
