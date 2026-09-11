# V15 cumulative 1-to-13 feature study

This package was built from `Voice_Coil_LSTM_v15_optional_independent_test(7).zip`.
It contains 13 independently runnable copies of the measured-feedback study:

- `F01` uses current only.
- Every next folder adds exactly one causal feature.
- `F13` uses all 13 features from the original V15 one-step model.
- Every folder has its own `Total_Data.xlsx`, code, results folder, and figures folder.

## Run one case

In PowerShell, activate the same environment used for V15, enter a case folder,
and run its `main.py`. For example:

```powershell
conda activate lstm-py312
cd "C:\path\to\V15_Incremental_13_Feature_Models\F01"
python main.py
```

Repeat with `F02`, ..., `F13` as desired. You may also double-click
`RUN_THIS_CASE.bat` from an already configured terminal.

Each case saves:

- numerical results in `ResultsData`;
- plots in `FiguresResults`;
- `ResultsData/feature_audit.json`, proving the actual LSTM input size;
- `ResultsData/case_summary.csv`, summarizing that case.

## Run and compare all 13 cases

From the top-level folder:

```powershell
conda activate lstm-py312
cd "C:\path\to\V15_Incremental_13_Feature_Models"
python main.py
```

The cases run in order from one feature to 13 features. When all cases finish,
the program creates:

- `ComparisonResults/all_case_metrics.csv`;
- `ComparisonResults/per_pure_test_comparison.csv`;
- `ComparisonResults/validation_ranking.csv`;
- `ComparisonResults/prediction_difference_vs_F13.csv`;
- `ComparisonResults/baseline_and_lstm_correction.csv`;
- `ComparisonResults/feature_schedule.csv`;
- `ComparisonResults/comparison_report.md`;
- four comparison plots in `ComparisonFigures`.

To reuse verified completed cases and run only missing cases:

```powershell
python main.py --skip-completed
```

To rebuild only the comparison after all 13 cases exist:

```powershell
python main.py --compare-only
```

For a short software check only (not reportable scientific results):

```powershell
$env:DLSTM_ONE_STEP_EPOCHS="1"
$env:DLSTM_ONE_STEP_SAMPLES_PER_EPOCH="1000"
$env:DLSTM_SKIP_GITHUB_PUSH="1"
python main.py
Remove-Item Env:DLSTM_ONE_STEP_EPOCHS
Remove-Item Env:DLSTM_ONE_STEP_SAMPLES_PER_EPOCH
Remove-Item Env:DLSTM_SKIP_GITHUB_PUSH
```

## Exact cumulative feature schedule

| Case | Features | New feature added |
|---:|---:|---|
| F01 | 1 | current |
| F02 | 2 | current change |
| F03 | 3 | causal current DC estimate |
| F04 | 4 | mass ratio |
| F05 | 5 | inverse mass ratio |
| F06 | 6 | elapsed time |
| F07 | 7 | startup indicator |
| F08 | 8 | previous displacement |
| F09 | 9 | previous velocity |
| F10 | 10 | previous acceleration |
| F11 | 11 | previous force |
| F12 | 12 | force change |
| F13 | 13 | quadratic displacement baseline |

Every case uses a 64-sample history of the retained features. The external
physical excitation is still one signal: current. The other columns are causal
derived variables or previous measured values available in one-step feedback.

## Important interpretation

The LSTM predicts a displacement correction, not force. Displacement is the
quadratic displacement-history baseline plus the learned correction. Force is
calculated by the unchanged separate causal force rule. Therefore:

- compare displacement metrics to study the effect of LSTM features;
- expect force metrics to remain identical across the 13 cases;
- do not artificially expect every added feature to improve accuracy;
- use the validation ranking to compare feature counts;
- use pure-test results only as confirmation, not to select the best case;
- do not describe these one-step measured-feedback results as autonomous simulation.

Because the original displacement-history baseline is strong, even `F01` may
show visually good tracking. The comparison therefore includes the LSTM
correction magnitude and the numerical prediction difference from `F13`, which
make small but real feature effects visible.

## Optional independent workbook

`Test_idpd.xlsx` remains optional. For one separately run case, place it beside
that case's `main.py`. If it is absent, the case runs normally. If you want the
same optional independent tests in all cases, copy the same workbook into every
`F01`–`F13` folder before running all cases.

## GitHub push

An individually run case keeps the V15 automatic GitHub push. During the
top-level 13-case run, child pushes are disabled and the complete project is
committed and pushed once after the comparison is generated. Set
`DLSTM_SKIP_GITHUB_PUSH=1` for a deliberately local run.

