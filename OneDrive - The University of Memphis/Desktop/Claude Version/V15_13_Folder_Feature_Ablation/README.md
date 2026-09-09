# V15 — 13 Standalone Cumulative Feature-Ablation Models

This package contains **13 separately runnable models**. The first model uses
all 13 causal LSTM features. Each following model removes the final retained
feature in the fixed V15 order until the last model uses current only.

The actuator still has **one externally applied physical input: coil current**.
The 13 items are causal engineered/history/configuration features available to
the one-step displacement-residual LSTM. The Lorentz-force prediction remains
the unchanged separate causal force rule, so this ablation directly tests the
displacement LSTM only.

## Folder structure

```text
V15_13_Folder_Feature_Ablation/
├── Total_Data.xlsx
├── Test_idpd.xlsx                 optional; add it here if available
├── main.py                        runs and compares all 13 cases
├── RUN_ALL_13_MODELS.bat
├── RUN_ALL_REUSE_COMPLETED.bat
├── push_now.py
├── SharedCode/                    top-level comparison support
├── Model_13_Features/             complete Python code + its own main.py
├── Model_12_Features/             complete Python code + its own main.py
├── ...
├── Model_02_Features/main.py
└── Model_01_Feature/              complete Python code + its own main.py
```

Every model folder contains its own complete Python source. The large measured
workbook is stored once at the top level to avoid 13 unnecessary copies of the
unchanged experimental data. Every model folder is independently runnable while
it remains inside this package, and it locates that workbook one level above.

## Run one case independently

Open the selected folder in VS Code and run its `main.py`, or use PowerShell:

```powershell
cd ".\Model_08_Features"
python main.py
```

That case writes only to:

```text
Model_08_Features\ResultsData
Model_08_Features\FiguresResults
```

Running one folder directly performs the existing automatic GitHub push unless
`DLSTM_SKIP_GITHUB_PUSH=1` is set.

## Run and compare all 13 cases

From the top-level project folder:

```powershell
python main.py
```

or double-click `RUN_ALL_13_MODELS.bat`. The cases run in this order:
13, 12, ..., 1 features. Per-case pushes are suppressed during this command,
and one GitHub push is performed after all comparisons are saved.

To reuse cases that have already completed and run only missing cases:

```powershell
python main.py --skip-completed
```

## Cumulative removal order

| Model | Retained features | Newly removed relative to previous model |
|---|---|---|
| 13 | features 1–13 | none |
| 12 | features 1–12 | quadratic displacement baseline |
| 11 | features 1–11 | force change |
| 10 | features 1–10 | previous force |
| 9 | features 1–9 | previous acceleration |
| 8 | features 1–8 | previous velocity |
| 7 | features 1–7 | previous displacement |
| 6 | features 1–6 | startup indicator |
| 5 | features 1–5 | elapsed time |
| 4 | features 1–4 | inverse mass ratio |
| 3 | features 1–3 | mass ratio |
| 2 | features 1–2 | causal DC-current estimate |
| 1 | current only | current change |

Exact full feature order:

1. `current`
2. `current_change`
3. `current_dc_estimate`
4. `mass_ratio`
5. `inverse_mass_ratio`
6. `elapsed_time`
7. `startup_indicator`
8. `previous_displacement`
9. `previous_velocity`
10. `previous_acceleration`
11. `previous_force`
12. `force_change`
13. `quadratic_displacement_baseline`

## Comparison outputs

After all cases finish:

```text
ComparisonResults\all_13_models_comparison.csv
ComparisonResults\validation_model_ranking.csv
ComparisonResults\COMPARISON_REPORT.txt
ComparisonFigures\01_all_13_models_accuracy.png
ComparisonFigures\02_training_validation_error_vs_features.png
ComparisonFigures\03_feature_retention_matrix.png
```

The comparison selects the best case by **validation displacement RMSE**.
Pure-test values confirm the already selected result and are not used to choose
the feature count.

## Notes

- CUDA is selected automatically and the GPU name is printed.
- All figures use Times New Roman when installed, with Garamond/serif fallbacks.
- `Total_Data.xlsx` is the required shared development workbook.
- `Test_idpd.xlsx` remains optional at the top level.
- To disable pushing temporarily in PowerShell:

```powershell
$env:DLSTM_SKIP_GITHUB_PUSH="1"
```

- To push saved code/results without retraining:

```powershell
python push_now.py
```
