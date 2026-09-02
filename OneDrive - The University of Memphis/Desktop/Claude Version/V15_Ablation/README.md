# V15 feature-ablation runner

This Windows-safe package supports three equivalent workflows. Every case
uses its own folder and saves its own numerical results and figures.

## 1. Check the workbook first

From this top-level folder:

```powershell
python check_data.py
```

The check should discover and cache 19 worksheets. The Excel workbook is
opened once, and progress is printed for every sheet.

## 2. Run one case independently

Open the desired folder and run `main.py`. For example:

```powershell
cd .\F12
python main.py
```

Alternatively, double-click `RUN_ME.bat` inside that folder. Results are
written only to that folder's `ResultsData` and `FiguresResults`. The
completed case is then committed and pushed to GitHub automatically.

Folder names correspond to retained feature counts: `F13` retains all 13
features, `F12` retains 12, and so on through `F01`.

## 3. Run selected cases or all cases

Return to this top-level folder before using these commands:

```powershell
python run_all.py --dry-run
python run_all.py --counts 13 12 11
python run_all.py
```

The first command checks the folder plan without training. The second runs
only the listed cases. The third runs all cases sequentially from 13 down to
1. Each completed case is pushed separately. If a case fails, the sequence
stops instead of skipping it.

## Compare all results currently available

You may run cases individually over several days. At any time, return to the
top-level folder and run:

```powershell
python compare.py
```

The comparison script finds every folder that currently contains
`ResultsData/one_step_metrics.csv`, skips unfinished folders, and creates:

```text
Comparison/
  all_pure_test_metrics.csv
  feature_ablation_summary.csv
  feature_ablation_comparison.png
```

It then pushes the updated comparison to GitHub. Running `run_all.py` also
runs this comparison automatically after its requested cases finish.

## CUDA warning and Windows path fix

Every `main.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing
PyTorch, eliminating the repeated deterministic-cuBLAS warnings seen in the
previous terminal output. The root and case folders are intentionally short
(`V15_Ablation/F01`, etc.) to remain safely below Windows' legacy 260-character
path limit when long prediction CSV filenames are created.

Extract the ZIP using Windows **Extract All**. Do not place it inside another
folder with the same name. Keeping it inside your correctly cloned
`LSTM_Modelling` repository is required for automatic GitHub pushing.

## Important model scope

These folders ablate the adapted V15 one-step model. They do not reproduce
the original Ogunmolu et al. paper architecture. No network architecture,
feature-removal rule, training setting, split, or measured Excel value was
changed as part of this runtime repair.

