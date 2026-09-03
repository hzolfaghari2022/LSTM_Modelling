# Voice-Coil One-Step LSTM Identification

## Excel files beside main.py

The development data filename is now exact:

```text
Total_Data.xlsx
```

Only that file supplies development, training, validation, internal-test, and
the original built-in pure-test records. Files such as `Total_Data(1).xlsx`
are deliberately ignored.

### Worksheet names and data roles

`Total_Data.xlsx` now places exactly one experiment in each worksheet. The
worksheet-name prefix makes its role visible without changing any measured
value:

- `TRAIN_VAL_TEST_...`: a development experiment that the code divides into
  training, validation, and internal-test time blocks;
- `PURE_TEST_...`: a complete untouched experiment used only for final pure
  testing;
- `DUPLICATE_IGNORED_...`: a preserved duplicate export that the existing
  duplicate check excludes from model development and evaluation.

These prefixes are labels for the reader. The program still discovers each
experiment from its metadata and measured columns, so renaming or reordering
worksheets does not silently change the model logic. To add or remove a
built-in pure-test experiment, add or remove its complete worksheet and keep
the experiment metadata and four measured columns in the same layout.

An additional workbook is optional:

```text
Test_idpd.xlsx
```

When `Test_idpd.xlsx` is absent, the program runs normally and does not require
it. When it is beside `main.py`, every recognised case in it is kept whole as
an untouched independent test. It contributes nothing to training,
normalisation, physical fitting, early stopping, residual-trust selection, or
model selection. The program adds its comparisons to `FiguresResults` and
writes its separate metric table to:

```text
ResultsData/one_step_independent_test_metrics.csv
```

`Test_idpd.xlsx` must use the same COMSOL workbook layout and units as
`Total_Data.xlsx`: the metadata must state the load mass, and each case must
have the four columns Time (s), Displacement (mm), Coil Current (A), and
Lorentz force (N). The order of sheets and the number of cases may differ.

## Run the correct file

Run `main.py` for the high-accuracy measured-versus-predicted results:

```bash
python -m pip install -r requirements.txt
python main.py
```

On Windows, `RUN_ME.bat` does the same thing in the `lstm-py312` environment.
Do not run `model.py`; it only defines a class and therefore exits silently.

`main.py` produces the primary one-step report in one folder,
`FiguresResults`. The model and identification procedure are unchanged; the
plotting stage now evaluates and displays every development-data role after
the checkpoint has been frozen:

In every measured-versus-predicted time plot, the measured signal is a thick
solid black curve. The LSTM prediction is drawn above it using a vivid color,
long dashes, and sparse white-centred markers. This makes nearly coincident
curves distinguishable on screen, in print, and for readers with limited
color perception.

- Figure 1: complete inventory of every development, pure-test, and optional
  independent-test record;
- Figure 2: training/validation/internal-test/pure-test split map;
- Figure 3: learning curves;
- Figures 4 and 5: measured versus predicted displacement and force for all
  13 development records, with colored data-role backgrounds;
- Figure 6.1--6.5: detailed untouched pure-test comparisons and errors;
- Figure 7: pure-test parity plots;
- Figure 8: accuracy summary for every record;
- Figures 9 and 10: measured step- and zero-input validation;
- Figure 11: untouched-chirp time and frequency detail;
- Figures 12 and 13: pooled and complete metric tables.
- Figure 14: explicit Load-2 zero-input test with `I(t)=0 A`, initialized only
  from measured `x0`, `velocity0`, and `force0`.
- Figures 15 and 16: signed displacement and force prediction errors for every
  development experiment;
- Figures 17 and 18: signed displacement and force prediction errors for every
  untouched pure or optional independent test. Each error subplot is scaled
  symmetrically around zero, marks the maximum absolute error with a red dot,
  and reports RMSE, MAE, maximum error, and the time of that maximum.

- `ResultsData/one_step_metrics.csv` contains per-record training, validation,
  internal-test, and pure-test results;
- `ResultsData/one_step_pooled_role_metrics.csv` contains pooled results for
  all four roles;
- `ResultsData/one_step_predictions/*.csv`
- `ResultsData/one_step_prediction_file_map.csv` maps the short numbered CSV
  filenames to their complete data role and experiment name. Short filenames
  prevent Windows' 260-character path-limit failure in deeply nested folders.
- `ResultsData/explicit_zero_input_test_metrics.csv`

After one complete `main.py` run, the same figure set can be rebuilt without
training again by running `python plot_results.py`.

## Automatic GitHub push

At the end of every successful `main.py` run, the project commits its changed
code, results, and figures and pushes them to:

```text
hzolfaghari2022/LSTM_Modelling
```

The project folder must be inside a local clone of that repository, `origin`
must point to that GitHub repository, and Git Credential Manager or GitHub CLI
must already be authenticated. No password or access token is stored in the
code. The push stages only explicitly detected changes inside this project
folder and refuses to include unrelated staged files.

To push existing saved results without retraining, run:

```bash
python push_now.py
```

For a deliberately local run only, set `DLSTM_SKIP_GITHUB_PUSH=1` before
running `main.py`.

The harder autonomous/free-running experiment has its own explicit entry point:

```bash
python autonomous_simulation.py
```

It writes to `AutonomousResultsData` and `AutonomousFiguresResults`. Its scores
must not be presented as the one-step scores.

## Prediction definition

The primary model is a causal one-step measured-feedback predictor. To predict
sample `k`, it may use:

- known coil current through sample `k`;
- measured displacement and Lorentz force only through sample `k-1`;
- load-mass/configuration features.

It never uses the measured output at sample `k`. The LSTM predicts a residual
correction to a causal constant-acceleration displacement baseline. A scalar
trust factor is selected using development validation blocks only. Force uses
the known causal actuator relationship and the latest reliable past force to
current gain, which prevents spikes at sine-wave zero crossings.

## Untouched pure-test policy

Five complete records are excluded from every fitted quantity, including
normalisation, physical fitting, LSTM gradients, early stopping, trust-factor
selection, and startup-force-gain fitting:

- Load-2 147 mA chirp;
- Load-2 150 mA step;
- Load-2 zero input;
- Load-3 200 mA step;
- Load-3 200 mA DC+sine.

The other 13 development records are divided into 0.2 s blocks following
`training, validation, training, internal test`. Chirp and non-chirp records
therefore contribute to training, validation, and internal testing.

The workbook units are displacement in mm, Lorentz force in N, and current in
A. One duplicate export is removed so it cannot leak across data roles.

## Metrics

The requested primary threshold is `Fit_percent >= 95`. R2 is also reported,
but R2 and fit percentage are not the same quantity. When a reference is flat,
such as zero-input force, R2 and fit percentage are mathematically undefined;
that channel passes only when maximum absolute error is at most 0.001 N.

The Load-3 step force contains an isolated approximately 0.006 N discontinuity
near 0.152 s while current and displacement remain smooth. It is preserved in
the untouched data and remains visible in the error plot.

Optional runtime settings:

| Variable | Default | Meaning |
|---|---:|---|
| `DLSTM_ONE_STEP_EPOCHS` | 20 | Maximum one-step LSTM epochs |
| `DLSTM_ONE_STEP_SAMPLES_PER_EPOCH` | 20000 | Balanced training samples per epoch |
| `DLSTM_CPU_THREADS` | up to 8 | CPU threads used by PyTorch |
| `DLSTM_RESULTS_FOLDER` | `ResultsData` | Primary numerical results folder |
| `DLSTM_FIGURES_FOLDER` | `FiguresResults` | Primary figure folder |

## Trustworthy statistical evaluation

### The advisor's variance question

- Measured-signal variance describes how much the real output changes. R2 and
  fit use this variation as their reference.
- Error standard deviation describes how widely the sample-by-sample errors
  vary around their mean error (bias).
- Across-seed standard deviation describes how much the final score changes
  when training is repeated with different random initializations and sample
  orders.
- Across-experiment variation describes whether performance is consistent on
  chirp, step, zero-input, and different-load tests.

`R2 = 0.972 +/- 0.006` means the average R2 was 0.972 and its across-run
standard deviation was 0.006. A 95% confidence interval is an uncertainty
range for the unknown average performance under the same protocol. It does
not mean that 95% of individual predictions fall inside that interval.

One score cannot show repeatability, and repeated seeds alone cannot prove a
model is best. A best-model claim additionally needs identical test samples,
fair baselines, paired comparisons, untouched experiments, and uncertainty on
the improvement. The safe wording is “best among the evaluated models under
the defined protocol,” never “globally best.” For constant or nearly constant
signals, R2 and fit divide by almost no measured variation and are therefore
undefined or unstable; read RMSE, MAE, maximum error, and final drift instead.

The original command remains unchanged:

```powershell
python main.py
```

It still creates the original one-step figures and results. The additional
evaluation is intentionally separate:

```powershell
python trustworthy_evaluation.py --mode baseline
python trustworthy_evaluation.py --mode repeatability
python trustworthy_evaluation.py --mode ablation
python trustworthy_evaluation.py --mode all --full --run-autonomous
```

Equivalent Windows launchers are included:

```text
RUN_BASELINE_COMPARISON.bat
RUN_REPEATABILITY.bat
RUN_FEATURE_ABLATION.bat
RUN_TRUSTWORTHY_EVALUATION.bat
RUN_SMOKE_TEST.bat
```

The final reportable command is:

```powershell
python trustworthy_evaluation.py --mode all --full --run-autonomous
```

`--full` uses ten deterministic seeds. Normal repeatability and baseline modes
use five seeds. Seeds can be selected explicitly:

```powershell
python trustworthy_evaluation.py --mode repeatability --seeds 123 456 789 1011 1213
```

Before a long run, check all workflows quickly with:

```powershell
$env:DLSTM_SKIP_GITHUB_PUSH="1"
python trustworthy_evaluation.py --mode all --smoke-test
Remove-Item Env:DLSTM_SKIP_GITHUB_PUSH
```

Smoke-test outputs are written to
`TrustworthyEvaluation_SMOKE_NOT_REPORTABLE` and are explicitly labelled as
non-reportable. Never use their numbers as scientific evidence.

### What is compared

The active one-step LSTM has exactly 13 features:

1. `current`
2. `current_change`
3. `current_dc_estimate`
4. `mass_ratio`
5. `inverse_mass_ratio`
6. `elapsed_time`
7. `startup_indicator`
8. `previous_displacement`
9. `estimated_velocity`
10. `estimated_acceleration`
11. `previous_force`
12. `previous_force_change`
13. `constant_acceleration_displacement`

The LSTM predicts only the displacement correction. Lorentz force is produced
by the separate causal force/current rule in `one_step_lstm.py`; it is not an
LSTM output in the active `main.py` workflow.

Three causal methods use exactly the same samples:

- persistence: measured output at `k-1`;
- causal baseline: constant-acceleration displacement plus the causal force
  rule;
- hybrid LSTM: causal displacement baseline plus the LSTM correction, with the
  same causal force rule.

Consequently, hybrid and causal-baseline force predictions are intentionally
identical. Feature ablation and LSTM ranking are based on displacement.

### Leakage protection

The trustworthy evaluator uses stricter windows than the legacy report: the
complete 64-sample measured-feedback history must remain within one training,
validation, or internal-test block. It verifies and records that:

- normalization uses strict training samples only;
- early stopping uses validation only;
- pure records never enter training, validation, or normalization;
- compared methods use identical target samples;
- complete pure-test experiments remain untouched;
- no duplicate can occur on both sides of the development/pure-test boundary;
- feature selection uses validation only.

If an optional independent record duplicates development data, loading now
stops with a data-leakage error instead of reporting it as independent proof.

### Feature selection rule

Backward elimination trains each candidate after removing one currently
retained feature. A removal is accepted only when, relative to the complete
13-feature validation result:

- validation fit decreases by no more than 0.5 percentage points; and
- validation RMSE increases by no more than 5%.

Among acceptable candidates, the lowest validation RMSE is selected. This is
repeated until no candidate passes. Pure-test targets are evaluated once only
after selection is frozen. Change limits if required:

```powershell
python trustworthy_evaluation.py --mode ablation --validation-fit-drop 0.25 --validation-rmse-increase 3
```

### New output folders

Reportable results are kept separate from the original outputs:

```text
TrustworthyEvaluation/
  DataIntegrity/
    data_integrity_checks.csv
    split_fingerprint.txt
  Repeatability/
    Results/per_run_metrics.csv
    Results/mean_sd_ci95_summary.csv
    Results/per_pure_test_summary.csv
    Results/residual_diagnostic_table.csv
    Figures/worst_case_pure_test.png
    Figures/residual_diagnostics.png
  BaselineComparison/
    Results/model_ranking.csv
    Results/baseline_comparison.csv
    Results/paired_improvement_ci95.csv
    Figures/model_comparison.png
    Figures/performance_distribution_across_seeds.png
  FeatureAblation/
    Results/feature_ablation_table.csv
    Results/selected_features.txt
    Results/final_reduced_pure_test_metrics.csv
    Figures/feature_ablation_results.png
  Autonomous/
    Results/
    Figures/
  trustworthy_evaluation_report.md
```

Every metric summary reports mean, standard deviation, minimum, maximum, and a
95% confidence interval across seeds. Individual pure tests and the pooled
result are kept separate. Residual tables include bias, residual standard
deviation, lag-one autocorrelation, trend, final drift, and warning flags.

The paired table defines an improvement so positive values favor the hybrid.
The report calls an improvement clear only when the complete paired 95%
confidence interval is above zero. It never automatically declares the LSTM
best. The scientifically safe conclusion is only “best among the evaluated
models under the defined protocol.”

The autonomous program remains a distinct experiment. Its results are never
mixed with one-step measured-feedback metrics. The full command above runs it
last and saves it under `TrustworthyEvaluation/Autonomous`.

At the end of a successful trustworthy evaluation, results are committed and
pushed using the existing GitHub workflow. Disable pushing only for deliberate
local checks with `DLSTM_SKIP_GITHUB_PUSH=1`.
