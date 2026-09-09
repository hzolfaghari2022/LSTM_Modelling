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
- Figure 19: the implemented one-input hybrid LSTM architecture. It distinguishes
  the single applied coil-current input from the 13 internally constructed
  causal features and distinguishes the learned displacement correction from
  the separate causal force rule.

All generated figure text uses Times New Roman when it is installed. Garamond,
Times, and compatible serif fonts are used as fallbacks.

- `ResultsData/one_step_metrics.csv` contains per-record training, validation,
  internal-test, and pure-test results;
- `ResultsData/one_step_pooled_role_metrics.csv` contains pooled results for
  all four roles;
- `ResultsData/one_step_predictions/*.csv`
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

## Feature-ablation study: 13 features down to 1

The primary one-step displacement LSTM uses these 13 causal features:

1. current;
2. current change;
3. causal current DC estimate;
4. mass ratio;
5. inverse mass ratio;
6. elapsed time;
7. startup indicator;
8. previous displacement;
9. previous velocity;
10. previous acceleration;
11. previous force;
12. force change; and
13. the quadratic displacement baseline.

These are model features, not 13 independently applied actuator inputs. The
only externally applied actuator input is coil current. In this V15 one-step
implementation, the LSTM predicts the displacement residual. Lorentz force is
calculated by a separate causal force rule, so changing the LSTM feature set
does not change that rule.

Run the short software check first:

```bash
python feature_ablation.py --mode quick
```

Quick-mode output is explicitly labelled **NOT REPORTABLE**. To run the complete
13-to-1 study for presentation to an advisor, use:

```bash
python feature_ablation.py --mode full
```

`RUN_FEATURE_ABLATION_QUICK.bat` and `RUN_FEATURE_ABLATION_FULL.bat` provide the
same commands on Windows. The full study is computationally expensive: greedy
backward elimination trains the complete model and every possible one-feature
removal at every step (91 training jobs in total).

At each step, the feature producing the lowest development-validation
displacement RMSE is removed. Pure-test values are recorded only after that
decision and never select a feature. The default validation acceptance limits
are no more than a 0.5 percentage-point loss of fit and no more than a 5 percent
increase in RMSE relative to the 13-feature model.

Full results are written to `FeatureAblationResults` and figures to
`FeatureAblationFigures`. Each retained-feature count has its own `F13` through
`F01` folder containing its checkpoint, metrics, training history, metadata,
and pure-test predictions. The main summary files are:

- `ablation_summary.csv`;
- `candidate_removal_trials.csv`;
- `all_selected_case_metrics.csv`;
- `per_pure_test_metrics.csv`; and
- `README_RESULTS.txt`.

The comparison figures show accuracy versus feature count, the feature-removal
path, and measured-versus-predicted responses for the full, recommended, and
one-feature models. The reported recommended feature count is selected only
from validation performance. Do not choose a different count after inspecting
pure-test performance.

After a successful quick or full ablation run, the same verified GitHub-push
workflow is called automatically. Set `DLSTM_SKIP_GITHUB_PUSH=1` only when you
deliberately want a local run.

Optional runtime settings:

| Variable | Default | Meaning |
|---|---:|---|
| `DLSTM_ONE_STEP_EPOCHS` | 20 | Maximum one-step LSTM epochs |
| `DLSTM_ONE_STEP_SAMPLES_PER_EPOCH` | 20000 | Balanced training samples per epoch |
| `DLSTM_CPU_THREADS` | up to 8 | CPU threads used by PyTorch |
| `DLSTM_RESULTS_FOLDER` | `ResultsData` | Primary numerical results folder |
| `DLSTM_FIGURES_FOLDER` | `FiguresResults` | Primary figure folder |
