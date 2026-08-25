# Voice-Coil One-Step LSTM Identification

## Excel files beside main.py

The development data filename is now exact:

```text
Total_Data.xlsx
```

Only that file supplies development, training, validation, internal-test, and
the original built-in pure-test records. Files such as `Total_Data(1).xlsx`
are deliberately ignored.

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

Optional runtime settings:

| Variable | Default | Meaning |
|---|---:|---|
| `DLSTM_ONE_STEP_EPOCHS` | 20 | Maximum one-step LSTM epochs |
| `DLSTM_ONE_STEP_SAMPLES_PER_EPOCH` | 20000 | Balanced training samples per epoch |
| `DLSTM_CPU_THREADS` | up to 8 | CPU threads used by PyTorch |
| `DLSTM_RESULTS_FOLDER` | `ResultsData` | Primary numerical results folder |
| `DLSTM_FIGURES_FOLDER` | `FiguresResults` | Primary figure folder |
