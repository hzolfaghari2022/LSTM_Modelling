# Voice-Coil Physics-Informed LSTM Identification

Run `main.py`. It discovers every record in `Total_Data.xlsx`, removes one
exact duplicate, trains on development data, restores the validation-selected
checkpoint, and evaluates untouched records. Numerical predictions are saved
in `ResultsData`; measured-versus-predicted figures are saved in
`FiguresResults`.

## What the model is

This is a transparent hybrid model, not a pure autoregressive LSTM:

1. A nonlinear mass-spring-damper model simulates displacement, velocity, and
   electromagnetic force from current, mass, and the initial state.
2. A configuration-conditioned LSTM predicts the remaining model error from a
   causal 200-sample input window.
3. A validation-fitted trust factor scales the learned correction.
4. Conservative family-support guards fall back to physics for unsupported
   extrapolations.
5. Zero coil current is projected onto exactly zero Lorentz force.

Figures therefore label the result **grey-box + LSTM residual** and separately
show **grey-box physics only**. The LSTM does not feed its previous output back
into the next sample, so these results must not be called a pure LSTM
free-running rollout.

## Initial conditions and generated checks

Every measured and generated simulation is initialized only from measured
initial displacement `x0`, finite-difference velocity `velocity0`, and initial
force `force0`. No measured-output warm-up trajectory is supplied.

Only the requested Load-2 generated checks are created:

- Exact zero current for 0.8 s.
- An ideal 150 mA step beginning at the second sample for 0.8 s.

The generated signals have no independent ground truth. They are sanity
checks for boundedness, settling, and the exact zero-current force condition.

## Data policy

The workbook contains 19 exports. One pair is byte-identical, so one copy is
removed and 18 unique records remain. The default pure-test design keeps five
whole records out of physical-model fitting, normalization, LSTM training,
validation, and residual-trust calibration:

- Load-2 147 mA chirp.
- Load-2 150 mA step.
- Load-2 zero input.
- Load-3 200 mA step.
- Load-3 200 mA DC+sine.

The remaining 13 records are development records. Their 0.4 s blocks rotate
through training, validation, and internal testing. Tiny final remainders are
merged into the preceding block so every unique real sample has a role.
Because the short 0.8 s records mostly contribute training samples, the whole
pure tests—not the internal chirp blocks—are the primary generalization check.

The default design tests unseen excitations at the heavy mass: Load-3 sine and
zero-input records remain in development while Load-3 step and DC+sine remain
untouched. Set `DLSTM_TEST_DESIGN=extrapolation` to hold out every heavy-load
record instead.

### Recommended unseen-record audit

The default five-record holdout is intentionally severe: it removes both
Load-2 and Load-3 step records at the same time, so the training pool contains
only the lightest step. For a more realistic estimate of a production model
that will use the complete measured library, run:

```bash
python evaluate_unseen_records.py
```

This trains one fresh fold per requested pure record. In each fold the target
record is absent from the physical fit, normalization, gradient updates,
validation, and trust calibration; the other records remain available. Fold
models, metrics, and measured-versus-predicted figures are written under
`UnseenRecordResults`. This is valid leave-one-record-out evaluation, not
training-set reconstruction.

The supplied workbook is still too sparse to guarantee negligible error for
arbitrary inputs. In particular, only a few deterministic waveforms and one
or two amplitudes exist per load. The code reports the resulting error rather
than tuning on pure-test targets. To reduce the remaining Load-3 step/DC+sine
error, add development measurements at that load using several amplitudes and
a persistently exciting input such as PRBS or multisine, while keeping new
records untouched for final testing.

## Important physical constraints

For Lorentz force,

```text
F = Bl(x) * current
```

so the reported force is exactly zero wherever current is zero. An
unconstrained additive neural correction is never allowed to violate this
identity. Zero current does **not** imply zero displacement: gravity, spring
preload, mass, and the selected initial state still create a damped mechanical
response.

The physical model is identified only from development training samples. The
bundled parameters are listed in `ResultsData/physical_model_parameters.json`.
A fresh run also stores them in `ResultsData/model.pt` together with the neural
weights, normalization, feature names, trust factors, and initialization
definition.

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

`RUN_ME.bat` performs the same steps on Windows. Optional settings include:

| Variable | Default | Meaning |
|---|---:|---|
| `DLSTM_EPOCHS` | 30 | Maximum training epochs |
| `DLSTM_SAMPLES_PER_EPOCH` | 40000 | Balanced windows drawn per epoch |
| `DLSTM_BALANCE_POWER` | 0.7 | Record-balancing strength |
| `DLSTM_FINE_TUNE_EPOCHS` | 0 | Disabled so validation remains clean |
| `DLSTM_TEST_DESIGN` | `unseen_excitations` | Pure-test design |
| `DLSTM_PURE_TEST_RECORD` | empty | Hold out exactly one named record |
| `DLSTM_CPU_THREADS` | up to 8 | CPU threads used by PyTorch |
| `DLSTM_RESULTS_FOLDER` | `ResultsData` | Numerical output folder |
| `DLSTM_FIGURES_FOLDER` | `FiguresResults` | Figure output folder |

## Reading the metrics

- RMSE and MAE are the main absolute-error measures.
- R2 is not meaningful for a nearly constant reference such as zero-current
  force; it is reported as unavailable there.
- `RMSE_physics_only` shows whether the LSTM correction actually improves the
  physical simulator.
- Synthetic-probe plots have no reference and therefore no accuracy claim.

The strongest evidence is the collection of complete pure-test time histories,
especially the Load-2 zero/step/chirp records and the Load-3 step/DC+sine
records. Internal held-out blocks are useful but easier because they come from
development trajectories already represented during training.
