# Actuator system identification, full data set

Physics informed LSTM identification of a voice coil actuator simulated in
COMSOL Multiphysics. The model takes the coil current and the physical
configuration and predicts displacement and Lorentz force.

## How to run

```
python main.py
```

Everything is written to `ResultsData` and `FiguresResults`. The workbook
`Total_Data.xlsx` must sit next to `main.py`.

Environment switches, all optional:

| variable | default | meaning |
|---|---|---|
| `DLSTM_EPOCHS` | 30 | maximum training epochs |
| `DLSTM_SAMPLES_PER_EPOCH` | 40000 | training windows drawn per epoch |
| `DLSTM_BALANCE_POWER` | 0.7 | 0 keeps the raw record sizes, 1 gives every record an equal share |
| `DLSTM_STATIC_BASELINE` | 1 | set to 0 to train on raw targets instead of residuals |
| `DLSTM_FINE_TUNE_EPOCHS` | 3 | final refresh on training plus validation |

Requirements: `numpy`, `pandas`, `scipy`, `torch`, `matplotlib`, `openpyxl`.

### Where the workbook has to be

`Total_Data.xlsx` is **not shipped with the code**, so that a stale copy can
never silently override the one you meant to use. Copy your own into the
folder that holds `main.py`.

The loader also looks one and two folders above `main.py`, which covers an
archive that unpacked with an extra wrapper folder around the code, and the
folder Python was launched from. Excel lock files (`~$Total_Data.xlsx`) are
ignored. If several workbooks match, the most recently modified one is used
and the choice is printed.

To point at a specific file instead:

```
set DLSTM_WORKBOOK=C:\full\path\to\Total_Data.xlsx
python main.py
```

If the run stops with a message about a sheet having fewer columns than
expected, the workbook layout changed in a way the scanner did not recognise.
The error names the sheet and the columns it wanted.

## What is in the data

Records are **discovered from the workbook**, not from a hardcoded table of
sheet names. This matters: two exports of `Total_Data.xlsx` were compared and
the sheets had been reordered, one record had been dropped and another
duplicated. A pipeline keyed on "Sheet4" would either crash or, far worse,
silently train on the wrong load mass.

Every sheet already describes itself in the rows above the data:

```
row 0    Coil Mass=1.427 gram, Load Mass=3.813 gram
row 15   Case 1: Coil Current= DC_Offset+Sine. DC_Offset=0.12A, ...
row 16   Time (s) | Displacement(mm) | Coil Current (A) | Lorentz force (N)
row 17+  numeric data
```

`workbook_scan.py` reads that block, finds a case wherever the title row says
`Time`, parses the load mass and the excitation description, and builds a name
from the physics: `Load2_DCChirp_147mA_31Hz_7s`. Those names are stable across
exports. Sheets carrying four cases side by side in columns A:D, F:I, K:N and
P:S are picked up automatically, which a reader restricted to columns A:D
misses entirely.

Export sample rates differ between sheets (2000, 1000 and 500 Hz have all been
seen). Every record is interpolated onto one common 1000 Hz grid.

**Duplicate detection.** Records with byte identical data are reported and, by
default, all but the first copy is dropped. A duplicate doubles that record's
weight in training, and if one copy lands in training while the other lands in
a test split it turns the test into a memorisation check that looks excellent
for the wrong reason. Set `DROP_DUPLICATE_RECORDS = False` in `config.py` to
keep them.

## Test policy

The held out set is chosen **by rule**, not by name, so it stays physically the
same set when the workbook is regenerated. The rules are in
`config.PURE_TEST_RULES`:

| rule | picks | why |
|---|---|---|
| `heaviest_load` | every record at the largest load mass | that mass sits outside the range spanned by the others, so this tests extrapolation |
| `strongest_dc_chirp` | the reference mass chirp with the highest DC offset | unseen excitation amplitude |
| `reference_step` | the step record at the reference mass | extra validation signal 1 |
| `reference_zero_input` | the zero current record at the reference mass | extra validation signal 2 |

Nothing in that set is read during training, validation, normalisation or the
baseline fit. It is touched once, after the weights are frozen.

The two extra signals sit at the reference mass on purpose. The step and zero
input families still appear in training at a different load mass, so these two
records ask whether the model can carry a signal type across to another
configuration rather than whether it has memorised it.

Development records are cut into contiguous blocks that rotate through
training, training, validation, training, training, internal test. A target
sample never crosses a block boundary. The input history of a target may reach
back into the preceding block, which is deliberate: past inputs are always
available at inference time, so withholding them would measure something the
deployed model never faces.

## Method

**Rest state padding.** A window of 200 samples is needed before the first
prediction. Each record is therefore padded backwards with the state it
started from, which is exact because every COMSOL case begins at rest. Without
this the first 0.2 s would be unavailable, and for the 0.8 s transient records
that is the entire step response.

**Ten input channels.** Coil current, its first difference, a causal low pass
estimate of its DC level, two mass scaled current channels, three
configuration channels (mass ratio, inverse mass ratio, natural frequency
ratio) and two startup channels.

The configuration channels are what make the zero input records solvable at
all. Their coil current is identically zero at every mass, yet displacement
differs by 3.4 mm between the light and the heavy load, so a network fed only
current derived channels cannot in principle tell them apart.

**Quasi static baseline.** At equilibrium the actuator obeys
`k x = Bl i - m g`, so the resting position is linear in current and linear in
mass. A weighted least squares fit on training samples only gives

```
x_static = a0 * I_dc + a1 * r + a2        r = total mass / reference mass
F_static = b0 * I  + b1
```

and the network predicts only what this misses. Two details matter here:

- An interaction term `I_dc * r` fits the training records just as well but
  extrapolates to a **negative actuator gain** at the heaviest load, which is
  physically impossible. Because the two training masses happen to use
  different DC levels, current and mass are correlated and the interaction is
  not identifiable. It is excluded on purpose.
- The fit is weighted so every record counts equally. Unweighted, the 20 s
  chirp contributes 20001 samples against 801 for a step response and sets the
  baseline on its own.

**Network.** Three LSTM layers of 32, 64 and 64 units, each followed by a FiLM
layer driven by the configuration channels, then separate displacement and
force heads, plus a zero initialised linear bypass. About 79 thousand
trainable parameters.

**Balanced sampling.** Training windows are drawn with probability
proportional to `(1 / windows in that record) ** BALANCE_POWER`, otherwise the
long chirps bury the transient records.

## Reading the results

`ResultsData/metrics.csv` and figure 13 carry the full table.

One caution. R2 is a variance ratio, so it is only meaningful when the
reference signal actually varies. In the zero input records the Lorentz force
reference is flat at roughly zero, with a peak to peak of about 1e-5 N, which
is COMSOL numerical noise. Any prediction at all produces an absurd R2 there.
The code detects this, reports R2 as not available, and you should read RMSE
and the maximum absolute error instead. The same caution applies more mildly
wherever the evaluated span of a signal is nearly constant.

## If the figures fail to save on Windows

`FiguresResults` is emptied file by file rather than deleted, because Windows
refuses to remove a folder that anything holds open, and OneDrive holds folders
open while it syncs. If an individual figure is locked by an image viewer it is
written alongside as `name__new.png`, and if that also fails the run reports it
and carries on.

A plotting failure never costs you the training. Everything needed to redraw is
in `ResultsData`, so you can close whatever holds the lock and rerun just:

```
python plot_results.py
```

Keeping the project outside a synced OneDrive folder avoids the problem
entirely.

## Figures

| file | question it answers |
|---|---|
| `01_record_inventory` | what is in the workbook |
| `02_data_split_map` | which samples were used for what |
| `03_learning_curves` | did training converge |
| `04` / `05_internal_test_*` | does the model reproduce held out blocks |
| `06_*_pure_test_*` | one detailed page per untouched record |
| `07_parity_*` | measured against predicted, pure tests |
| `08_metric_summary` | accuracy across every evaluation set |
| `09_step_input_validation` | extra validation signal 1 |
| `10_zero_input_validation` | extra validation signal 2 |
| `11_untouched_chirp_detail` | time and frequency detail on the 147 mA chirp |
| `12_synthetic_probe_response` | physical sanity check on signals never simulated |
| `13_metric_table` | printable table |

Figure 12 has no reference data. It drives the frozen model with an ideal step
and an exact zero input at all three masses and asks only whether the response
is physically plausible: force should follow Bl times current, and
displacement should settle on the gravity sag of each mass.

## Known limitation

The 7.625 g load sits outside the 1.906 to 3.813 g training range, so its
displacement is genuine extrapolation rather than interpolation. The
configuration channels let the model represent the difference, but nothing in
the architecture can substitute for training coverage at that mass. If the
Load_3 records matter, the fix is a COMSOL run at an intermediate mass, around
5.5 g, which would turn the problem into interpolation.

Note also that validation blocks come only from the longer records. The 0.8 s
records are short enough that the block pattern assigns them entirely to
training, so early stopping is driven by chirp performance.
