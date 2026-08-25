"""
Central configuration for the actuator system identification study.

Everything that describes the experiment lives here: which COMSOL records
exist, which of them are allowed to be used for training, which ones are
locked away as pure test data, and every hyperparameter of the network.

The workbook Total_Data.xlsx stores each simulated case as a block of four
columns (Time, Displacement, Coil Current, Lorentz force). Some sheets hold
a single case in columns A:D, other sheets hold four cases side by side in
columns A:D, F:I, K:N and P:S. The record library below encodes that layout
explicitly so that no data is left behind.
"""

import os


# ----------------------------------------------------------------------
# Workbook layout
# ----------------------------------------------------------------------

# The development workbook is deliberately an exact filename.  A file such
# as ``Total_Data(1).xlsx`` or an old OneDrive copy can therefore never be
# selected silently.
DEVELOPMENT_WORKBOOK_NAME = "Total_Data.xlsx"

# Optional, completely external evaluation data.  When this exact file is
# beside main.py, every recognised record in it is appended as an untouched
# independent test.  When it is absent, the normal study runs unchanged.
INDEPENDENT_TEST_WORKBOOK_NAME = "Test_idpd.xlsx"

# Row index (zero based) of the row that carries the column titles.
HEADER_ROW = 16

# Mass of the moving coil assembly, shared by every simulated case.
COIL_MASS_GRAM = 1.427

# The load mass used as the normalisation reference for the configuration
# features. This is the mass that appears in most of the records.
REFERENCE_LOAD_MASS_GRAM = 3.813

REFERENCE_TOTAL_MASS_GRAM = COIL_MASS_GRAM + REFERENCE_LOAD_MASS_GRAM


# ----------------------------------------------------------------------
# Record discovery
# ----------------------------------------------------------------------
# There is deliberately no hardcoded table of sheet names here.
#
# The workbook is regenerated from COMSOL from time to time, and between two
# exports the sheets were reordered, one record was dropped and another was
# duplicated. Any pipeline keyed on "Sheet4" would then either crash or,
# far worse, silently train on the wrong load mass. Records are therefore
# discovered from the metadata block that each sheet already carries.
# See workbook_scan.py.


# ----------------------------------------------------------------------
# Pure test selection
# ----------------------------------------------------------------------
# The held out set is also chosen by rule rather than by name, for the same
# reason. Each rule below picks records out of whatever the workbook happens
# to contain.
#
#   heaviest_load        every record at the largest load mass present.
#                        That mass sits outside the range spanned by the
#                        others, so these records test extrapolation.
#   strongest_dc_chirp   the chirp at the reference mass with the highest DC
#                        offset, testing an unseen excitation amplitude.
#   reference_step       the step record at the reference mass.
#   reference_zero_input the zero current record at the reference mass.
#
# The last two are the extra validation signals. They sit at the reference
# mass on purpose: the same two families still appear in training at another
# mass, so they ask whether the model carries a signal type across to a
# different configuration rather than whether it has memorised it.

# Two ways to hold out the heaviest load, and they ask different questions.
#
#   "heaviest_load"
#       Holds out all four records at 7.625 g. Training then sees only 3.333 g
#       and 5.240 g total mass, and the test mass of 9.052 g is 73 percent
#       beyond that range. This measures extrapolation to an unseen mass. It is
#       the harder question and the honest answer to it is that two masses do
#       not determine a third: displacement error on this setting bottoms out
#       around 0.10 to 0.21 mm on the step and DC sine records no matter what
#       is done to the model.
#
#   "heaviest_load_unseen_excitations"
#       Keeps the sine and zero current records at 7.625 g in training and holds
#       out the step and DC sine records there. The mass is then represented, so
#       the physics has three points to fit and the network is no longer
#       extrapolating in its own input space. The test still uses excitations
#       the model has never seen at that mass.
#
# Switch with DLSTM_TEST_DESIGN=extrapolation to get the first one back.
_DESIGN = os.environ.get("DLSTM_TEST_DESIGN", "unseen_excitations")

PURE_TEST_RULES = (
    "heaviest_load" if _DESIGN == "extrapolation"
    else "heaviest_load_unseen_excitations",
    "strongest_dc_chirp",
    "reference_step",
    "reference_zero_input",
)

# Optional single-record holdout used by leave-one-record-out evaluation.
# The named record is still excluded from every fit, statistic and gradient;
# the override only prevents several related records from being removed at
# the same time.  This is the recommended way to estimate how a final model
# trained on the complete measured library will generalise.
PURE_TEST_RECORD_OVERRIDE = os.environ.get(
    "DLSTM_PURE_TEST_RECORD", ""
).strip()

EXTRA_STEP_TEST_RULE = "reference_step"
EXTRA_ZERO_INPUT_TEST_RULE = "reference_zero_input"

# A record whose data is byte for byte identical to another is dropped before
# training. A duplicate doubles that record's weight, and if one copy lands in
# training while the other lands in a test split it turns the test into a
# memorisation check that looks excellent for entirely the wrong reason.
DROP_DUPLICATE_RECORDS = True


# ----------------------------------------------------------------------
# Resampling and windowing
# ----------------------------------------------------------------------
# The COMSOL exports do not share a single sample rate. The chirp sheets are
# stored at 2000 Hz, the heavy load sheet at 1000 Hz and one sheet at 500 Hz.
# Every record is therefore placed on one common uniform grid before any
# feature is computed.

TARGET_SAMPLE_RATE_HZ = 1000.0

TARGET_TIME_STEP = 1.0 / TARGET_SAMPLE_RATE_HZ

# Length of the input window fed to the network, in samples.
# 200 samples at 1000 Hz is 0.2 s, which covers more than one full period of
# the slowest resonance observed in the data (about 6 Hz).
SEQUENCE_LENGTH = 200

# Development records are cut into contiguous blocks and each block is given
# one role. Windows never cross a block boundary, so no training sample can
# leak into a validation or internal test sample.
BLOCK_SECONDS = 0.20

BLOCK_SIZE = int(round(BLOCK_SECONDS * TARGET_SAMPLE_RATE_HZ))

# Every development record of at least 0.8 s now contributes to training,
# validation and internal test.  This is important here: it prevents the
# impressive internal score from being a chirp-only result while all short
# step/sine/zero records quietly remain training-only.
SPLIT_PATTERN = (
    "training",
    "validation",
    "training",
    "test",
)

TRAIN_STRIDE = 1
VALIDATION_STRIDE = 2
TEST_STRIDE = 1

# Stride used when a complete pure test record is swept end to end.
PURE_TEST_STRIDE = 1


# ----------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------
# Ten input channels are used. The first five carry the excitation, the next
# three identify the physical configuration and the last two describe where
# in the record the current window sits.
#
#   0  coil current
#   1  first difference of coil current
#   2  causal low pass estimate of the DC level of the current
#   3  coil current divided by the mass ratio
#   4  DC estimate divided by the mass ratio
#   5  mass ratio            (total mass / reference total mass)
#   6  inverse mass ratio
#   7  natural frequency ratio = sqrt(inverse mass ratio)
#   8  elapsed time expressed in reference mass units, clipped
#   9  startup transient indicator, exp(-t / tau)
#  10  displacement predicted by the physical model, mm
#  11  velocity predicted by the physical model, mm/s
#  12  actuator force predicted by the physical model, N

FEATURE_NAMES = [
    "current",
    "current_rate",
    "current_dc_estimate",
    "current_over_mass_ratio",
    "current_dc_over_mass_ratio",
    "mass_ratio",
    "inverse_mass_ratio",
    "frequency_ratio",
    "scaled_elapsed_time",
    "startup_indicator",
    # State of the physical model, which is what turns this from "predict the
    # response" into the far easier "correct a simulation that is already
    # close". The velocity channel matters most: it is the only input that
    # carries the phase of the ring down.
    "model_displacement",
    "model_velocity",
    "model_force",
]

INPUT_FEATURES = len(FEATURE_NAMES)

# Channels that describe the physical configuration. They are constant along
# a record and are used to drive the FiLM conditioning layers.
CONFIGURATION_FEATURE_INDICES = (5, 6, 7)

CONFIGURATION_FEATURES = len(CONFIGURATION_FEATURE_INDICES)

# Time constant of the causal DC estimator, in seconds. It has to be short
# enough that the 0.8 s transient records reach their operating point well
# before they end, and long enough to average out the excitation ripple.
DC_ESTIMATOR_TIME_CONSTANT = 0.10

# Time constant of the startup indicator, in seconds.
STARTUP_TIME_CONSTANT = 0.30

# The scaled elapsed time feature saturates after this many seconds so that
# the 20 s record does not dominate the numerical range.
SCALED_TIME_CLIP_SECONDS = 3.0


# ----------------------------------------------------------------------
# Quasi static physical baseline
# ----------------------------------------------------------------------
# A least squares baseline is fitted on training samples only. The network
# then has to predict the residual that the baseline cannot explain, which
# removes the large mass dependent offset from the learning problem.
#
# Set DLSTM_STATIC_BASELINE=0 to disable it and train on raw targets.

USE_STATIC_BASELINE = os.environ.get("DLSTM_STATIC_BASELINE", "1") != "0"

# ----------------------------------------------------------------------
# Grey box physical baseline
# ----------------------------------------------------------------------
# The network predicts the residual left over by a simulated mass spring
# damper model of the actuator, not the response itself. See grey_box.py for
# why: the free ring down after a gravity release carries no trace in the coil
# current, so no amount of network capacity can recover it from the input,
# whereas Newton's second law reproduces it exactly.
#
# Set DLSTM_GREY_BOX=0 to fall back to the old algebraic quasi static
# baseline, which is useful for showing what the physical model contributes.
USE_GREY_BOX = os.environ.get("DLSTM_GREY_BOX", "1") != "0"

# Refitting the grey box against simulated trajectories instead of by equation
# error is DELIBERATELY OFF.
#
# It looks like an obvious win and it is not. Equation error differentiates the
# measured displacement twice and is a biased estimator, so refitting against
# simulated trajectories should be strictly better. Measured on this model it
# cuts the training cost from 0.222 to 0.101 and improves every reference mass
# test record. It also destroys the heaviest load: 0.037 mm to 0.351 mm on the
# zero input record, 0.050 mm to 0.406 mm on the sine record.
#
# Leave one mass out settles it. Fitting without the 1.906 g records and then
# predicting them gives:
#
#     equation error          transfer RMSE 0.0242 mm
#     refined, no penalty     transfer RMSE 0.0813 mm
#     refined, lambda 0.2     transfer RMSE 0.0741 mm
#     refined, lambda 0.5     transfer RMSE 0.0662 mm
#
# Every refined variant transfers worse across mass, penalty or not. With a
# nine degree spring and only two training masses, the extra freedom is spent
# contorting the curve where the training records happen to sit. The noise in
# the equation error fit acts as regularisation that keeps the polynomial
# tame, which is exactly what extrapolating to a third mass needs.
#
# Turn this on only after confirming transfer with evaluate_unseen_records.py.
REFINE_PHYSICAL_MODEL = os.environ.get("DLSTM_REFINE", "0") != "0"

REFINEMENT_EVALUATIONS = int(os.environ.get("DLSTM_REFINE_EVALS", "200"))

# Long records are subsampled during refinement. The parameters are physical
# constants, so a representative stretch identifies them as well as 20 s does.
REFINEMENT_SAMPLES_PER_RECORD = int(
    os.environ.get("DLSTM_REFINE_SAMPLES", "6000")
)

# Signals whose peak to peak value falls below these thresholds carry no
# information, so a coefficient of determination computed against them is
# meaningless. The zero current records are exactly this case for the force
# channel. R2 and fit percentage are reported as not available there and the
# absolute errors are used instead.
NEGLIGIBLE_SIGNAL_RANGE = {
    "Displacement": 2e-3,   # mm; below this a settled block is effectively flat
    "Lorentz force": 2e-3,  # N
}


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

HIDDEN_1 = 32
HIDDEN_2 = 64
HIDDEN_3 = 64

HEAD_HIDDEN = 32

FILM_HIDDEN = 32

OUTPUTS = 2

DROPOUT = 0.10


# ----------------------------------------------------------------------
# Optimisation
# ----------------------------------------------------------------------

BATCH_SIZE = 256

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-6

EPOCHS = int(os.environ.get("DLSTM_EPOCHS", "30"))

EARLY_STOPPING_PATIENCE = 10

SCHEDULER_PATIENCE = 4

SCHEDULER_FACTOR = 0.5

MIN_LEARNING_RATE = 1e-5

MINIMUM_IMPROVEMENT = 1e-6

GRADIENT_CLIP_NORM = 1.0

DISPLACEMENT_LOSS_WEIGHT = 0.50
FORCE_LOSS_WEIGHT = 0.50

DISPLACEMENT_PEAK_WEIGHT = 0.35

# Long records hold far more samples than the short transient records. Every
# training window is drawn with a probability proportional to
# (1 / number of windows in its record) ** BALANCE_POWER.
#   0.0 gives the raw, unbalanced distribution
#   1.0 gives every record exactly the same expected share
BALANCE_POWER = float(os.environ.get("DLSTM_BALANCE_POWER", "0.7"))

# Number of training windows drawn per epoch by the balanced sampler.
SAMPLES_PER_EPOCH = int(os.environ.get("DLSTM_SAMPLES_PER_EPOCH", "40000"))

# Off by default. The validation blocks are needed as a clean surface on
# which to fit the residual trust factors, so they are kept out of every
# gradient update.
FINAL_FINE_TUNE_EPOCHS = int(os.environ.get("DLSTM_FINE_TUNE_EPOCHS", "0"))

FINAL_FINE_TUNE_LEARNING_RATE = 1e-4

SEED = 123

CPU_THREADS = int(os.environ.get(
    "DLSTM_CPU_THREADS", str(min(8, os.cpu_count() or 4))
))


# ----------------------------------------------------------------------
# Synthetic probe signals
# ----------------------------------------------------------------------
# After the model is frozen it is also driven with two signals that were
# never simulated in COMSOL. There is no ground truth for these, they exist
# purely as a physical sanity check of the identified model.

SYNTHETIC_PROBE_SECONDS = 0.8

SYNTHETIC_STEP_AMPLITUDE = 0.15

SYNTHETIC_STEP_TIME = TARGET_TIME_STEP

# The requested generated checks are only for the reference Load-2 case.
SYNTHETIC_PROBE_LOAD_MASSES = (REFERENCE_LOAD_MASS_GRAM,)

# An exact physical boundary used after the learned residual correction.
ZERO_CURRENT_TOLERANCE_A = 1e-7
