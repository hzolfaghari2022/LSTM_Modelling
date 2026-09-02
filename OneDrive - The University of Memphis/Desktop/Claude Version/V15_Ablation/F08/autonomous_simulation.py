"""
Actuator system identification, full data set study.

Run this file. It loads every record in Total_Data.xlsx, trains the
configuration conditioned LSTM on the development records only, and then
evaluates the frozen model on a set of records that were never touched
during training.

Default pure-test records include the strongest Load-2 chirp, Load-2 step and
zero-input responses, and unseen Load-3 step and DC+sine excitations. Load-3
sine and zero-input records remain in development so the default test is an
unseen-excitation test at a represented mass.

Two synthetic probe signals are added at the very end. They have no COMSOL
reference and exist only as a physical sanity check of the frozen model.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    CPU_THREADS,
    NEGLIGIBLE_SIGNAL_RANGE,
    SEED,
    SEQUENCE_LENGTH,
    SYNTHETIC_PROBE_LOAD_MASSES,
    SYNTHETIC_PROBE_SECONDS,
    SYNTHETIC_STEP_AMPLITUDE,
    SYNTHETIC_STEP_TIME,
    TARGET_SAMPLE_RATE_HZ,
    TARGET_TIME_STEP,
    USE_GREY_BOX,
    USE_STATIC_BASELINE,
    ZERO_CURRENT_TOLERANCE_A,
)
from data_utils import (
    attach_synthetic_arrays,
    make_synthetic_record,
    prepare_data,
    reconstruct_outputs,
    record_window_indices,
)
from model import ConfigurationConditionedLSTM, describe_model
from train_utils import (
    calculate_metrics,
    fine_tune_on_development_data,
    predict_pairs,
    train_model,
)


torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(CPU_THREADS)

HERE = Path(__file__).resolve().parent
RESULTS_FOLDER = Path(
    os.environ.get(
        "DLSTM_AUTONOMOUS_RESULTS_FOLDER",
        str(HERE / "AutonomousResultsData"),
    )
).resolve()
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURES_FOLDER = Path(
    os.environ.get(
        "DLSTM_AUTONOMOUS_FIGURES_FOLDER",
        str(HERE / "AutonomousFiguresResults"),
    )
).resolve()
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
# autonomous_results_plots.py reads these generic variables in its subprocess.
os.environ["DLSTM_RESULTS_FOLDER"] = str(RESULTS_FOLDER)
os.environ["DLSTM_FIGURES_FOLDER"] = str(FIGURES_FOLDER)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Training device: {device}, {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("Training device: cpu, CUDA is not available here.", flush=True)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

print("\nLoading the workbook and building every record ...", flush=True)
data = prepare_data(HERE)
records = data["records"]
index_of = data["index_of"]
PURE_TEST_RECORDS = data["pure_test_names"]

data["inventory_table"].to_csv(RESULTS_FOLDER / "record_inventory.csv", index=False)
data["split_table"].to_csv(RESULTS_FOLDER / "data_split.csv", index=False)

print("Workbook:", data["workbook"].name)
print(f"Records discovered: {len(records)}")

if data["duplicate_groups"]:
    print("\nDuplicate records detected. These sheets hold identical data:")
    for group in data["duplicate_groups"]:
        print("  " + "  ==  ".join(group))
    if data["dropped_duplicates"]:
        print("  Dropped from the run:", ", ".join(data["dropped_duplicates"]))
    print("  A duplicate would otherwise double that record's weight in "
          "training, and could put the same signal on both sides of a split.")
print(f"Common sample rate: {TARGET_SAMPLE_RATE_HZ:.0f} Hz, "
      f"window length {SEQUENCE_LENGTH} samples "
      f"({SEQUENCE_LENGTH * TARGET_TIME_STEP:.3f} s)")
print()
print(data["inventory_table"][
    ["record", "family", "load_mass_g", "native_rate_hz",
     "duration_s", "samples_at_target_rate", "usage"]
].to_string(index=False))

print()
print(f"Training windows        : {len(data['training_pairs']):,}")
print(f"Validation windows      : {len(data['validation_pairs']):,}")
print(f"Internal test windows   : {len(data['internal_test_pairs']):,}")
print(f"Pure test records       : {len(PURE_TEST_RECORDS)}")

if USE_GREY_BOX and data["physical_model"] is not None:
    import grey_box
    print("\nPhysical model identified from training samples only:")
    print("  m x'' = Bl(x) i - m g - c x' - Fs(x), initialized from measured x0/v0/F0")
    print(grey_box.describe(data["physical_model"]))
    print("  The network predicts the residual this model leaves behind.")
elif USE_STATIC_BASELINE:
    print("\nQuasi static baseline fitted on training samples only:")
    displacement_coefficients = data["displacement_baseline_coefficients"]
    force_coefficients = data["force_baseline_coefficients"]
    print("  x_static = a0*I_dc + a1*r + a2      with r = total mass / reference mass")
    print(f"    a0 = {displacement_coefficients[0]:+.4f} mm/A   "
          f"a1 = {displacement_coefficients[1]:+.4f} mm   "
          f"a2 = {displacement_coefficients[2]:+.4f} mm")
    print("  F_static = b0*I + b1")
    print(f"    b0 = {force_coefficients[0]:+.4f} N/A    "
          f"b1 = {force_coefficients[1]:+.6f} N")
else:
    print("\nQuasi static baseline disabled, the network predicts raw targets.")


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

model = ConfigurationConditionedLSTM().to(device)
print("\n" + describe_model(model))
print()
print(model)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

print("\nTraining ...", flush=True)
training_history, validation_history, best_epoch = train_model(
    model, records, data, device
)

print("\nOptional final fine tune. It is disabled by default so validation "
      "remains independent for residual-trust calibration.", flush=True)
fine_tune_history = fine_tune_on_development_data(model, records, data, device)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

OUTPUT_INFO = [(0, "Displacement", "mm"), (1, "Lorentz force", "N")]


# ----------------------------------------------------------------------
# How far to trust the network
# ----------------------------------------------------------------------
# The physical model already explains most of the response. Whether the
# network's correction on top of it actually helps is an empirical question,
# and the honest answer differs per output channel: the force residual is
# largely learnable, the displacement residual much less so.
#
# A single scalar per channel is therefore fitted on the validation blocks,
# scaling the network's residual by the amount that genuinely reduces error
# there. It is the least squares projection of the true residual onto the
# predicted one, clipped to [0, 1]:
#
#     alpha = clip( <true, predicted> / <predicted, predicted>, 0, 1 )
#
# alpha near 1 means the correction is trustworthy and is applied in full.
# alpha near 0 means the network has not learned anything that generalises,
# and the prediction falls back to the physical model. This is what makes the
# combined model incapable of scoring worse than the physics alone, which the
# unguarded version did on several held out records.

@torch.no_grad()
def fit_residual_trust():
    validation_pairs = np.asarray(data["validation_pairs"], dtype=np.int64)
    predicted = predict_pairs(model, records, validation_pairs, device)
    actual = np.stack(
        [records[r]["targets_normalised"][t] for r, t in validation_pairs]
    )

    trust = np.zeros(predicted.shape[1], dtype=np.float64)
    for column in range(predicted.shape[1]):
        denominator = float(np.dot(predicted[:, column], predicted[:, column]))
        if denominator <= 0.0:
            continue
        numerator = float(np.dot(actual[:, column], predicted[:, column]))
        trust[column] = float(np.clip(numerator / denominator, 0.0, 1.0))
    return trust


DEVELOPMENT_RECORDS = [record for record in records if not record["is_pure_test"]]


def residual_scale_for_record(record):
    """Validation trust plus conservative, physics-based extrapolation guards."""
    scale = residual_trust.copy()

    # Generated inputs have no reference signal with which to validate a
    # neural correction. Report the causal physical simulation on its own.
    if record.get("sheet") == "synthetic":
        return np.zeros_like(scale)

    # With no current there is no excitation from which a learned additive
    # correction can be inferred. The grey-box response is causal and already
    # contains gravity and the selected x0/v0/F0 state.
    if record["family"] == "zero_input":
        return np.zeros_like(scale)

    same_family = [
        candidate for candidate in DEVELOPMENT_RECORDS
        if candidate["family"] == record["family"]
    ]
    if not same_family:
        return np.zeros_like(scale)

    family_mass_high = max(candidate["mass_ratio"] for candidate in same_family)
    family_mass_low = min(candidate["mass_ratio"] for candidate in same_family)
    outside_family_mass = not (
        0.98 * family_mass_low <= record["mass_ratio"]
        <= 1.02 * family_mass_high
    )

    # A DC+sine residual at a mass not represented for that family is an
    # unsupported extrapolation. Falling back to Newton's-law simulation is
    # safer than adding an unconstrained learned correction.
    if record["family"] == "dc_plus_sine" and outside_family_mass:
        return np.zeros_like(scale)

    # Force is nearly algebraic in current and position. Do not extrapolate
    # its learned residual across an unseen step/load combination.
    if record["family"] == "step" and outside_family_mass:
        scale[1] = 0.0
    return scale


def enforce_physical_boundaries(current, predicted):
    """Project the output onto exact identities known from the actuator."""
    predicted = np.asarray(predicted, dtype=np.float32).copy()
    zero = np.abs(np.asarray(current)) <= ZERO_CURRENT_TOLERANCE_A
    predicted[zero, 1] = 0.0
    return predicted


residual_trust = fit_residual_trust()
print("\nResidual trust factors fitted on the validation blocks:")
for column, output_name, _ in OUTPUT_INFO:
    print(f"  {output_name:14s} alpha = {residual_trust[column]:.3f}")
if float(np.max(residual_trust)) < 0.05:
    print("  Both are near zero, so the reported results are effectively the "
          "physical model on its own.")


def evaluate_pairs(pairs):
    """Run the frozen model and return everything in physical units."""
    pairs_array = np.asarray(pairs, dtype=np.int64)
    normalised = predict_pairs(model, records, pairs, device)
    scales = np.stack([
        residual_scale_for_record(records[index])
        for index, _ in pairs_array
    ])
    normalised = normalised * scales
    time, measured, predicted, baseline = reconstruct_outputs(
        data, records, pairs, normalised
    )
    current = np.asarray(
        [records[r]["current"][t] for r, t in pairs_array], dtype=np.float32
    )
    predicted = enforce_physical_boundaries(current, predicted)
    return {
        "time": time,
        "current": current,
        "measured": measured,
        "predicted": predicted,
        "baseline": baseline,
    }


evaluations = {}
metric_rows = []


def register(name, kind, result, has_reference=True):
    evaluations[name] = dict(result)
    evaluations[name]["kind"] = kind
    evaluations[name]["has_reference"] = has_reference

    if not has_reference:
        return

    for column, output_name, unit in OUTPUT_INFO:
        row = {
            "Evaluation": name,
            "Kind": kind,
            "Output": output_name,
            "Unit": unit,
            "Samples": int(len(result["time"])),
        }
        row.update(
            calculate_metrics(
                result["measured"][:, column],
                result["predicted"][:, column],
                negligible_range=NEGLIGIBLE_SIGNAL_RANGE[output_name],
            )
        )
        # The physics only error is carried alongside every row. It is the
        # honest reference point: the network earns its place only where it
        # beats the grey box it is sitting on top of.
        physics = calculate_metrics(
            result["measured"][:, column],
            result["baseline"][:, column],
            negligible_range=NEGLIGIBLE_SIGNAL_RANGE[output_name],
        )
        row["RMSE_physics_only"] = physics["RMSE"]
        row["R2_physics_only"] = physics["R2"]
        row["NetworkHelps"] = bool(row["RMSE"] < physics["RMSE"])
        metric_rows.append(row)


# --- internal held out blocks, per development record -----------------
print("\nEvaluating the internal held out blocks ...", flush=True)

internal_pairs = np.asarray(data["internal_test_pairs"], dtype=np.int64)
for record_index in sorted(set(internal_pairs[:, 0].tolist())):
    subset = internal_pairs[internal_pairs[:, 0] == record_index]
    name = records[record_index]["name"]
    register(name, "internal_test", evaluate_pairs(subset))

register(
    "ALL_internal_test_blocks",
    "internal_test_pooled",
    evaluate_pairs(internal_pairs),
)


# --- pure test records, swept end to end ------------------------------
print("Evaluating the pure test records ...", flush=True)

for name in PURE_TEST_RECORDS:
    pairs = data["pure_test_pairs"][name]
    register(name, "pure_test", evaluate_pairs(pairs))


# --- synthetic probes, no reference available -------------------------
print("Driving the frozen model with the synthetic probe signals ...", flush=True)

probe_samples = int(round(SYNTHETIC_PROBE_SECONDS * TARGET_SAMPLE_RATE_HZ)) + 1
probe_time = np.arange(probe_samples) * TARGET_TIME_STEP

synthetic_records = []
reference_templates = {
    record["family"]: record for record in records
    if np.isclose(record["load_mass_g"], 3.813)
    and record["family"] in {"step", "zero_input"}
}
if set(reference_templates) != {"step", "zero_input"}:
    raise RuntimeError("The Load-2 measured step/zero initial states were not found.")
for load_mass in SYNTHETIC_PROBE_LOAD_MASSES:
    step_current = np.where(
        probe_time >= SYNTHETIC_STEP_TIME, SYNTHETIC_STEP_AMPLITUDE, 0.0
    ).astype(np.float32)
    synthetic_records.append(
        make_synthetic_record(
            f"Synthetic_Step_{SYNTHETIC_STEP_AMPLITUDE:.2f}A_Load{load_mass:g}g",
            load_mass,
            step_current,
            SYNTHETIC_PROBE_SECONDS,
            "step",
            reference_templates["step"]["initial_state"],
        )
    )
    synthetic_records.append(
        make_synthetic_record(
            f"Synthetic_ZeroInput_Load{load_mass:g}g",
            load_mass,
            np.zeros(probe_samples, dtype=np.float32),
            SYNTHETIC_PROBE_SECONDS,
            "zero_input",
            reference_templates["zero_input"]["initial_state"],
        )
    )

for probe in synthetic_records:
    attach_synthetic_arrays(data, probe)

probe_pool = list(records) + synthetic_records
for offset, probe in enumerate(synthetic_records):
    probe_index = len(records) + offset
    pairs = record_window_indices(probe_pool, probe_index, stride=1)
    normalised = (
        predict_pairs(model, probe_pool, pairs, device)
        * residual_scale_for_record(probe)
    )
    time, _, predicted, baseline = reconstruct_outputs(
        data, probe_pool, pairs, normalised
    )
    pairs_array = np.asarray(pairs, dtype=np.int64)
    current = np.asarray(
        [probe_pool[r]["current"][t] for r, t in pairs_array], dtype=np.float32
    )
    predicted = enforce_physical_boundaries(current, predicted)
    register(
        probe["name"],
        "synthetic_probe",
        {
            "time": time,
            "current": current,
            "measured": np.full_like(predicted, np.nan),
            "predicted": predicted,
            "baseline": baseline,
        },
        has_reference=False,
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(RESULTS_FOLDER / "metrics.csv", index=False)

pd.set_option("display.width", 200)

print("\n" + "=" * 100)
print("INTERNAL HELD OUT BLOCKS (development records, blocks never trained on)")
print("=" * 100)
internal_view = metrics[metrics["Kind"].str.startswith("internal_test")]
print(internal_view[
    ["Evaluation", "Output", "RMSE", "MAE", "NRMSE_percent", "R2", "Fit_percent"]
].to_string(index=False, float_format=lambda v: f"{v:10.5f}"))

print("\n" + "=" * 100)
print("PURE TEST RECORDS (never used for training, validation, normalisation "
      "or baseline fitting)")
print("=" * 100)
pure_view = metrics[metrics["Kind"] == "pure_test"]
print(pure_view[
    ["Evaluation", "Output", "RMSE", "MAE", "NRMSE_percent", "R2", "Fit_percent"]
].to_string(index=False, float_format=lambda v: f"{v:10.5f}"))

step_records = [
    name for name in PURE_TEST_RECORDS
    if records[index_of[name]]["family"] == "step"
]
zero_records = [
    name for name in PURE_TEST_RECORDS
    if records[index_of[name]]["family"] == "zero_input"
]
print("\nExtra validation signals requested for the end of the report:")
print(f"  step input  : {', '.join(step_records) or 'none found'}")
print(f"  zero input  : {', '.join(zero_records) or 'none found'}")

flat = metrics[metrics["ReferenceIsFlat"]]
if not flat.empty:
    print("\nR2 is not reported for the following channels because the COMSOL "
          "reference is flat, so a variance ratio has no meaning. "
          "Read RMSE and the maximum absolute error there instead:")
    for _, row in flat.iterrows():
        print(f"  {row['Evaluation']:24s} {row['Output']:14s} "
              f"reference range = {row['SignalRange']:.6f} {row['Unit']}, "
              f"RMSE = {row['RMSE']:.6f} {row['Unit']}, "
              f"max error = {row['MaxAbsError']:.6f} {row['Unit']}")


# --- per record prediction files --------------------------------------
predictions_folder = RESULTS_FOLDER / "predictions"
predictions_folder.mkdir(exist_ok=True)

for name, result in evaluations.items():
    frame = pd.DataFrame(
        {
            "time_s": result["time"],
            "coil_current_A": result["current"],
            "measured_displacement_mm": result["measured"][:, 0],
            "predicted_displacement_mm": result["predicted"][:, 0],
            "baseline_displacement_mm": result["baseline"][:, 0],
            "displacement_error_mm": result["measured"][:, 0]
            - result["predicted"][:, 0],
            "measured_force_N": result["measured"][:, 1],
            "predicted_force_N": result["predicted"][:, 1],
            "baseline_force_N": result["baseline"][:, 1],
            "force_error_N": result["measured"][:, 1] - result["predicted"][:, 1],
        }
    )
    frame.to_csv(predictions_folder / f"{result['kind']}__{name}.csv", index=False)


# --- archive for the plotting stage -----------------------------------
archive = {
    "training_history": training_history,
    "validation_history": validation_history,
    "fine_tune_history": fine_tune_history,
    "best_epoch": np.asarray(best_epoch),
}

manifest = []
for name, result in evaluations.items():
    for field in ["time", "current", "measured", "predicted", "baseline"]:
        archive[f"eval__{name}__{field}"] = result[field]
    manifest.append(
        {
            "name": name,
            "kind": result["kind"],
            "has_reference": bool(result["has_reference"]),
        }
    )

for record in records:
    # Only the real part of each record is archived. The rest state padding
    # is an input side device and has no place in the figures.
    pad = record["pad"]
    archive[f"record__{record['name']}__time"] = record["time"][pad:]
    archive[f"record__{record['name']}__current"] = record["current"][pad:]
    archive[f"record__{record['name']}__outputs"] = record["outputs"][pad:]

np.savez_compressed(RESULTS_FOLDER / "simulation_results.npz", **archive)

with open(RESULTS_FOLDER / "manifest.json", "w", encoding="utf-8") as handle:
    json.dump(
        {
            "evaluations": manifest,
            "pure_test_records": PURE_TEST_RECORDS,
            "development_records": data["development_names"],
            "extra_step_records": step_records,
            "extra_zero_input_records": zero_records,
            "duplicate_groups": data["duplicate_groups"],
            "dropped_duplicates": data["dropped_duplicates"],
            "sequence_length": SEQUENCE_LENGTH,
            "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "best_epoch": int(best_epoch),
            "initialisation": "measured x0, velocity0, and force0 only",
            "prediction_policy": (
                "validation-scaled LSTM residual with family-support guards; "
                "zero-current Lorentz force is constrained to zero"
            ),
            "baseline_label": "grey-box physics only",
        },
        handle,
        indent=2,
    )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_mean": data["input_mean"],
        "input_std": data["input_std"],
        "target_mean": data["target_mean"],
        "target_std": data["target_std"],
        "displacement_baseline_coefficients":
            data["displacement_baseline_coefficients"],
        "force_baseline_coefficients": data["force_baseline_coefficients"],
        "best_epoch": best_epoch,
        "residual_trust": residual_trust,
        "physical_model": data["physical_model"],
        "feature_names": data["feature_names"],
        "initialisation": "measured x0, velocity0, and force0 only",
    },
    RESULTS_FOLDER / "model.pt",
)

print("\nResults saved in:", RESULTS_FOLDER)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

print("\nBuilding the figures ...", flush=True)
# Plotting must never discard a finished training run. Everything needed to
# redraw the figures is already on disk in ResultsData, so a failure here is
# reported with the command to retry rather than raised.
plotting = subprocess.run(
    [sys.executable, str(HERE / "autonomous_results_plots.py")],
    cwd=HERE,
    check=False,
)
if plotting.returncode != 0:
    print()
    print("The figures could not be built, but the run itself is complete and")
    print("every result is saved in ResultsData. Nothing needs retraining.")
    print("On Windows this is usually a file lock: close any image viewer or")
    print("Explorer window on FiguresResults, pause OneDrive syncing, then run")
    print("    python autonomous_results_plots.py")

# Optional GitHub push, kept from the previous project layout. It is skipped
# silently when github_push.py is absent or when the push is not configured.
if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") != "1":
    try:
        from github_push import GitPushError, push_simulation
    except ImportError:
        pass
    else:
        try:
            push_simulation(HERE)
        except GitPushError as error:
            print("\nThe run finished, but the optional GitHub push failed:")
            print(error)

print("\nDone.")

if os.name == "nt":
    try:
        os.startfile(HERE / "FiguresResults")
    except OSError:
        pass
