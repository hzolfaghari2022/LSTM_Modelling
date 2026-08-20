"""
Actuator system identification, full data set study.

Run this file. It loads every record in Total_Data.xlsx, trains the
configuration conditioned LSTM on the development records only, and then
evaluates the frozen model on a set of records that were never touched
during training.

Pure test records, in the order they appear in the report:

    Chirp_DC147_Load2     unseen chirp amplitude at the reference mass
    Load3_DCSine_200mA    unseen load mass
    Load3_Sine_200mA      unseen load mass
    Load3_Step_200mA      unseen load mass
    Load3_ZeroInput       unseen load mass
    Load2_Step_150mA      extra validation signal 1, step input
    Load2_ZeroInput       extra validation signal 2, zero input

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
RESULTS_FOLDER = HERE / "ResultsData"
RESULTS_FOLDER.mkdir(exist_ok=True)

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
    print("  m x'' = Bl(x) i - m g - c x' - Fs(x),  x(0) = 0, x'(0) = 0")
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

print("\nShort fine tune on training plus validation windows. "
      "Internal test blocks and every pure test record stay untouched.",
      flush=True)
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


# The network may only correct configurations it was actually trained on.
#
# Its inputs include the mass ratio, and at the 7.625 g load that ratio is 1.73
# against a training range that stops at 1.00. The network is then extrapolating
# in its own input space and its correction is not merely useless but actively
# destructive: on the heaviest load it turned a physics error of 0.037 mm into
# 0.938 mm on the zero input record, and 0.050 mm into 0.962 mm on the sine
# record. The validation blocks cannot catch this because they all sit at the
# two training masses.
#
# Outside the training envelope the physics is therefore reported on its own.
TRAINED_MASS_RATIOS = [
    record["mass_ratio"] for record in records if not record["is_pure_test"]
]
MASS_RATIO_LOW = min(TRAINED_MASS_RATIOS) * 0.98
MASS_RATIO_HIGH = max(TRAINED_MASS_RATIOS) * 1.02


def network_is_in_range(record):
    return MASS_RATIO_LOW <= record["mass_ratio"] <= MASS_RATIO_HIGH


residual_trust = fit_residual_trust()
print("\nResidual trust factors fitted on the validation blocks:")
for column, output_name, _ in OUTPUT_INFO:
    print(f"  {output_name:14s} alpha = {residual_trust[column]:.3f}")
if float(np.max(residual_trust)) < 0.05:
    print("  Both are near zero, so the reported results are effectively the "
          "physical model on its own.")


def evaluate_pairs(pairs):
    """Run the frozen model and return everything in physical units."""
    normalised = predict_pairs(model, records, pairs, device) * residual_trust
    in_range = np.array(
        [network_is_in_range(records[index]) for index, _ in np.asarray(pairs)],
        dtype=bool,
    )
    normalised[~in_range] = 0.0
    time, measured, predicted, baseline = reconstruct_outputs(
        data, records, pairs, normalised
    )
    pairs_array = np.asarray(pairs, dtype=np.int64)
    current = np.asarray(
        [records[r]["current"][t] for r, t in pairs_array], dtype=np.float32
    )
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
        )
    )
    synthetic_records.append(
        make_synthetic_record(
            f"Synthetic_ZeroInput_Load{load_mass:g}g",
            load_mass,
            np.zeros(probe_samples, dtype=np.float32),
            SYNTHETIC_PROBE_SECONDS,
        )
    )

for probe in synthetic_records:
    attach_synthetic_arrays(data, probe)

probe_pool = list(records) + synthetic_records
for offset, probe in enumerate(synthetic_records):
    probe_index = len(records) + offset
    pairs = record_window_indices(probe_pool, probe_index, stride=1)
    normalised = predict_pairs(model, probe_pool, pairs, device) * residual_trust
    time, _, predicted, baseline = reconstruct_outputs(
        data, probe_pool, pairs, normalised
    )
    pairs_array = np.asarray(pairs, dtype=np.int64)
    current = np.asarray(
        [probe_pool[r]["current"][t] for r, t in pairs_array], dtype=np.float32
    )
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
    [sys.executable, str(HERE / "plot_results.py")], cwd=HERE, check=False
)
if plotting.returncode != 0:
    print()
    print("The figures could not be built, but the run itself is complete and")
    print("every result is saved in ResultsData. Nothing needs retraining.")
    print("On Windows this is usually a file lock: close any image viewer or")
    print("Explorer window on FiguresResults, pause OneDrive syncing, then run")
    print("    python plot_results.py")

# Optional GitHub push, kept from the previous project layout. It is skipped
# silently when github_push.py is absent or when the push is not configured.
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
