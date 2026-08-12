"""Run the five-series deep LSTM simulation."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from config import CPU_THREADS, FINAL_TEST_SHEET, MIXED_SHEET, SEED
from data_utils import prepare_data
from github_push import GitPushError, push_simulation
from model import DeepLSTMSystemIdentifier
from train_utils import (
    calculate_metrics,
    fine_tune_on_all_development_data,
    predict,
    train_model,
)


# ---------------------------------------------------------------------
# 1. SETUP AND DATA
# ---------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(CPU_THREADS)

HERE = Path(__file__).resolve().parent
RESULTS_FOLDER = HERE / "ResultsData"
RESULTS_FOLDER.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = prepare_data(HERE)
data["split_table"].to_csv(
    RESULTS_FOLDER / "data_split.csv",
    index=False,
)

print("Workbook:", data["workbook"].name)
print("Training windows (sheets 1-3 + distributed sheet 4):", len(data["x_train"]))
print("Validation windows (distributed sheet 4):", len(data["x_validation"]))
print("Internal-test windows (distributed sheet 4):", len(data["x_development_test"]))
print("Pure-test windows (untouched sheet 5):", len(data["x_final_test"]))


# ---------------------------------------------------------------------
# 2. MODEL AND TRAINING
# ---------------------------------------------------------------------

model = DeepLSTMSystemIdentifier().to(device)

print("\nComplete signal-to-output path:")
print("1 measured current -> 3 features [I, delta_I, I_DC]")
print("-> LSTM 32 -> LSTM 64 -> LSTM 64 -> Linear 2 outputs")
print("Compact form: 1 -> 3 -> 32 -> 64 -> 64 -> 2\n")
print(model)
print("Device:", device)

training_history, validation_history, best_epoch = train_model(
    model,
    data,
    device,
)

print(
    "\nFine-tuning the selected model on training + validation only; "
    "both test sets remain untouched..."
)
fine_tune_history = fine_tune_on_all_development_data(
    model,
    data,
    device,
)


# ---------------------------------------------------------------------
# 3. INTERNAL FOURTH-SERIES TEST AND PURE FIFTH-SERIES TEST
# ---------------------------------------------------------------------

def evaluate(x, y):
    """Predict and return denormalized measured/predicted arrays."""

    prediction_n = predict(model, x, device)
    prediction = prediction_n * data["output_std"] + data["output_mean"]
    measured = y.numpy() * data["output_std"] + data["output_mean"]
    return measured, prediction


development_measured, development_prediction = evaluate(
    data["x_development_test"],
    data["y_development_test"],
)
final_measured, final_prediction = evaluate(
    data["x_final_test"],
    data["y_final_test"],
)

evaluation_sets = [
    (
        f"{MIXED_SHEET} distributed internal test",
        development_measured,
        development_prediction,
    ),
    (
        f"{FINAL_TEST_SHEET} untouched pure test",
        final_measured,
        final_prediction,
    ),
]

metric_rows = []
for evaluation_name, measured, prediction in evaluation_sets:
    for output_index, output_name, unit in [
        (0, "Displacement", "mm"),
        (1, "Lorentz force", "N"),
    ]:
        row = {
            "Evaluation": evaluation_name,
            "Output": output_name,
            "Unit": unit,
        }
        row.update(
            calculate_metrics(
                measured[:, output_index],
                prediction[:, output_index],
            )
        )
        metric_rows.append(row)

metrics = pd.DataFrame(metric_rows)
print("\nTest metrics:")
print(metrics.to_string(index=False))
metrics.to_csv(RESULTS_FOLDER / "metrics.csv", index=False)


def save_predictions(file_name, time, measured, prediction):
    pd.DataFrame(
        {
            "time_s": time,
            "measured_displacement_mm": measured[:, 0],
            "predicted_displacement_mm": prediction[:, 0],
            "displacement_error_mm": measured[:, 0] - prediction[:, 0],
            "measured_force_N": measured[:, 1],
            "predicted_force_N": prediction[:, 1],
            "force_error_N": measured[:, 1] - prediction[:, 1],
        }
    ).to_csv(RESULTS_FOLDER / file_name, index=False)


save_predictions(
    "development_test_predictions.csv",
    data["development_test_time"],
    development_measured,
    development_prediction,
)
save_predictions(
    "final_test_predictions.csv",
    data["final_test_time"],
    final_measured,
    final_prediction,
)


# ---------------------------------------------------------------------
# 4. SAVE NUMERICAL RESULTS, MODEL, AND FIGURES
# ---------------------------------------------------------------------

np.savez_compressed(
    RESULTS_FOLDER / "simulation_results.npz",
    development_test_time=data["development_test_time"],
    development_measured=development_measured,
    development_prediction=development_prediction,
    final_test_time=data["final_test_time"],
    final_measured=final_measured,
    final_prediction=final_prediction,
    training_history=training_history,
    validation_history=validation_history,
    best_epoch=np.asarray(best_epoch),
    fine_tune_history=fine_tune_history,
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_mean": data["input_mean"],
        "input_std": data["input_std"],
        "output_mean": data["output_mean"],
        "output_std": data["output_std"],
        "best_epoch": best_epoch,
        "fine_tune_history": fine_tune_history,
    },
    RESULTS_FOLDER / "model.pt",
)

print("\nResults saved in:", RESULTS_FOLDER)
subprocess.run(
    [sys.executable, str(HERE / "plot_results.py")],
    cwd=HERE,
    check=True,
)


# ---------------------------------------------------------------------
# 5. OPTIONAL EXISTING GITHUB HELPER
# ---------------------------------------------------------------------

# Optional repository update after the numerical outputs are saved.
try:
    push_simulation(HERE)
except GitPushError as error:
    print("\nSimulation completed, but the optional GitHub push failed:")
    print(error)

if os.name == "nt":
    try:
        os.startfile(HERE / "FiguresResults")
    except OSError:
        pass
