"""
MAIN SIMULATION

This is the short file you run first:

    python main.py

It connects the modular parts:
    data -> model -> training -> pure fifth-sheet test
"""

# The OpenMP fix must appear before numerical-library imports.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# PyTorch thread count is configured below.


from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from config import CPU_THREADS, SEED
from data_utils import prepare_data
from model import OgunmoluCOMSOLLSTM
from github_push import GitPushError, push_simulation
from train_utils import (
    calculate_metrics,
    fine_tune_on_all_development_data,
    predict,
    train_model,
)


# ---------------------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(CPU_THREADS)

HERE = Path(__file__).resolve().parent

RESULTS_FOLDER = HERE / "ResultsData"
RESULTS_FOLDER.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------
# 2. DATA
# ---------------------------------------------------------------------

data = prepare_data(HERE)

data["split_table"].to_csv(
    RESULTS_FOLDER / "data_split.csv",
    index=False,
)

print("Workbook:", data["workbook"].name)
print("Training windows:", len(data["x_train"]))
print("Validation windows:", len(data["x_validation"]))
print("Pure-test windows:", len(data["x_test"]))


# ---------------------------------------------------------------------
# 3. MODEL
# ---------------------------------------------------------------------

model = OgunmoluCOMSOLLSTM().to(device)

print("\nComplete signal-to-output path:")
print("1 measured current -> 3 features [I, delta_I, I_DC]")
print("-> LSTM 32 -> LSTM 64 -> LSTM 64 -> Linear 2 outputs")
print("Compact form: 1 -> 3 -> 32 -> 64 -> 64 -> 2\n")

print(model)
print("Device:", device)
print("CPU threads:", CPU_THREADS)


# ---------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------

training_history, validation_history, best_epoch = train_model(
    model,
    data,
    device,
)


# ---------------------------------------------------------------------
# 4A. MINIMAL FINAL FINE-TUNING
# ---------------------------------------------------------------------

print(
    "\nUsing all 67-127 mA development data for four "
    "low-learning-rate fine-tuning epochs..."
)

fine_tune_history = fine_tune_on_all_development_data(
    model,
    data,
    device,
)


# ---------------------------------------------------------------------
# 5. PURE FIFTH-SHEET TEST
# ---------------------------------------------------------------------

prediction_n = predict(
    model,
    data["x_test"],
    device,
)

prediction = (
    prediction_n * data["output_std"]
    + data["output_mean"]
)

measured = (
    data["y_test"].numpy()
    * data["output_std"]
    + data["output_mean"]
)


# ---------------------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------------------

metric_rows = []

for output_index, output_name, unit in [
    (0, "Displacement", "mm"),
    (1, "Lorentz force", "N"),
]:
    row = {
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

print("\nPure 147 mA test metrics:")
print(metrics.to_string(index=False))

metrics.to_csv(
    RESULTS_FOLDER / "metrics.csv",
    index=False,
)

# Save every final-test prediction in a readable CSV file.
pd.DataFrame(
    {
        "time_s": data["test_time"],
        "measured_displacement_mm": measured[:, 0],
        "predicted_displacement_mm": prediction[:, 0],
        "displacement_error_mm": measured[:, 0] - prediction[:, 0],
        "measured_force_N": measured[:, 1],
        "predicted_force_N": prediction[:, 1],
        "force_error_N": measured[:, 1] - prediction[:, 1],
    }
).to_csv(
    RESULTS_FOLDER / "final_test_predictions.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------------------

np.savez_compressed(
    RESULTS_FOLDER / "simulation_results.npz",
    test_time=data["test_time"],
    measured=measured,
    prediction=prediction,
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
print("Creating figures automatically...")

subprocess.run(
    [
        sys.executable,
        str(HERE / "plot_results.py"),
    ],
    cwd=HERE,
    check=True,
)

print("Figures saved in:", HERE / "FiguresResults")


# ---------------------------------------------------------------------
# 9. PUSH THIS COMPLETED SIMULATION TO GITHUB
# ---------------------------------------------------------------------

# This is the only place where main.py calls the GitHub helper.
# It runs after the results and figures have been created successfully.
try:
    push_simulation(HERE)
except GitPushError as error:
    print(
        "\nSimulation completed, but the GitHub push failed:"
    )
    print(error)


if os.name == "nt":
    try:
        os.startfile(HERE / "FiguresResults")
    except OSError:
        pass
