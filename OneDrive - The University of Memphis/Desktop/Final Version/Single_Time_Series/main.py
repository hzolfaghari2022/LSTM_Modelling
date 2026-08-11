"""Run the selectable one-series FARNN simulation with ``python main.py``."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from config import CPU_THREADS, SEED, SINGLE_SERIES_SHEET
from data_utils import prepare_data
from github_push import GitPushError, push_simulation
from model import OgunmoluCOMSOLLSTM
from train_utils import (
    calculate_metrics,
    fine_tune_on_all_development_data,
    predict,
    train_model,
)


torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(CPU_THREADS)

HERE = Path(__file__).resolve().parent
RESULTS_FOLDER = HERE / "ResultsData"
RESULTS_FOLDER.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1. ONE SELECTED RECORD, DISTRIBUTED OVER ALL THREE ROLES
# ---------------------------------------------------------------------

data = prepare_data(HERE)
data["split_table"].to_csv(RESULTS_FOLDER / "data_split.csv", index=False)

print("Workbook:", data["workbook"].name)
print("Selected series:", SINGLE_SERIES_SHEET)
print("Training windows:", len(data["x_train"]))
print("Validation windows:", len(data["x_validation"]))
print("Test windows:", len(data["x_test"]))


# ---------------------------------------------------------------------
# 2. SAME PAPER-ALIGNED MODEL AS THE FIVE-SERIES VERSION
# ---------------------------------------------------------------------

model = OgunmoluCOMSOLLSTM().to(device)
print("\nCompact architecture: 1 -> 3 -> 32 -> 64 -> 64 -> 2")
print(model)
print("Device:", device)

training_history, validation_history, best_epoch = train_model(
    model,
    data,
    device,
)

print(
    "\nFine-tuning the validation-selected model on training + "
    "validation blocks; test blocks remain untouched..."
)
fine_tune_history = fine_tune_on_all_development_data(
    model,
    data,
    device,
)


# ---------------------------------------------------------------------
# 3. DISTRIBUTED TEST BLOCKS FROM THE SAME RECORD
# ---------------------------------------------------------------------

prediction_n = predict(model, data["x_test"], device)
prediction = prediction_n * data["output_std"] + data["output_mean"]
measured = data["y_test"].numpy() * data["output_std"] + data["output_mean"]

metric_rows = []
for output_index, output_name, unit in [
    (0, "Displacement", "mm"),
    (1, "Lorentz force", "N"),
]:
    row = {
        "Series": SINGLE_SERIES_SHEET,
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
print("\nDistributed one-series test metrics:")
print(metrics.to_string(index=False))
metrics.to_csv(RESULTS_FOLDER / "metrics.csv", index=False)

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
).to_csv(RESULTS_FOLDER / "test_predictions.csv", index=False)

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
        "selected_series": SINGLE_SERIES_SHEET,
        "input_mean": data["input_mean"],
        "input_std": data["input_std"],
        "output_mean": data["output_mean"],
        "output_std": data["output_std"],
        "best_epoch": best_epoch,
        "fine_tune_history": fine_tune_history,
    },
    RESULTS_FOLDER / "model.pt",
)

subprocess.run(
    [sys.executable, str(HERE / "plot_results.py")],
    cwd=HERE,
    check=True,
)


# Retained from the supplied project.  It is optional and runs only after
# all simulation outputs have been saved successfully.
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
