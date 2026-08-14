# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Run the five-series deep LSTM simulation."""

# Import operating-system tools for environment variables, CPU counts, and Windows checks.
import os

# Set this process environment option before numerical libraries are imported.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import selected names from pathlib instead of importing its complete namespace.
from pathlib import Path
# Import subprocess support for running Git commands or the separate plotting script.
import subprocess
# Import interpreter information so another Python script can be started with the same environment.
import sys

# Import NumPy for numerical arrays, normalization, errors, and metrics.
import numpy as np
# Import pandas for reading Excel/CSV files and building result tables.
import pandas as pd
# Import PyTorch for tensors, the LSTM model, training, and prediction.
import torch

# Import selected names from config instead of importing its complete namespace.
from config import CPU_THREADS, FINAL_TEST_SHEET, MIXED_SHEET, SEED
# Import selected names from data_utils instead of importing its complete namespace.
from data_utils import prepare_data
# Import selected names from github_push instead of importing its complete namespace.
from github_push import GitPushError, push_simulation
# Import selected names from model instead of importing its complete namespace.
from model import DeepLSTMSystemIdentifier
# Import selected names from train_utils instead of importing its complete namespace.
from train_utils import (
    # Pass `calculate_metrics` as the next value required by the surrounding call or collection.
    calculate_metrics,
    # Pass `fine_tune_on_all_development_data` as the next value required by the surrounding call or collection.
    fine_tune_on_all_development_data,
    # Pass `predict` as the next value required by the surrounding call or collection.
    predict,
    # Pass `train_model` as the next value required by the surrounding call or collection.
    train_model,
# Close the current function call, tuple, or grouped expression.
)


# 1. Fix the random seed, choose a device, and prepare the data

# Apply the fixed seed so this source of randomness is reproducible.
torch.manual_seed(SEED)
# Apply the fixed seed so this source of randomness is reproducible.
np.random.seed(SEED)
# Limit PyTorch to the configured number of CPU worker threads.
torch.set_num_threads(CPU_THREADS)

# Store the folder containing the current script so every path is relative to the project.
HERE = Path(__file__).resolve().parent
# Store the folder used for numerical output files.
RESULTS_FOLDER = HERE / "ResultsData"
# Call `RESULTS_FOLDER.mkdir`; the following indented continuation lines provide its arguments.
RESULTS_FOLDER.mkdir(exist_ok=True)
# Choose a CUDA GPU when available; otherwise use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store the prepared data or the table returned by the current read operation.
data = prepare_data(HERE)
# Select data["split_table"].to_csv( from the prepared dataset for this operation.
data["split_table"].to_csv(
    # Use the expression `RESULTS_FOLDER / "data_split.csv"` as the next part of the surrounding Python statement.
    RESULTS_FOLDER / "data_split.csv",
    # Control whether pandas writes its own row-number column; False keeps result files clean.
    index=False,
# Close the current function call, tuple, or grouped expression.
)

# Print this progress or result message in the terminal.
print("Workbook:", data["workbook"].name)
# Print this progress or result message in the terminal.
print("Training windows (sheets 1-3 + distributed sheet 4):", len(data["x_train"]))
# Print this progress or result message in the terminal.
print("Validation windows (distributed sheet 4):", len(data["x_validation"]))
# Print this progress or result message in the terminal.
print("Internal-test windows (distributed sheet 4):", len(data["x_development_test"]))
# Print this progress or result message in the terminal.
print("Pure-test windows (untouched sheet 5):", len(data["x_final_test"]))


# 2. Build the 3-32-64-64-2 model and train its parameters

# Create the stacked LSTM model and place its parameters on the selected device.
model = DeepLSTMSystemIdentifier().to(device)

# Print this progress or result message in the terminal.
print("\nComplete signal-to-output path:")
# Print this progress or result message in the terminal.
print("1 measured current -> 3 features [I, delta_I, I_DC]")
# Print this progress or result message in the terminal.
print("-> LSTM 32 -> LSTM 64 -> LSTM 64 -> Linear 2 outputs")
# Print this progress or result message in the terminal.
print("Compact form: 1 -> 3 -> 32 -> 64 -> 64 -> 2\n")
# Print this progress or result message in the terminal.
print(model)
# Print this progress or result message in the terminal.
print("Device:", device)

# Use the expression `training_history, validation_history, best_epoch = train_model(` as the next part of the surrounding Python statement.
training_history, validation_history, best_epoch = train_model(
    # Pass `model` as the next value required by the surrounding call or collection.
    model,
    # Pass `data` as the next value required by the surrounding call or collection.
    data,
    # Pass `device` as the next value required by the surrounding call or collection.
    device,
# Close the current function call, tuple, or grouped expression.
)

# Print this progress or result message in the terminal.
print(
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "\nFine-tuning the selected model on training + validation only; "
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "both test sets remain untouched..."
# Close the current function call, tuple, or grouped expression.
)
# Evaluate `fine_tune_on_all_development_data(` and store the result in `fine_tune_history` for the following steps.
fine_tune_history = fine_tune_on_all_development_data(
    # Pass `model` as the next value required by the surrounding call or collection.
    model,
    # Pass `data` as the next value required by the surrounding call or collection.
    data,
    # Pass `device` as the next value required by the surrounding call or collection.
    device,
# Close the current function call, tuple, or grouped expression.
)


# 3. Evaluate held-out fourth-series blocks and the untouched fifth series

# Define the evaluate function; its indented lines form the function body.
def evaluate(x, y):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Predict and return denormalized measured/predicted arrays."""

    # Evaluate `predict(model, x, device)` and store the result in `prediction_n` for the following steps.
    prediction_n = predict(model, x, device)
    # Convert standardized outputs back to mm and N before scoring.
    prediction = prediction_n * data["output_std"] + data["output_mean"]
    # Store the measured output values used as the reference for evaluation.
    measured = y.numpy() * data["output_std"] + data["output_mean"]
    # Return this value to the code that called the current function.
    return measured, prediction


# Use the expression `development_measured, development_prediction = evaluate(` as the next part of the surrounding Python statement.
development_measured, development_prediction = evaluate(
    # Select `data["x_development_test"]` from the current array, tensor, table, or dictionary.
    data["x_development_test"],
    # Select `data["y_development_test"]` from the current array, tensor, table, or dictionary.
    data["y_development_test"],
# Close the current function call, tuple, or grouped expression.
)
# Use the expression `final_measured, final_prediction = evaluate(` as the next part of the surrounding Python statement.
final_measured, final_prediction = evaluate(
    # Select `data["x_final_test"]` from the current array, tensor, table, or dictionary.
    data["x_final_test"],
    # Select `data["y_final_test"]` from the current array, tensor, table, or dictionary.
    data["y_final_test"],
# Close the current function call, tuple, or grouped expression.
)

# Evaluate `[` and store the result in `evaluation_sets` for the following steps.
evaluation_sets = [
    # Begin the grouped expression or collection continued on the following lines.
    (
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        f"{MIXED_SHEET} distributed internal test",
        # Pass `development_measured` as the next value required by the surrounding call or collection.
        development_measured,
        # Pass `development_prediction` as the next value required by the surrounding call or collection.
        development_prediction,
    # Close the current function call, tuple, or grouped expression.
    ),
    # Begin the grouped expression or collection continued on the following lines.
    (
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        f"{FINAL_TEST_SHEET} untouched pure test",
        # Pass `final_measured` as the next value required by the surrounding call or collection.
        final_measured,
        # Pass `final_prediction` as the next value required by the surrounding call or collection.
        final_prediction,
    # Close the current function call, tuple, or grouped expression.
    ),
# Close the current list or index expression.
]

# Calculate the same metrics for both outputs and both evaluations.
metric_rows = []
# Repeat the following indented block once for each item in this iterable.
for evaluation_name, measured, prediction in evaluation_sets:
    # Repeat the following indented block once for each item in this iterable.
    for output_index, output_name, unit in [
        # Begin the grouped expression or collection continued on the following lines.
        (0, "Displacement", "mm"),
        # Begin the grouped expression or collection continued on the following lines.
        (1, "Lorentz force", "N"),
    # Begin the indented block controlled by this statement.
    ]:
        # Evaluate `{` and store the result in `row` for the following steps.
        row = {
            # Store the 'Evaluation' field in the current dictionary.
            "Evaluation": evaluation_name,
            # Store the 'Output' field in the current dictionary.
            "Output": output_name,
            # Store the 'Unit' field in the current dictionary.
            "Unit": unit,
        # Close the current dictionary.
        }
        # Call `row.update`; the following indented continuation lines provide its arguments.
        row.update(
            # Call `calculate_metrics`; the following indented continuation lines provide its arguments.
            calculate_metrics(
                # Select `measured[:, output_index]` from the current array, tensor, table, or dictionary.
                measured[:, output_index],
                # Select `prediction[:, output_index]` from the current array, tensor, table, or dictionary.
                prediction[:, output_index],
            # Close the current function call, tuple, or grouped expression.
            )
        # Close the current function call, tuple, or grouped expression.
        )
        # Call `metric_rows.append`; the following indented continuation lines provide its arguments.
        metric_rows.append(row)

# Store the calculated performance rows in a pandas table.
metrics = pd.DataFrame(metric_rows)
# Print this progress or result message in the terminal.
print("\nTest metrics:")
# Print this progress or result message in the terminal.
print(metrics.to_string(index=False))
# Call `metrics.to_csv`; the following indented continuation lines provide its arguments.
metrics.to_csv(RESULTS_FOLDER / "metrics.csv", index=False)


# Define the save_predictions function; its indented lines form the function body.
def save_predictions(file_name, time, measured, prediction):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Save measured values, predictions, and pointwise errors."""

    # Call `pd.DataFrame`; the following indented continuation lines provide its arguments.
    pd.DataFrame(
        # Begin the grouped expression or collection continued on the following lines.
        {
            # Store the 'time_s' field in the current dictionary.
            "time_s": time,
            # Store the 'measured_displacement_mm' field in the current dictionary.
            "measured_displacement_mm": measured[:, 0],
            # Store the 'predicted_displacement_mm' field in the current dictionary.
            "predicted_displacement_mm": prediction[:, 0],
            # Store the 'displacement_error_mm' field in the current dictionary.
            "displacement_error_mm": measured[:, 0] - prediction[:, 0],
            # Store the 'measured_force_N' field in the current dictionary.
            "measured_force_N": measured[:, 1],
            # Store the 'predicted_force_N' field in the current dictionary.
            "predicted_force_N": prediction[:, 1],
            # Store the 'force_error_N' field in the current dictionary.
            "force_error_N": measured[:, 1] - prediction[:, 1],
        # Close the current dictionary.
        }
    # Use the expression `).to_csv(RESULTS_FOLDER / file_name, index=False)` as the next part of the surrounding Python statement.
    ).to_csv(RESULTS_FOLDER / file_name, index=False)


# Call `save_predictions`; the following indented continuation lines provide its arguments.
save_predictions(
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "development_test_predictions.csv",
    # Select `data["development_test_time"]` from the current array, tensor, table, or dictionary.
    data["development_test_time"],
    # Pass `development_measured` as the next value required by the surrounding call or collection.
    development_measured,
    # Pass `development_prediction` as the next value required by the surrounding call or collection.
    development_prediction,
# Close the current function call, tuple, or grouped expression.
)
# Call `save_predictions`; the following indented continuation lines provide its arguments.
save_predictions(
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "final_test_predictions.csv",
    # Select `data["final_test_time"]` from the current array, tensor, table, or dictionary.
    data["final_test_time"],
    # Pass `final_measured` as the next value required by the surrounding call or collection.
    final_measured,
    # Pass `final_prediction` as the next value required by the surrounding call or collection.
    final_prediction,
# Close the current function call, tuple, or grouped expression.
)


# 4. Save histories, scaling values, weights, tables, and figures

# Call `np.savez_compressed`; the following indented continuation lines provide its arguments.
np.savez_compressed(
    # Use the expression `RESULTS_FOLDER / "simulation_results.npz"` as the next part of the surrounding Python statement.
    RESULTS_FOLDER / "simulation_results.npz",
    # Pass `data["development_test_time"]` as the `development_test_time` argument of the surrounding function call.
    development_test_time=data["development_test_time"],
    # Load the measured internal-test outputs used as the plotting reference.
    development_measured=development_measured,
    # Load the model predictions aligned with the internal-test measurements.
    development_prediction=development_prediction,
    # Pass `data["final_test_time"]` as the `final_test_time` argument of the surrounding function call.
    final_test_time=data["final_test_time"],
    # Load measured displacement and force from the untouched external test.
    final_measured=final_measured,
    # Load predictions for the untouched external test.
    final_prediction=final_prediction,
    # Create storage for the loss recorded after each training epoch.
    training_history=training_history,
    # Create storage for the loss recorded after each validation epoch.
    validation_history=validation_history,
    # Initialize the index of the best validation epoch before training begins.
    best_epoch=np.asarray(best_epoch),
    # Pass `fine_tune_history` as the `fine_tune_history` argument of the surrounding function call.
    fine_tune_history=fine_tune_history,
# Close the current function call, tuple, or grouped expression.
)

# Call `torch.save`; the following indented continuation lines provide its arguments.
torch.save(
    # Begin the grouped expression or collection continued on the following lines.
    {
        # Store the 'model_state_dict' field in the current dictionary.
        "model_state_dict": model.state_dict(),
        # Store the 'input_mean' field in the current dictionary.
        "input_mean": data["input_mean"],
        # Store the 'input_std' field in the current dictionary.
        "input_std": data["input_std"],
        # Store the 'output_mean' field in the current dictionary.
        "output_mean": data["output_mean"],
        # Store the 'output_std' field in the current dictionary.
        "output_std": data["output_std"],
        # Store the 'best_epoch' field in the current dictionary.
        "best_epoch": best_epoch,
        # Store the 'fine_tune_history' field in the current dictionary.
        "fine_tune_history": fine_tune_history,
    # Close the current dictionary.
    },
    # Use the expression `RESULTS_FOLDER / "model.pt"` as the next part of the surrounding Python statement.
    RESULTS_FOLDER / "model.pt",
# Close the current function call, tuple, or grouped expression.
)

# Print this progress or result message in the terminal.
print("\nResults saved in:", RESULTS_FOLDER)
# Start the requested external command and wait for it to finish.
subprocess.run(
    # Begin the grouped expression or collection continued on the following lines.
    [sys.executable, str(HERE / "plot_results.py")],
    # Run the external command from this explicitly selected working directory.
    cwd=HERE,
    # Do not let subprocess raise automatically; this code checks the return code and produces its own message.
    check=True,
# Close the current function call, tuple, or grouped expression.
)


# 5. Optionally commit and push the completed simulation folder

# Optional repository update after the numerical outputs are saved.
try:
    # Call `push_simulation`; the following indented continuation lines provide its arguments.
    push_simulation(HERE)
# Handle the stated exception instead of ending with an unprocessed traceback.
except GitPushError as error:
    # Print this progress or result message in the terminal.
    print("\nSimulation completed, but the optional GitHub push failed:")
    # Print this progress or result message in the terminal.
    print(error)

# Evaluate this condition and run the following indented block only when it is true.
if os.name == "nt":
    # Attempt the following operation so an expected failure can be handled cleanly.
    try:
        # Call `os.startfile`; the following indented continuation lines provide its arguments.
        os.startfile(HERE / "FiguresResults")
    # Handle the stated exception instead of ending with an unprocessed traceback.
    except OSError:
        # Pass `pass` as the next value required by the surrounding call or collection.
        pass
