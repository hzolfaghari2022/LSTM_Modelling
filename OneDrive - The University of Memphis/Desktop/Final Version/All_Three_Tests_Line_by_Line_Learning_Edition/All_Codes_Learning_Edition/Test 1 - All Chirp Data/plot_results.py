# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Create figures from the saved five-series simulation results."""

# Import operating-system tools for environment variables, CPU counts, and Windows checks.
import os
# Import high-level file copying tools used for report figures.
import shutil

# Set this process environment option before numerical libraries are imported.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set this process environment option before numerical libraries are imported.
os.environ["OMP_NUM_THREADS"] = "1"
# Set this process environment option before numerical libraries are imported.
os.environ["MKL_NUM_THREADS"] = "1"

# Import selected names from pathlib instead of importing its complete namespace.
from pathlib import Path

# Import Matplotlib for creating and saving result figures.
import matplotlib

# Select a non-interactive plotting backend so figures can be saved without a display window.
matplotlib.use("Agg")
# Import Matplotlib for creating and saving result figures.
import matplotlib.pyplot as plt
# Import NumPy for numerical arrays, normalization, errors, and metrics.
import numpy as np
# Import pandas for reading Excel/CSV files and building result tables.
import pandas as pd

# Import selected names from config instead of importing its complete namespace.
from config import FINAL_TEST_SHEET, MIXED_SHEET, TRAIN_ONLY_SHEETS
# Import selected names from data_utils instead of importing its complete namespace.
from data_utils import find_workbook, load_sheet


# Store the folder containing the current script so every path is relative to the project.
HERE = Path(__file__).resolve().parent
# Store the folder used for numerical output files.
RESULTS_FOLDER = HERE / "ResultsData"
# Store the folder used for figure files.
FIGURES_FOLDER = HERE / "FiguresResults"
# Call `FIGURES_FOLDER.mkdir`; the following indented continuation lines provide its arguments.
FIGURES_FOLDER.mkdir(exist_ok=True)

# Load saved values once; this file does not retrain the model.
results = np.load(RESULTS_FOLDER / "simulation_results.npz")
# Store the calculated performance rows in a pandas table.
metrics = pd.read_csv(RESULTS_FOLDER / "metrics.csv")
# Read the saved block-role table so plots show exactly which data entered each stage.
split_table = pd.read_csv(RESULTS_FOLDER / "data_split.csv")

# Load the time values belonging to the internal development evaluation.
development_time = results["development_test_time"]
# Load the measured internal-test outputs used as the plotting reference.
development_measured = results["development_measured"]
# Load the model predictions aligned with the internal-test measurements.
development_prediction = results["development_prediction"]
# Load time values for the untouched external test record.
final_time = results["final_test_time"]
# Load measured displacement and force from the untouched external test.
final_measured = results["final_measured"]
# Load predictions for the untouched external test.
final_prediction = results["final_prediction"]
# Create storage for the loss recorded after each training epoch.
training_history = results["training_history"]
# Create storage for the loss recorded after each validation epoch.
validation_history = results["validation_history"]
# Initialize the index of the best validation epoch before training begins.
best_epoch = int(results["best_epoch"])

# Map each data role to a consistent plot color.
ROLE_COLORS = {
    # Store the 'training' field in the current dictionary.
    "training": "tab:blue",
    # Store the 'validation' field in the current dictionary.
    "validation": "tab:orange",
    # Store the 'test' field in the current dictionary.
    "test": "tab:green",
    # Store the 'unused_short_remainder' field in the current dictionary.
    "unused_short_remainder": "0.75",
# Close the current dictionary.
}
# Map each output column to its name and physical unit for plotting.
OUTPUTS = [
    # Begin the grouped expression or collection continued on the following lines.
    (0, "displacement", "Displacement", "mm"),
    # Begin the grouped expression or collection continued on the following lines.
    (1, "force", "Lorentz force", "N"),
# Close the current list or index expression.
]


# Shared helpers for saving figures and handling separated test blocks
def finish_figure(file_name):
    # Call `plt.grid`; the following indented continuation lines provide its arguments.
    plt.grid(True, alpha=0.3)
    # Call `plt.tight_layout`; the following indented continuation lines provide its arguments.
    plt.tight_layout()
    # Call `plt.savefig`; the following indented continuation lines provide its arguments.
    plt.savefig(FIGURES_FOLDER / file_name, dpi=180, bbox_inches="tight")
    # Close the figure after saving it so memory is released.
    plt.close()


# Define the plot_discontinuous function; its indented lines form the function body.
def plot_discontinuous(time, values, **kwargs):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Plot distributed blocks without drawing lines across time gaps."""

    # Evaluate this condition and run the following indented block only when it is true.
    if len(time) < 2:
        # Call `plt.plot`; the following indented continuation lines provide its arguments.
        plt.plot(time, values, **kwargs)
        # Return this value to the code that called the current function.
        return
    # Set the time-gap threshold used to separate noncontiguous held-out blocks in a plot.
    gap_threshold = 3.0 * np.median(np.diff(time))
    # Find the first index of each continuous plotted test segment.
    segment_starts = np.r_[0, np.where(np.diff(time) > gap_threshold)[0] + 1]
    # Find the exclusive final index of each continuous plotted test segment.
    segment_stops = np.r_[segment_starts[1:], len(time)]
    # Repeat the following indented block once for each item in this iterable.
    for segment_number, (start, stop) in enumerate(
        # Call `zip`; the following indented continuation lines provide its arguments.
        zip(segment_starts, segment_stops)
    # Begin the indented block controlled by this statement.
    ):
        # Build the plot options used for this continuous time segment.
        segment_kwargs = dict(kwargs)
        # Evaluate this condition and run the following indented block only when it is true.
        if segment_number > 0:
            # Call `segment_kwargs.pop`; the following indented continuation lines provide its arguments.
            segment_kwargs.pop("label", None)
        # Call `plt.plot`; the following indented continuation lines provide its arguments.
        plt.plot(time[start:stop], values[start:stop], **segment_kwargs)


# 1. Show the exact role allocation in sheets 1-4.
workbook = find_workbook(HERE)
# Repeat the following indented block once for each item in this iterable.
for sheet_number, sheet in enumerate(TRAIN_ONLY_SHEETS + [MIXED_SHEET], 1):
    # Use the expression `time, current, _ = load_sheet(workbook, sheet)` as the next part of the surrounding Python statement.
    time, current, _ = load_sheet(workbook, sheet)
    # Evaluate `split_table[split_table["sheet"] == sheet]` and store the result in `rows` for the following steps.
    rows = split_table[split_table["sheet"] == sheet]
    # Call `plt.figure`; the following indented continuation lines provide its arguments.
    plt.figure(figsize=(12, 4))
    # Track legend labels already shown so repeated blocks do not duplicate legend entries.
    used_labels = set()
    # Repeat the following indented block once for each item in this iterable.
    for _, row in rows.iterrows():
        # Store the first sample index of the current block or plotted segment.
        start = int(row["start_sample"])
        # Store the exclusive ending index of the current block or plotted segment.
        stop = int(row["stop_sample"])
        # Store whether the current block is training, validation, test, pure test, or diagnostic.
        role = str(row["role"])
        # Evaluate this condition and run the following indented block only when it is true.
        if role == "unused_short_remainder":
            # Skip the remaining statements in this loop iteration and continue with the next item.
            continue
        # Call `plt.plot`; the following indented continuation lines provide its arguments.
        plt.plot(
            # Select `time[start:stop]` from the current array, tensor, table, or dictionary.
            time[start:stop],
            # Select `current[start:stop]` from the current array, tensor, table, or dictionary.
            current[start:stop],
            # Choose the visual color for this plotted series; it has no effect on numerical results.
            color=ROLE_COLORS[role],
            # Set the visual line thickness; it changes presentation only.
            linewidth=1.1,
            # Set the legend text that identifies this plotted series.
            label=role if role not in used_labels else None,
        # Close the current function call, tuple, or grouped expression.
        )
        # Call `used_labels.add`; the following indented continuation lines provide its arguments.
        used_labels.add(role)
    # Call `plt.xlabel`; the following indented continuation lines provide its arguments.
    plt.xlabel("Time (s)")
    # Call `plt.ylabel`; the following indented continuation lines provide its arguments.
    plt.ylabel("Coil current (A)")
    # Call `plt.title`; the following indented continuation lines provide its arguments.
    plt.title(f"{sheet}: time/frequency-distributed data roles")
    # Call `plt.legend`; the following indented continuation lines provide its arguments.
    plt.legend()
    # Call `finish_figure`; the following indented continuation lines provide its arguments.
    finish_figure(f"01_{sheet_number}_data_split_{sheet}.png")


# 2. Show the complete untouched fifth input.
time, current, _ = load_sheet(workbook, FINAL_TEST_SHEET)
# Call `plt.figure`; the following indented continuation lines provide its arguments.
plt.figure(figsize=(12, 4))
# Call `plt.plot`; the following indented continuation lines provide its arguments.
plt.plot(time, current, color="tab:green", label="Pure test")
# Call `plt.xlabel`; the following indented continuation lines provide its arguments.
plt.xlabel("Time (s)")
# Call `plt.ylabel`; the following indented continuation lines provide its arguments.
plt.ylabel("Coil current (A)")
# Call `plt.title`; the following indented continuation lines provide its arguments.
plt.title(f"{FINAL_TEST_SHEET}: complete untouched final test")
# Call `plt.legend`; the following indented continuation lines provide its arguments.
plt.legend()
# Call `finish_figure`; the following indented continuation lines provide its arguments.
finish_figure("02_untouched_final_test_input.png")


# 3. Show training and validation convergence.
epochs = np.arange(1, len(training_history) + 1)
# Call `plt.figure`; the following indented continuation lines provide its arguments.
plt.figure(figsize=(9, 5))
# Call `plt.plot`; the following indented continuation lines provide its arguments.
plt.plot(epochs, training_history, label="Training", color="tab:blue")
# Call `plt.plot`; the following indented continuation lines provide its arguments.
plt.plot(epochs, validation_history, label="Validation", color="tab:orange")
# Call `plt.axvline`; the following indented continuation lines provide its arguments.
plt.axvline(best_epoch, color="tab:green", linestyle="--", label=f"Best epoch = {best_epoch}")
# Call `plt.xlabel`; the following indented continuation lines provide its arguments.
plt.xlabel("Epoch")
# Call `plt.ylabel`; the following indented continuation lines provide its arguments.
plt.ylabel("Normalized loss")
# Call `plt.title`; the following indented continuation lines provide its arguments.
plt.title("Training and validation loss")
# Call `plt.legend`; the following indented continuation lines provide its arguments.
plt.legend()
# Call `finish_figure`; the following indented continuation lines provide its arguments.
finish_figure("03_training_validation_loss.png")


# 4. Compare measured and predicted internal-test values.
for index, short_name, display_name, unit in OUTPUTS:
    # Call `plt.figure`; the following indented continuation lines provide its arguments.
    plt.figure(figsize=(12, 5))
    # Call `plot_discontinuous`; the following indented continuation lines provide its arguments.
    plot_discontinuous(
        # Pass `development_time` as the next value required by the surrounding call or collection.
        development_time,
        # Select `development_measured[:, index]` from the current array, tensor, table, or dictionary.
        development_measured[:, index],
        # Choose the visual color for this plotted series; it has no effect on numerical results.
        color="tab:green",
        # Set the legend text that identifies this plotted series.
        label="Measured",
        # Set the visual line thickness; it changes presentation only.
        linewidth=1.4,
    # Close the current function call, tuple, or grouped expression.
    )
    # Call `plot_discontinuous`; the following indented continuation lines provide its arguments.
    plot_discontinuous(
        # Pass `development_time` as the next value required by the surrounding call or collection.
        development_time,
        # Select `development_prediction[:, index]` from the current array, tensor, table, or dictionary.
        development_prediction[:, index],
        # Choose the visual color for this plotted series; it has no effect on numerical results.
        color="tab:orange",
        # Choose the line pattern so measured and predicted signals remain distinguishable.
        linestyle="--",
        # Set the legend text that identifies this plotted series.
        label="Predicted",
        # Set the visual line thickness; it changes presentation only.
        linewidth=1.1,
    # Close the current function call, tuple, or grouped expression.
    )
    # Call `plt.xlabel`; the following indented continuation lines provide its arguments.
    plt.xlabel("Time (s)")
    # Call `plt.ylabel`; the following indented continuation lines provide its arguments.
    plt.ylabel(f"{display_name} ({unit})")
    # Call `plt.title`; the following indented continuation lines provide its arguments.
    plt.title(f"{MIXED_SHEET} distributed test: {short_name}")
    # Call `plt.legend`; the following indented continuation lines provide its arguments.
    plt.legend()
    # Call `finish_figure`; the following indented continuation lines provide its arguments.
    finish_figure(f"04_internal_test_{short_name}.png")


# 5. Compare measured and predicted values on the pure test.
for index, short_name, display_name, unit in OUTPUTS:
    # Call `plt.figure`; the following indented continuation lines provide its arguments.
    plt.figure(figsize=(12, 5))
    # Call `plt.plot`; the following indented continuation lines provide its arguments.
    plt.plot(final_time, final_measured[:, index], color="tab:green", label="Measured", linewidth=1.4)
    # Call `plt.plot`; the following indented continuation lines provide its arguments.
    plt.plot(final_time, final_prediction[:, index], color="tab:orange", linestyle="--", label="Predicted", linewidth=1.1)
    # Call `plt.xlabel`; the following indented continuation lines provide its arguments.
    plt.xlabel("Time (s)")
    # Call `plt.ylabel`; the following indented continuation lines provide its arguments.
    plt.ylabel(f"{display_name} ({unit})")
    # Call `plt.title`; the following indented continuation lines provide its arguments.
    plt.title(f"{FINAL_TEST_SHEET} pure test: {short_name}")
    # Call `plt.legend`; the following indented continuation lines provide its arguments.
    plt.legend()
    # Call `finish_figure`; the following indented continuation lines provide its arguments.
    finish_figure(f"05_final_test_{short_name}.png")


# 6. Show pure-test errors and measured-versus-predicted parity.
for index, short_name, display_name, unit in OUTPUTS:
    # Subtract prediction from measurement to obtain the pointwise prediction error.
    error = final_measured[:, index] - final_prediction[:, index]
    # Call `plt.figure`; the following indented continuation lines provide its arguments.
    plt.figure(figsize=(12, 4))
    # Call `plt.plot`; the following indented continuation lines provide its arguments.
    plt.plot(final_time, error)
    # Call `plt.axhline`; the following indented continuation lines provide its arguments.
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    # Call `plt.xlabel`; the following indented continuation lines provide its arguments.
    plt.xlabel("Time (s)")
    # Call `plt.ylabel`; the following indented continuation lines provide its arguments.
    plt.ylabel(f"{display_name} error ({unit})")
    # Call `plt.title`; the following indented continuation lines provide its arguments.
    plt.title(f"Pure-test {short_name} prediction error")
    # Call `finish_figure`; the following indented continuation lines provide its arguments.
    finish_figure(f"06_final_test_{short_name}_error.png")

    # Find the smallest measured or predicted value needed to draw an equal-scale parity plot.
    minimum = min(final_measured[:, index].min(), final_prediction[:, index].min())
    # Find the largest measured or predicted value needed to draw an equal-scale parity plot.
    maximum = max(final_measured[:, index].max(), final_prediction[:, index].max())
    # Call `plt.figure`; the following indented continuation lines provide its arguments.
    plt.figure(figsize=(6, 6))
    # Call `plt.scatter`; the following indented continuation lines provide its arguments.
    plt.scatter(final_measured[:, index], final_prediction[:, index], s=8, alpha=0.35)
    # Call `plt.plot`; the following indented continuation lines provide its arguments.
    plt.plot([minimum, maximum], [minimum, maximum], "k--", label="Ideal prediction")
    # Call `plt.xlabel`; the following indented continuation lines provide its arguments.
    plt.xlabel(f"Measured {display_name} ({unit})")
    # Call `plt.ylabel`; the following indented continuation lines provide its arguments.
    plt.ylabel(f"Predicted {display_name} ({unit})")
    # Call `plt.title`; the following indented continuation lines provide its arguments.
    plt.title(f"Pure-test {display_name}: parity")
    # Call `plt.legend`; the following indented continuation lines provide its arguments.
    plt.legend()
    # Call `finish_figure`; the following indented continuation lines provide its arguments.
    finish_figure(f"07_final_test_{short_name}_parity.png")


# 7. Put both independent evaluations in one metrics table.
display_metrics = metrics.copy()
# Repeat the following indented block once for each item in this iterable.
for column in ["MSE", "RMSE", "MAE", "R2", "Fit_percent"]:
    # Use the expression `display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.5g}")` as the next part of the surrounding Python statement.
    display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.5g}")
# Call `plt.figure`; the following indented continuation lines provide its arguments.
plt.figure(figsize=(14, 3.2))
# Call `plt.axis`; the following indented continuation lines provide its arguments.
plt.axis("off")
# Store the pandas table read or assembled in this block.
table = plt.table(
    # Supply the already formatted values displayed in the plotted table.
    cellText=display_metrics.values,
    # Supply the displayed column headings for the plotted table.
    colLabels=display_metrics.columns,
    # Choose the table or legend position within the figure.
    loc="center",
    # Choose horizontal alignment for text inside table cells.
    cellLoc="center",
# Close the current function call, tuple, or grouped expression.
)
# Call `table.auto_set_font_size`; the following indented continuation lines provide its arguments.
table.auto_set_font_size(False)
# Call `table.set_fontsize`; the following indented continuation lines provide its arguments.
table.set_fontsize(7.5)
# Call `table.scale`; the following indented continuation lines provide its arguments.
table.scale(1, 1.5)
# Call `plt.title`; the following indented continuation lines provide its arguments.
plt.title("Internal fourth-series test and untouched fifth-series test metrics", pad=18)
# Call `plt.tight_layout`; the following indented continuation lines provide its arguments.
plt.tight_layout()
# Call `plt.savefig`; the following indented continuation lines provide its arguments.
plt.savefig(FIGURES_FOLDER / "08_all_test_metrics.png", dpi=180, bbox_inches="tight")
# Close the figure after saving it so memory is released.
plt.close()


# 8. Calculate accuracy over lower, middle, and higher chirp regions.
region_rows = []
# Repeat the following indented block once for each item in this iterable.
for region_name, indices in zip(
    # Begin the grouped expression or collection continued on the following lines.
    ["Beginning / lower frequency", "Middle / medium frequency", "End / higher frequency"],
    # Call `np.array_split`; the following indented continuation lines provide its arguments.
    np.array_split(np.arange(len(final_time)), 3),
# Begin the indented block controlled by this statement.
):
    # Repeat the following indented block once for each item in this iterable.
    for index, _, display_name, unit in OUTPUTS:
        # Subtract prediction from measurement to obtain the pointwise prediction error.
        error = final_measured[indices, index] - final_prediction[indices, index]
        # Calculate the reference signal range or energy used to normalize an error metric.
        denominator = np.sum(
            # Begin the grouped expression or collection continued on the following lines.
            (final_measured[indices, index] - final_measured[indices, index].mean()) ** 2
        # Close the current function call, tuple, or grouped expression.
        )
        # Calculate the coefficient of determination relative to variation around the measured mean.
        r2 = 1 - np.sum(error ** 2) / denominator if denominator > 0 else np.nan
        # Call `region_rows.append`; the following indented continuation lines provide its arguments.
        region_rows.append(
            # Begin the grouped expression or collection continued on the following lines.
            {
                # Store the 'Region' field in the current dictionary.
                "Region": region_name,
                # Store the 'Output' field in the current dictionary.
                "Output": display_name,
                # Store the 'Unit' field in the current dictionary.
                "Unit": unit,
                # Store the 'RMSE' field in the current dictionary.
                "RMSE": float(np.sqrt(np.mean(error ** 2))),
                # Store the 'R2' field in the current dictionary.
                "R2": float(r2),
            # Close the current dictionary.
            }
        # Close the current function call, tuple, or grouped expression.
        )

# Calculate metrics separately for the requested time/frequency region.
region_metrics = pd.DataFrame(region_rows)
# Call `region_metrics.to_csv`; the following indented continuation lines provide its arguments.
region_metrics.to_csv(RESULTS_FOLDER / "frequency_region_metrics.csv", index=False)

# Locate the report's figure folder so selected outputs can be copied there.
report_folder = HERE.parent / "Overleaf_Report" / "figures" / "Test1"
# Call `report_folder.mkdir`; the following indented continuation lines provide its arguments.
report_folder.mkdir(parents=True, exist_ok=True)
# Repeat the following indented block once for each item in this iterable.
for source_name, report_name in {
    # Store the '03_training_validation_loss.png' field in the current dictionary.
    "03_training_validation_loss.png": "test1_convergence.png",
    # Store the '05_final_test_displacement.png' field in the current dictionary.
    "05_final_test_displacement.png": "test1_pure_test_displacement.png",
    # Store the '05_final_test_force.png' field in the current dictionary.
    "05_final_test_force.png": "test1_pure_test_force.png",
    # Store the '08_all_test_metrics.png' field in the current dictionary.
    "08_all_test_metrics.png": "test1_metrics.png",
# Begin the indented block controlled by this statement.
}.items():
    # Call `shutil.copy2`; the following indented continuation lines provide its arguments.
    shutil.copy2(FIGURES_FOLDER / source_name, report_folder / report_name)

# Print this progress or result message in the terminal.
print("\nFigures saved in:", FIGURES_FOLDER)
# Repeat the following indented block once for each item in this iterable.
for figure_file in sorted(FIGURES_FOLDER.glob("*.png")):
    # Print this progress or result message in the terminal.
    print(" -", figure_file.name)
