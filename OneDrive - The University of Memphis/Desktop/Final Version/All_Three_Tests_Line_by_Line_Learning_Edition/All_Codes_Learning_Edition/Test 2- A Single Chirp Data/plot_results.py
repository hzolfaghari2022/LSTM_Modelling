# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Create figures for the one-series simulation."""

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
from config import SINGLE_SERIES_SHEET
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

# Load the time values for distributed held-out blocks.
test_time = results["test_time"]
# Load the measured outputs for distributed held-out blocks.
test_measured = results["test_measured"]
# Load the corresponding predictions for the distributed held-out blocks.
test_prediction = results["test_prediction"]
# Load the complete record's time column for the descriptive full-record plot.
all_time = results["all_time"]
# Load the complete record's measured outputs.
all_measured = results["all_measured"]
# Load predictions over the complete record.
all_prediction = results["all_prediction"]
# Load the original time base before window alignment.
raw_time = results["raw_time"]
# Load original measured signals before window alignment.
raw_measured = results["raw_measured"]
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
    """Do not connect separate test blocks with artificial lines."""

    # Set the time-gap threshold used to separate noncontiguous held-out blocks in a plot.
    gap_threshold = 3.0 * np.median(np.diff(time))
    # Find the first index of each continuous held-out segment.
    starts = np.r_[0, np.where(np.diff(time) > gap_threshold)[0] + 1]
    # Find the exclusive final index of each continuous held-out segment.
    stops = np.r_[starts[1:], len(time)]
    # Repeat the following indented block once for each item in this iterable.
    for segment_number, (start, stop) in enumerate(zip(starts, stops)):
        # Build the plot options used for this continuous time segment.
        segment_kwargs = dict(kwargs)
        # Evaluate this condition and run the following indented block only when it is true.
        if segment_number > 0:
            # Call `segment_kwargs.pop`; the following indented continuation lines provide its arguments.
            segment_kwargs.pop("label", None)
        # Call `plt.plot`; the following indented continuation lines provide its arguments.
        plt.plot(time[start:stop], values[start:stop], **segment_kwargs)


# 1. Show the exact distributed role assignment.
workbook = find_workbook(HERE)
# Use the expression `time, current, _ = load_sheet(workbook, SINGLE_SERIES_SHEET)` as the next part of the surrounding Python statement.
time, current, _ = load_sheet(workbook, SINGLE_SERIES_SHEET)
# Call `plt.figure`; the following indented continuation lines provide its arguments.
plt.figure(figsize=(12, 4))
# Track legend labels already shown so repeated blocks do not duplicate legend entries.
used_labels = set()
# Repeat the following indented block once for each item in this iterable.
for _, row in split_table.iterrows():
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
plt.title(f"{SINGLE_SERIES_SHEET}: train/validation/test across the chirp")
# Call `plt.legend`; the following indented continuation lines provide its arguments.
plt.legend()
# Call `finish_figure`; the following indented continuation lines provide its arguments.
finish_figure("01_distributed_data_split.png")


# 2. Show training and validation convergence.
epochs = np.arange(1, len(training_history) + 1)
# Call `plt.figure`; the following indented continuation lines provide its arguments.
plt.figure(figsize=(9, 5))
# Call `plt.plot`; the following indented continuation lines provide its arguments.
plt.plot(epochs, training_history, color="tab:blue", label="Training")
# Call `plt.plot`; the following indented continuation lines provide its arguments.
plt.plot(epochs, validation_history, color="tab:orange", label="Validation")
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
finish_figure("02_training_validation_loss.png")


# 3. Show prediction over the complete selected record.
figure, axes = plt.subplots(
    # Use the expression `2` as the next part of the surrounding Python statement.
    2,
    # Use the expression `1` as the next part of the surrounding Python statement.
    1,
    # Set the figure dimensions in inches; this changes readability only, not model training or metrics.
    figsize=(14, 8),
    # Share the horizontal scale across panels so time alignment is directly comparable.
    sharex=True,
# Close the current function call, tuple, or grouped expression.
)

# Repeat the following indented block once for each item in this iterable.
for axis, (index, _, display_name, unit) in zip(axes, OUTPUTS):
    # Call `axis.plot`; the following indented continuation lines provide its arguments.
    axis.plot(
        # Pass `raw_time` as the next value required by the surrounding call or collection.
        raw_time,
        # Select `raw_measured[:, index]` from the current array, tensor, table, or dictionary.
        raw_measured[:, index],
        # Choose the visual color for this plotted series; it has no effect on numerical results.
        color="tab:green",
        # Set the legend text that identifies this plotted series.
        label="Measured",
        # Set the visual line thickness; it changes presentation only.
        linewidth=1.4,
    # Close the current function call, tuple, or grouped expression.
    )
    # Call `axis.plot`; the following indented continuation lines provide its arguments.
    axis.plot(
        # Pass `all_time` as the next value required by the surrounding call or collection.
        all_time,
        # Select `all_prediction[:, index]` from the current array, tensor, table, or dictionary.
        all_prediction[:, index],
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
    # Call `axis.axvspan`; the following indented continuation lines provide its arguments.
    axis.axvspan(
        # Select `raw_time[0]` from the current array, tensor, table, or dictionary.
        raw_time[0],
        # Select `all_time[0]` from the current array, tensor, table, or dictionary.
        all_time[0],
        # Choose the visual color for this plotted series; it has no effect on numerical results.
        color="0.75",
        # Set drawing transparency between 0 and 1; this is only a visualization setting.
        alpha=0.35,
        # Set the legend text that identifies this plotted series.
        label="120-sample history",
    # Close the current function call, tuple, or grouped expression.
    )
    # Call `axis.set_ylabel`; the following indented continuation lines provide its arguments.
    axis.set_ylabel(f"{display_name} ({unit})")
    # Call `axis.grid`; the following indented continuation lines provide its arguments.
    axis.grid(True, alpha=0.3)
    # Call `axis.legend`; the following indented continuation lines provide its arguments.
    axis.legend()

# Use the expression `axes[-1].set_xlabel("Time (s)")` as the next part of the surrounding Python statement.
axes[-1].set_xlabel("Time (s)")
# Call `figure.suptitle`; the following indented continuation lines provide its arguments.
figure.suptitle(
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    f"{SINGLE_SERIES_SHEET}: complete record "
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "(prediction follows the initial history window)"
# Close the current function call, tuple, or grouped expression.
)
# Call `figure.tight_layout`; the following indented continuation lines provide its arguments.
figure.tight_layout(rect=(0, 0, 1, 0.96))
# Call `figure.savefig`; the following indented continuation lines provide its arguments.
figure.savefig(
    # Use the expression `FIGURES_FOLDER / "03_full_record_prediction_all_outputs.png"` as the next part of the surrounding Python statement.
    FIGURES_FOLDER / "03_full_record_prediction_all_outputs.png",
    # Set saved-image resolution in dots per inch; this is a presentation choice with no effect on accuracy.
    dpi=180,
    # Use a tight saved-image boundary so labels are not surrounded by unnecessary whitespace.
    bbox_inches="tight",
# Close the current function call, tuple, or grouped expression.
)
# Close the figure after saving it so memory is released.
plt.close(figure)


# 4. Show held-out predictions, errors, and parity.
for index, short_name, display_name, unit in OUTPUTS:
    # Call `plt.figure`; the following indented continuation lines provide its arguments.
    plt.figure(figsize=(12, 5))
    # Call `plot_discontinuous`; the following indented continuation lines provide its arguments.
    plot_discontinuous(
        # Pass `test_time` as the next value required by the surrounding call or collection.
        test_time,
        # Select `test_measured[:, index]` from the current array, tensor, table, or dictionary.
        test_measured[:, index],
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
        # Pass `test_time` as the next value required by the surrounding call or collection.
        test_time,
        # Select `test_prediction[:, index]` from the current array, tensor, table, or dictionary.
        test_prediction[:, index],
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
    plt.title(f"Distributed test blocks: {short_name}")
    # Call `plt.legend`; the following indented continuation lines provide its arguments.
    plt.legend()
    # Call `finish_figure`; the following indented continuation lines provide its arguments.
    finish_figure(f"04_test_{short_name}_tracking.png")

    # Subtract prediction from measurement to obtain the pointwise prediction error.
    error = test_measured[:, index] - test_prediction[:, index]
    # Call `plt.figure`; the following indented continuation lines provide its arguments.
    plt.figure(figsize=(12, 4))
    # Call `plot_discontinuous`; the following indented continuation lines provide its arguments.
    plot_discontinuous(test_time, error, color="tab:blue")
    # Call `plt.axhline`; the following indented continuation lines provide its arguments.
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    # Call `plt.xlabel`; the following indented continuation lines provide its arguments.
    plt.xlabel("Time (s)")
    # Call `plt.ylabel`; the following indented continuation lines provide its arguments.
    plt.ylabel(f"{display_name} error ({unit})")
    # Call `plt.title`; the following indented continuation lines provide its arguments.
    plt.title(f"Distributed test error: {short_name}")
    # Call `finish_figure`; the following indented continuation lines provide its arguments.
    finish_figure(f"05_test_{short_name}_error.png")

    # Find the smallest measured or predicted value needed to draw an equal-scale parity plot.
    minimum = min(test_measured[:, index].min(), test_prediction[:, index].min())
    # Find the largest measured or predicted value needed to draw an equal-scale parity plot.
    maximum = max(test_measured[:, index].max(), test_prediction[:, index].max())
    # Call `plt.figure`; the following indented continuation lines provide its arguments.
    plt.figure(figsize=(6, 6))
    # Call `plt.scatter`; the following indented continuation lines provide its arguments.
    plt.scatter(
        # Select `test_measured[:, index]` from the current array, tensor, table, or dictionary.
        test_measured[:, index],
        # Select `test_prediction[:, index]` from the current array, tensor, table, or dictionary.
        test_prediction[:, index],
        # Set scatter-marker area for readability; this is only a visualization setting.
        s=12,
        # Set drawing transparency between 0 and 1; this is only a visualization setting.
        alpha=0.45,
    # Close the current function call, tuple, or grouped expression.
    )
    # Call `plt.plot`; the following indented continuation lines provide its arguments.
    plt.plot([minimum, maximum], [minimum, maximum], "k--", label="Ideal prediction")
    # Call `plt.xlabel`; the following indented continuation lines provide its arguments.
    plt.xlabel(f"Measured {display_name} ({unit})")
    # Call `plt.ylabel`; the following indented continuation lines provide its arguments.
    plt.ylabel(f"Predicted {display_name} ({unit})")
    # Call `plt.title`; the following indented continuation lines provide its arguments.
    plt.title(f"Test parity: {display_name}")
    # Call `plt.legend`; the following indented continuation lines provide its arguments.
    plt.legend()
    # Call `finish_figure`; the following indented continuation lines provide its arguments.
    finish_figure(f"06_test_{short_name}_parity.png")


# 5. Put held-out and complete-record metrics in one table.
display_metrics = metrics.copy()
# Repeat the following indented block once for each item in this iterable.
for column in ["MSE", "RMSE", "MAE", "R2", "Fit_percent"]:
    # Use the expression `display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.5g}")` as the next part of the surrounding Python statement.
    display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.5g}")
# Call `plt.figure`; the following indented continuation lines provide its arguments.
plt.figure(figsize=(17, 4.0))
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
    # Set relative table-column widths for readability only.
    colWidths=[0.28, 0.12, 0.12, 0.06, 0.08, 0.08, 0.08, 0.08, 0.10],
# Close the current function call, tuple, or grouped expression.
)
# Call `table.auto_set_font_size`; the following indented continuation lines provide its arguments.
table.auto_set_font_size(False)
# Call `table.set_fontsize`; the following indented continuation lines provide its arguments.
table.set_fontsize(7.5)
# Call `table.scale`; the following indented continuation lines provide its arguments.
table.scale(1, 1.5)
# Call `plt.title`; the following indented continuation lines provide its arguments.
plt.title(
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    f"{SINGLE_SERIES_SHEET}: held-out test and complete-record metrics",
    # Set layout padding between panels and figure edges; this has no effect on the simulation.
    pad=18,
# Close the current function call, tuple, or grouped expression.
)
# Call `plt.tight_layout`; the following indented continuation lines provide its arguments.
plt.tight_layout()
# Call `plt.savefig`; the following indented continuation lines provide its arguments.
plt.savefig(FIGURES_FOLDER / "07_metrics.png", dpi=180, bbox_inches="tight")
# Close the figure after saving it so memory is released.
plt.close()

# Locate the report's figure folder so selected outputs can be copied there.
report_folder = HERE.parent / "Overleaf_Report" / "figures" / "Test2"
# Call `report_folder.mkdir`; the following indented continuation lines provide its arguments.
report_folder.mkdir(parents=True, exist_ok=True)
# Repeat the following indented block once for each item in this iterable.
for source_name, report_name in {
    # Store the '02_training_validation_loss.png' field in the current dictionary.
    "02_training_validation_loss.png": "test2_convergence.png",
    # Store the '03_full_record_prediction_all_outputs.png' field in the current dictionary.
    "03_full_record_prediction_all_outputs.png": "test2_full_record_prediction.png",
    # Store the '04_test_displacement_tracking.png' field in the current dictionary.
    "04_test_displacement_tracking.png": "test2_heldout_displacement.png",
    # Store the '04_test_force_tracking.png' field in the current dictionary.
    "04_test_force_tracking.png": "test2_heldout_force.png",
    # Store the '07_metrics.png' field in the current dictionary.
    "07_metrics.png": "test2_metrics.png",
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
