"""
SIMPLE COMPLETE-TEST FIGURES
============================

Run this after main.py:

    python simple_results_plots.py

This script does not train or change the model.  It only reads the saved
pure-test results and creates three figures that explain them simply.
"""

# ---------------------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------------------

import os
import tempfile
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "farnn-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from config import DOWNSAMPLE_FACTOR, FINAL_TEST_SHEET


HERE = Path(__file__).resolve().parent
RESULTS_FOLDER = HERE / "ResultsData"
FIGURES_FOLDER = HERE / "SimpleResultsFigures"
FIGURES_FOLDER.mkdir(exist_ok=True)

PREDICTIONS_FILE = RESULTS_FOLDER / "final_test_predictions.csv"
METRICS_FILE = RESULTS_FOLDER / "metrics.csv"

for required_file in [PREDICTIONS_FILE, METRICS_FILE]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Missing {required_file.name}. Run main.py first."
        )


# ---------------------------------------------------------------------
# 2. LOAD THE SAVED PURE-TEST RESULTS
# ---------------------------------------------------------------------

predictions = pd.read_csv(PREDICTIONS_FILE)
metrics = pd.read_csv(METRICS_FILE)

time = predictions["time_s"].to_numpy()

measured_displacement = predictions[
    "measured_displacement_mm"
].to_numpy()
predicted_displacement = predictions[
    "predicted_displacement_mm"
].to_numpy()

measured_force = predictions["measured_force_N"].to_numpy()
predicted_force = predictions["predicted_force_N"].to_numpy()

# Load only the fifth-sheet current so the complete test can be understood
# from input to both predicted outputs.  This repeats the same small loading
# rule used by data_utils.py without importing the training libraries.
workbooks = list(HERE.glob("COMSOL_07_13_2026*.xlsx"))

if not workbooks:
    raise FileNotFoundError(
        "Place COMSOL_07_13_2026*.xlsx beside this plotting script."
    )

workbook = max(workbooks, key=lambda file: file.stat().st_mtime)

test_data = pd.read_excel(
    workbook,
    sheet_name=FINAL_TEST_SHEET,
    header=16,
    usecols="A:D",
    engine="openpyxl",
)
test_data.columns = ["time", "displacement", "current", "force"]
test_data = test_data.apply(pd.to_numeric, errors="coerce").dropna()
test_data = (
    test_data.sort_values("time", kind="stable")
    .drop_duplicates("time", keep="last")
    .reset_index(drop=True)
)

raw_time = test_data["time"].to_numpy(np.float32)
raw_current = test_data["current"].to_numpy(np.float32)
input_current = resample_poly(
    raw_current,
    up=1,
    down=DOWNSAMPLE_FACTOR,
).astype(np.float32)
input_time = np.linspace(
    float(raw_time[0]),
    float(raw_time[-1]),
    num=len(input_current),
    dtype=np.float32,
)

# Interpolation aligns the input with prediction timestamps after the
# 120-sample history window.
test_current = np.interp(time, input_time, input_current)


# ---------------------------------------------------------------------
# 3. APPEARANCE
# ---------------------------------------------------------------------

MEASURED_COLOR = "#18864b"
PREDICTED_COLOR = "#e87516"
INPUT_COLOR = "#3268a8"
ERROR_COLOR = "#8d4bbb"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.22,
    }
)


def save_figure(figure, file_name):
    """Save one clear PNG and close it."""

    figure.tight_layout()
    figure.savefig(
        FIGURES_FOLDER / file_name,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


# ---------------------------------------------------------------------
# 4. FIGURE 1 — THE COMPLETE UNTOUCHED TEST IN ONE VIEW
# ---------------------------------------------------------------------

figure, axes = plt.subplots(
    3,
    1,
    figsize=(13, 9),
    sharex=True,
)

axes[0].plot(time, test_current, color=INPUT_COLOR, linewidth=1.2)
axes[0].set_ylabel("Current (A)")
axes[0].set_title("Input given to the model: untouched 147 mA chirp")

axes[1].plot(
    time,
    measured_displacement,
    color=MEASURED_COLOR,
    linewidth=1.5,
    label="COMSOL measured",
)
axes[1].plot(
    time,
    predicted_displacement,
    color=PREDICTED_COLOR,
    linestyle="--",
    linewidth=1.2,
    label="FARNN predicted",
)
axes[1].set_ylabel("Displacement (mm)")
axes[1].set_title("Displacement: measured and predicted over every test sample")
axes[1].legend(loc="best")

axes[2].plot(
    time,
    measured_force,
    color=MEASURED_COLOR,
    linewidth=1.5,
    label="COMSOL measured",
)
axes[2].plot(
    time,
    predicted_force,
    color=PREDICTED_COLOR,
    linestyle="--",
    linewidth=1.2,
    label="FARNN predicted",
)
axes[2].set_ylabel("Lorentz force (N)")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Force: measured and predicted over every test sample")
axes[2].legend(loc="best")

figure.suptitle(
    "Complete pure test — 3,382 predictions from the unseen 147 mA dataset\n"
    "Prediction starts at 0.238 s because the model first needs a 120-sample history",
    fontsize=14,
)
figure.subplots_adjust(top=0.89)

save_figure(figure, "01_complete_test_overview.png")


# ---------------------------------------------------------------------
# 5. FIGURE 2 — WHERE THE PREDICTION IS WRONG
# ---------------------------------------------------------------------

displacement_absolute_error = np.abs(
    measured_displacement - predicted_displacement
)
force_absolute_error = np.abs(measured_force - predicted_force)

displacement_mae = displacement_absolute_error.mean()
force_mae = force_absolute_error.mean()

figure, axes = plt.subplots(
    2,
    1,
    figsize=(13, 6.5),
    sharex=True,
)

axes[0].plot(
    time,
    displacement_absolute_error,
    color=ERROR_COLOR,
    linewidth=1.1,
)
axes[0].axhline(
    displacement_mae,
    color=PREDICTED_COLOR,
    linestyle="--",
    label=f"Mean absolute error = {displacement_mae:.3f} mm",
)
axes[0].set_ylabel("Absolute error (mm)")
axes[0].set_title("Displacement error across the complete test")
axes[0].legend(loc="upper right")

axes[1].plot(
    time,
    force_absolute_error,
    color=ERROR_COLOR,
    linewidth=1.1,
)
axes[1].axhline(
    force_mae,
    color=PREDICTED_COLOR,
    linestyle="--",
    label=f"Mean absolute error = {force_mae:.5f} N",
)
axes[1].set_ylabel("Absolute error (N)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Force error across the complete test")
axes[1].legend(loc="upper right")

figure.suptitle(
    "Smaller values mean better prediction",
    fontsize=14,
)
figure.subplots_adjust(top=0.88)

save_figure(figure, "02_complete_test_absolute_error.png")


# ---------------------------------------------------------------------
# 6. FIGURE 3 — SIMPLE NUMERICAL SUMMARY
# ---------------------------------------------------------------------

labels = metrics["Output"].tolist()
r2_percent = 100.0 * metrics["R2"].to_numpy()
fit_percent = metrics["Fit_percent"].to_numpy()
bar_colors = ["#3268a8", "#18864b"]

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for axis, values, title in [
    (axes[0], r2_percent, "R²: variation explained by the model"),
    (axes[1], fit_percent, "Model fit"),
]:
    bars = axis.bar(labels, values, color=bar_colors, width=0.55)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Percent (%)")
    axis.set_title(title)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

figure.suptitle(
    "Pure 147 mA test: force is predicted more accurately than displacement",
    fontsize=14,
)
figure.subplots_adjust(top=0.82)

save_figure(figure, "03_complete_test_accuracy_summary.png")


# ---------------------------------------------------------------------
# 7. PRINT THE SAME RESULTS IN PLAIN LANGUAGE
# ---------------------------------------------------------------------

print("\nSimple figures saved in:")
print(FIGURES_FOLDER)

for _, row in metrics.iterrows():
    print(
        f"{row['Output']}: "
        f"RMSE = {row['RMSE']:.6g} {row['Unit']}, "
        f"R2 = {100 * row['R2']:.1f}%, "
        f"fit = {row['Fit_percent']:.1f}%"
    )

for figure_file in sorted(FIGURES_FOLDER.glob("*.png")):
    print(" -", figure_file.name)
