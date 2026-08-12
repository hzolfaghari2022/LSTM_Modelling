"""Create figures for the one-series simulation."""

import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SINGLE_SERIES_SHEET
from data_utils import find_workbook, load_sheet


HERE = Path(__file__).resolve().parent
RESULTS_FOLDER = HERE / "ResultsData"
FIGURES_FOLDER = HERE / "FiguresResults"
FIGURES_FOLDER.mkdir(exist_ok=True)

results = np.load(RESULTS_FOLDER / "simulation_results.npz")
metrics = pd.read_csv(RESULTS_FOLDER / "metrics.csv")
split_table = pd.read_csv(RESULTS_FOLDER / "data_split.csv")

test_time = results["test_time"]
test_measured = results["test_measured"]
test_prediction = results["test_prediction"]
all_time = results["all_time"]
all_measured = results["all_measured"]
all_prediction = results["all_prediction"]
raw_time = results["raw_time"]
raw_measured = results["raw_measured"]
training_history = results["training_history"]
validation_history = results["validation_history"]
best_epoch = int(results["best_epoch"])

ROLE_COLORS = {
    "training": "tab:blue",
    "validation": "tab:orange",
    "test": "tab:green",
    "unused_short_remainder": "0.75",
}
OUTPUTS = [
    (0, "displacement", "Displacement", "mm"),
    (1, "force", "Lorentz force", "N"),
]


def finish_figure(file_name):
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_FOLDER / file_name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_discontinuous(time, values, **kwargs):
    """Do not connect separate test blocks with artificial lines."""

    gap_threshold = 3.0 * np.median(np.diff(time))
    starts = np.r_[0, np.where(np.diff(time) > gap_threshold)[0] + 1]
    stops = np.r_[starts[1:], len(time)]
    for segment_number, (start, stop) in enumerate(zip(starts, stops)):
        segment_kwargs = dict(kwargs)
        if segment_number > 0:
            segment_kwargs.pop("label", None)
        plt.plot(time[start:stop], values[start:stop], **segment_kwargs)


# 1. Exact distributed role assignment.
workbook = find_workbook(HERE)
time, current, _ = load_sheet(workbook, SINGLE_SERIES_SHEET)
plt.figure(figsize=(12, 4))
used_labels = set()
for _, row in split_table.iterrows():
    start = int(row["start_sample"])
    stop = int(row["stop_sample"])
    role = str(row["role"])
    if role == "unused_short_remainder":
        continue
    plt.plot(
        time[start:stop],
        current[start:stop],
        color=ROLE_COLORS[role],
        linewidth=1.1,
        label=role if role not in used_labels else None,
    )
    used_labels.add(role)
plt.xlabel("Time (s)")
plt.ylabel("Coil current (A)")
plt.title(f"{SINGLE_SERIES_SHEET}: train/validation/test across the chirp")
plt.legend()
finish_figure("01_distributed_data_split.png")


# 2. Training and validation history.
epochs = np.arange(1, len(training_history) + 1)
plt.figure(figsize=(9, 5))
plt.plot(epochs, training_history, color="tab:blue", label="Training")
plt.plot(epochs, validation_history, color="tab:orange", label="Validation")
plt.axvline(best_epoch, color="tab:green", linestyle="--", label=f"Best epoch = {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("Normalized loss")
plt.title("Training and validation loss")
plt.legend()
finish_figure("02_training_validation_loss.png")


# 3. Prediction over the complete selected record.
figure, axes = plt.subplots(
    2,
    1,
    figsize=(14, 8),
    sharex=True,
)

for axis, (index, _, display_name, unit) in zip(axes, OUTPUTS):
    axis.plot(
        raw_time,
        raw_measured[:, index],
        color="tab:green",
        label="Measured",
        linewidth=1.4,
    )
    axis.plot(
        all_time,
        all_prediction[:, index],
        color="tab:orange",
        linestyle="--",
        label="Predicted",
        linewidth=1.1,
    )
    axis.axvspan(
        raw_time[0],
        all_time[0],
        color="0.75",
        alpha=0.35,
        label="120-sample history",
    )
    axis.set_ylabel(f"{display_name} ({unit})")
    axis.grid(True, alpha=0.3)
    axis.legend()

axes[-1].set_xlabel("Time (s)")
figure.suptitle(
    f"{SINGLE_SERIES_SHEET}: complete record "
    "(prediction follows the initial history window)"
)
figure.tight_layout(rect=(0, 0, 1, 0.96))
figure.savefig(
    FIGURES_FOLDER / "03_full_record_prediction_all_outputs.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(figure)


# 4. Measured/predicted held-out test blocks, errors, and parity.
for index, short_name, display_name, unit in OUTPUTS:
    plt.figure(figsize=(12, 5))
    plot_discontinuous(
        test_time,
        test_measured[:, index],
        color="tab:green",
        label="Measured",
        linewidth=1.4,
    )
    plot_discontinuous(
        test_time,
        test_prediction[:, index],
        color="tab:orange",
        linestyle="--",
        label="Predicted",
        linewidth=1.1,
    )
    plt.xlabel("Time (s)")
    plt.ylabel(f"{display_name} ({unit})")
    plt.title(f"Distributed test blocks: {short_name}")
    plt.legend()
    finish_figure(f"04_test_{short_name}_tracking.png")

    error = test_measured[:, index] - test_prediction[:, index]
    plt.figure(figsize=(12, 4))
    plot_discontinuous(test_time, error, color="tab:blue")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel(f"{display_name} error ({unit})")
    plt.title(f"Distributed test error: {short_name}")
    finish_figure(f"05_test_{short_name}_error.png")

    minimum = min(test_measured[:, index].min(), test_prediction[:, index].min())
    maximum = max(test_measured[:, index].max(), test_prediction[:, index].max())
    plt.figure(figsize=(6, 6))
    plt.scatter(
        test_measured[:, index],
        test_prediction[:, index],
        s=12,
        alpha=0.45,
    )
    plt.plot([minimum, maximum], [minimum, maximum], "k--", label="Ideal prediction")
    plt.xlabel(f"Measured {display_name} ({unit})")
    plt.ylabel(f"Predicted {display_name} ({unit})")
    plt.title(f"Test parity: {display_name}")
    plt.legend()
    finish_figure(f"06_test_{short_name}_parity.png")


# 5. Held-out and complete-record metrics table.
display_metrics = metrics.copy()
for column in ["MSE", "RMSE", "MAE", "R2", "Fit_percent"]:
    display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.5g}")
plt.figure(figsize=(17, 4.0))
plt.axis("off")
table = plt.table(
    cellText=display_metrics.values,
    colLabels=display_metrics.columns,
    loc="center",
    cellLoc="center",
    colWidths=[0.28, 0.12, 0.12, 0.06, 0.08, 0.08, 0.08, 0.08, 0.10],
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.5)
plt.title(
    f"{SINGLE_SERIES_SHEET}: held-out test and complete-record metrics",
    pad=18,
)
plt.tight_layout()
plt.savefig(FIGURES_FOLDER / "07_metrics.png", dpi=180, bbox_inches="tight")
plt.close()

report_folder = HERE.parent / "Overleaf_Report" / "figures" / "Test2"
report_folder.mkdir(parents=True, exist_ok=True)
for source_name, report_name in {
    "02_training_validation_loss.png": "test2_convergence.png",
    "03_full_record_prediction_all_outputs.png": "test2_full_record_prediction.png",
    "04_test_displacement_tracking.png": "test2_heldout_displacement.png",
    "04_test_force_tracking.png": "test2_heldout_force.png",
    "07_metrics.png": "test2_metrics.png",
}.items():
    shutil.copy2(FIGURES_FOLDER / source_name, report_folder / report_name)

print("\nFigures saved in:", FIGURES_FOLDER)
for figure_file in sorted(FIGURES_FOLDER.glob("*.png")):
    print(" -", figure_file.name)
