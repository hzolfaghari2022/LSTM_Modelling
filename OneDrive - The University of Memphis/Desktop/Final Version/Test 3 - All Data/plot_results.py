"""
SEPARATE PLOTTING SCRIPT
========================

Run after main.py:

    python plot_results.py

The numerical files remain in:
    ResultsData

All figures are saved in:
    FiguresResults
"""

# ---------------------------------------------------------------------
# 1. WINDOWS / CONDA FIX
# ---------------------------------------------------------------------

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------
# 2. IMPORTS
# ---------------------------------------------------------------------

from pathlib import Path

import matplotlib

# Save figures without opening a Qt window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DEVELOPMENT_SHEETS, FINAL_TEST_SHEET
from data_utils import find_workbook, load_sheet

TRAINING_COLOR = "tab:blue"
VALIDATION_COLOR = "tab:orange"
TEST_COLOR = "tab:green"
MEASURED_COLOR = "tab:green"
PREDICTED_COLOR = "tab:orange"


# ---------------------------------------------------------------------
# 3. FOLDERS AND FILE CHECKS
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
RESULTS_FOLDER = HERE / "ResultsData"
FIGURES_FOLDER = HERE / "FiguresResults"

FIGURES_FOLDER.mkdir(exist_ok=True)

results_file = RESULTS_FOLDER / "simulation_results.npz"
metrics_file = RESULTS_FOLDER / "metrics.csv"
split_file = RESULTS_FOLDER / "data_split.csv"

for required_file in [results_file, metrics_file, split_file]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Missing {required_file.name}. Run main.py first."
        )


# ---------------------------------------------------------------------
# 4. LOAD THE SAVED SIMULATION
# ---------------------------------------------------------------------

results = np.load(results_file)
metrics = pd.read_csv(metrics_file)
split_table = pd.read_csv(split_file)

test_time = results["test_time"]
measured = results["measured"]
prediction = results["prediction"]

training_history = results["training_history"]
validation_history = results["validation_history"]
best_epoch = int(results["best_epoch"])


# ---------------------------------------------------------------------
# 5. HELPER FOR SAVING ONE FIGURE
# ---------------------------------------------------------------------

def finish_figure(file_name):
    """Format, save, and close the current figure."""

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_FOLDER / file_name,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# 6. EXACT TRAINING / VALIDATION SPLIT FOR EACH DEVELOPMENT SHEET
# ---------------------------------------------------------------------

workbook = find_workbook(HERE)

for sheet_number, sheet in enumerate(
    DEVELOPMENT_SHEETS,
    start=1,
):
    time, current, _ = load_sheet(
        workbook,
        sheet,
    )

    rows = split_table[
        split_table["sheet"] == sheet
    ]

    plt.figure(figsize=(12, 4))

    used_labels = set()

    for _, row in rows.iterrows():
        start = int(row["start_sample"])
        stop = int(row["stop_sample"])
        role = str(row["role"])

        label = (
            role
            if role not in used_labels
            else None
        )

        role_color = (
            TRAINING_COLOR
            if role == "training"
            else VALIDATION_COLOR
        )

        plt.plot(
            time[start:stop],
            current[start:stop],
            color=role_color,
            label=label,
            linewidth=1.1,
        )

        used_labels.add(role)

    plt.xlabel("Time (s)")
    plt.ylabel("Coil current (A)")
    plt.title(
        f"{sheet}: distributed training and validation blocks"
    )
    plt.legend()

    finish_figure(
        f"01_{sheet_number}_data_split_{sheet}.png"
    )


# ---------------------------------------------------------------------
# 7. UNTOUCHED FIFTH-SHEET INPUT
# ---------------------------------------------------------------------

time, current, _ = load_sheet(
    workbook,
    FINAL_TEST_SHEET,
)

plt.figure(figsize=(12, 4))

plt.plot(
    time,
    current,
    color=TEST_COLOR,
    label="Pure test",
)

plt.xlabel("Time (s)")
plt.ylabel("Coil current (A)")
plt.title(
    f"{FINAL_TEST_SHEET}: complete untouched final test"
)
plt.legend()

finish_figure(
    "02_final_test_input_147mA.png"
)


# ---------------------------------------------------------------------
# 8. TRAINING AND VALIDATION LOSS
# ---------------------------------------------------------------------

epochs = np.arange(
    1,
    len(training_history) + 1,
)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    training_history,
    color=TRAINING_COLOR,
    label="Training",
)

plt.plot(
    epochs,
    validation_history,
    color=VALIDATION_COLOR,
    label="Validation",
)

plt.axvline(
    best_epoch,
    color=TEST_COLOR,
    linestyle="--",
    label=f"Best epoch = {best_epoch}",
)

plt.xlabel("Epoch")
plt.ylabel("Normalized MSE")
plt.title("Training and validation loss")
plt.legend()

finish_figure(
    "03_training_validation_loss.png"
)


# ---------------------------------------------------------------------
# 9. FULL-SIGNAL TRACKING FIGURES
# ---------------------------------------------------------------------

output_information = [
    (0, "displacement", "Displacement", "mm"),
    (1, "force", "Lorentz force", "N"),
]

for index, short_name, display_name, unit in output_information:
    measured_signal = measured[:, index]
    predicted_signal = prediction[:, index]

    plt.figure(figsize=(12, 5))

    plt.plot(
        test_time,
        measured_signal,
        color=MEASURED_COLOR,
        label="Measured",
        linewidth=1.4,
    )

    plt.plot(
        test_time,
        predicted_signal,
        color=PREDICTED_COLOR,
        linestyle="--",
        label="Predicted",
        linewidth=1.1,
    )

    plt.xlabel("Time (s)")
    plt.ylabel(f"{display_name} ({unit})")
    plt.title(
        f"Pure 147 mA test: measured and predicted {short_name}"
    )
    plt.legend()

    finish_figure(
        f"04_{short_name}_tracking_full.png"
    )


# ---------------------------------------------------------------------
# 10. RESONANCE-REGION ZOOM
# ---------------------------------------------------------------------

resonance_mask = (
    test_time >= 1.0
) & (
    test_time <= 3.3
)

for index, short_name, display_name, unit in output_information:
    plt.figure(figsize=(12, 5))

    plt.plot(
        test_time[resonance_mask],
        measured[resonance_mask, index],
        color=MEASURED_COLOR,
        label="Measured",
        linewidth=1.4,
    )

    plt.plot(
        test_time[resonance_mask],
        prediction[resonance_mask, index],
        color=PREDICTED_COLOR,
        linestyle="--",
        label="Predicted",
        linewidth=1.1,
    )

    plt.xlabel("Time (s)")
    plt.ylabel(f"{display_name} ({unit})")
    plt.title(f"Resonance-region zoom: {short_name}")
    plt.legend()

    finish_figure(
        f"05_{short_name}_resonance_zoom.png"
    )


# ---------------------------------------------------------------------
# 11. HIGH-FREQUENCY ZOOM
# The final 20% of the chirp represents the highest-frequency region.
# ---------------------------------------------------------------------

zoom_start = int(
    0.80 * len(test_time)
)

for index, short_name, display_name, unit in output_information:
    plt.figure(figsize=(12, 5))

    plt.plot(
        test_time[zoom_start:],
        measured[zoom_start:, index],
        color=MEASURED_COLOR,
        label="Measured",
        linewidth=1.4,
    )

    plt.plot(
        test_time[zoom_start:],
        prediction[zoom_start:, index],
        color=PREDICTED_COLOR,
        linestyle="--",
        label="Predicted",
        linewidth=1.1,
    )

    plt.xlabel("Time (s)")
    plt.ylabel(f"{display_name} ({unit})")
    plt.title(
        f"High-frequency zoom: {short_name}"
    )
    plt.legend()

    finish_figure(
        f"06_{short_name}_high_frequency_zoom.png"
    )


# ---------------------------------------------------------------------
# 11. ERROR VERSUS TIME
# ---------------------------------------------------------------------

for index, short_name, display_name, unit in output_information:
    error = (
        measured[:, index]
        - prediction[:, index]
    )

    plt.figure(figsize=(12, 4))

    plt.plot(
        test_time,
        error,
    )

    plt.axhline(
        0,
        linestyle="--",
        label="Zero error",
    )

    plt.xlabel("Time (s)")
    plt.ylabel(f"{display_name} error ({unit})")
    plt.title(
        f"Pure-test {short_name} prediction error"
    )
    plt.legend()

    finish_figure(
        f"07_{short_name}_error_vs_time.png"
    )


# ---------------------------------------------------------------------
# 12. ERROR HISTOGRAMS
# ---------------------------------------------------------------------

for index, short_name, display_name, unit in output_information:
    error = (
        measured[:, index]
        - prediction[:, index]
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        error,
        bins=60,
    )

    plt.xlabel(f"{display_name} error ({unit})")
    plt.ylabel("Count")
    plt.title(
        f"Distribution of {short_name} prediction errors"
    )

    finish_figure(
        f"08_{short_name}_error_histogram.png"
    )


# ---------------------------------------------------------------------
# 13. MEASURED-VERSUS-PREDICTED PARITY
# ---------------------------------------------------------------------

for index, short_name, display_name, unit in output_information:
    measured_signal = measured[:, index]
    predicted_signal = prediction[:, index]

    minimum = min(
        measured_signal.min(),
        predicted_signal.min(),
    )

    maximum = max(
        measured_signal.max(),
        predicted_signal.max(),
    )

    plt.figure(figsize=(6, 6))

    plt.scatter(
        measured_signal,
        predicted_signal,
        s=8,
        alpha=0.35,
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Ideal prediction",
    )

    plt.xlabel(f"Measured {display_name} ({unit})")
    plt.ylabel(f"Predicted {display_name} ({unit})")
    plt.title(
        f"{display_name}: measured versus predicted"
    )
    plt.legend()

    finish_figure(
        f"09_{short_name}_parity.png"
    )


# ---------------------------------------------------------------------
# 14. METRICS BY CHIRP REGION
# ---------------------------------------------------------------------

def calculate_metrics(
    measured_signal,
    predicted_signal,
):
    error = measured_signal - predicted_signal

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    denominator = float(
        np.sum(
            (
                measured_signal
                - measured_signal.mean()
            ) ** 2
        )
    )

    r2 = (
        1
        - float(np.sum(error ** 2))
        / denominator
        if denominator > 0
        else np.nan
    )

    fit_denominator = float(
        np.linalg.norm(
            measured_signal
            - measured_signal.mean()
        )
    )

    fit = (
        100
        * (
            1
            - float(np.linalg.norm(error))
            / fit_denominator
        )
        if fit_denominator > 0
        else np.nan
    )

    return mse, rmse, mae, r2, fit


region_names = [
    "Beginning / lower frequency",
    "Middle / medium frequency",
    "End / higher frequency",
]

region_indices = np.array_split(
    np.arange(len(test_time)),
    3,
)

region_rows = []

for region_name, indices in zip(
    region_names,
    region_indices,
):
    for index, short_name, display_name, unit in output_information:
        mse, rmse, mae, r2, fit = calculate_metrics(
            measured[indices, index],
            prediction[indices, index],
        )

        region_rows.append(
            {
                "Region": region_name,
                "Output": display_name,
                "Unit": unit,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Fit_percent": fit,
            }
        )

region_metrics = pd.DataFrame(
    region_rows
)

region_metrics.to_csv(
    RESULTS_FOLDER / "frequency_region_metrics.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 15. OVERALL METRICS TABLE FIGURE
# ---------------------------------------------------------------------

display_metrics = metrics.copy()

for column in [
    "MSE",
    "RMSE",
    "MAE",
    "R2",
    "Fit_percent",
]:
    display_metrics[column] = (
        display_metrics[column].map(
            lambda value: f"{value:.5g}"
        )
    )

plt.figure(figsize=(10, 2.5))
plt.axis("off")

table = plt.table(
    cellText=display_metrics.values,
    colLabels=display_metrics.columns,
    loc="center",
    cellLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title(
    "Pure 147 mA test metrics",
    pad=18,
)

plt.tight_layout()

plt.savefig(
    FIGURES_FOLDER / "10_overall_metrics_table.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# 16. FREQUENCY-REGION METRICS TABLE FIGURE
# ---------------------------------------------------------------------

region_display = region_metrics[
    [
        "Region",
        "Output",
        "RMSE",
        "R2",
        "Fit_percent",
    ]
].copy()

for column in [
    "RMSE",
    "R2",
    "Fit_percent",
]:
    region_display[column] = (
        region_display[column].map(
            lambda value: f"{value:.5g}"
        )
    )

plt.figure(figsize=(11, 4))
plt.axis("off")

table = plt.table(
    cellText=region_display.values,
    colLabels=region_display.columns,
    loc="center",
    cellLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title(
    "Accuracy across the chirp frequency regions",
    pad=18,
)

plt.tight_layout()

plt.savefig(
    FIGURES_FOLDER / "11_frequency_region_metrics_table.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# 17. LIST GENERATED FIGURES
# ---------------------------------------------------------------------

print("\nFigures saved in:")
print(FIGURES_FOLDER)

for figure_file in sorted(
    FIGURES_FOLDER.glob("*.png")
):
    print(" -", figure_file.name)
