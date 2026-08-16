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


BLUE = "#1769aa"

ORANGE = "#ef6c00"

GREEN = "#2e7d32"

RED = "#c62828"

PURPLE = "#6a1b9a"

GRAY = "#455a64"

LIGHT_GRAY = "#eceff1"


plt.rcParams.update(

    {

        "font.family": "DejaVu Sans",

        "font.size": 11,

        "axes.titlesize": 14,

        "axes.labelsize": 12,

        "legend.fontsize": 10,

        "xtick.labelsize": 10,

        "ytick.labelsize": 10,

        "axes.facecolor": "white",

        "figure.facecolor": "white",

        "grid.color": "#cfd8dc",

        "grid.linewidth": 0.8,

        "grid.alpha": 0.55,

        "axes.spines.top": False,

        "axes.spines.right": False,

    }

)


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

    "training": BLUE,

    "validation": ORANGE,

    "test": RED,

    "unused_short_remainder": LIGHT_GRAY,

}

OUTPUTS = [

    (0, "displacement", "Displacement", "mm"),

    (1, "force", "Lorentz force", "N"),

]


def finish_figure(file_name):

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(FIGURES_FOLDER / file_name, dpi=180, bbox_inches="tight")

    plt.close()


def plot_discontinuous(time, values, **kwargs):


    gap_threshold = 3.0 * np.median(np.diff(time))

    starts = np.r_[0, np.where(np.diff(time) > gap_threshold)[0] + 1]

    stops = np.r_[starts[1:], len(time)]

    for segment_number, (start, stop) in enumerate(zip(starts, stops)):

        segment_kwargs = dict(kwargs)

        if segment_number > 0:

            segment_kwargs.pop("label", None)

        plt.plot(time[start:stop], values[start:stop], **segment_kwargs)


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


epochs = np.arange(1, len(training_history) + 1)

plt.figure(figsize=(9, 5))

plt.semilogy(epochs, training_history, color=BLUE, label="Training")

plt.semilogy(epochs, validation_history, color=ORANGE, label="Validation")

plt.axvline(best_epoch, color=GREEN, linestyle=":", label=f"Best epoch: {best_epoch}")

plt.xlabel("Epoch")

plt.ylabel("Normalized loss")

plt.title("Model convergence")

plt.legend(frameon=False)

finish_figure("02_training_validation_loss.png")


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

        color=BLUE,

        label="Measured",

        linewidth=1.4,

    )

    axis.plot(

        all_time,

        all_prediction[:, index],

        color=ORANGE,

        linestyle="--",

        label="Predicted",

        linewidth=1.1,

    )

    axis.axvspan(

        raw_time[0],

        all_time[0],

        color=LIGHT_GRAY,

        alpha=0.35,

        label="120-sample history",

    )

    axis.set_ylabel(f"{display_name} ({unit})")

    axis.grid(True)

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


for index, short_name, display_name, unit in OUTPUTS:

    plt.figure(figsize=(12, 5))

    plot_discontinuous(

        test_time,

        test_measured[:, index],

        color=BLUE,

        label="Measured",

        linewidth=1.4,

    )

    plot_discontinuous(

        test_time,

        test_prediction[:, index],

        color=ORANGE,

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

    plot_discontinuous(test_time, error, color=RED)

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

        color=BLUE,

    )

    plt.plot([minimum, maximum], [minimum, maximum], color=GRAY, linestyle="--", label="Ideal prediction")

    plt.xlabel(f"Measured {display_name} ({unit})")

    plt.ylabel(f"Predicted {display_name} ({unit})")

    plt.title(f"Test parity: {display_name}")

    plt.legend()

    finish_figure(f"06_test_{short_name}_parity.png")


display_metrics = metrics.drop(columns=["Series"]).copy()

display_metrics["Evaluation"] = display_metrics["Evaluation"].replace(

    {

        "Distributed held-out test blocks": "Internal held-out 127 mA",

        "Complete selected record (descriptive)": "Complete 127 mA record*",

    }

)

display_metrics["Output"] = display_metrics["Output"].replace({"Lorentz force": "Force"})

for column in ["MSE", "RMSE", "MAE"]:

    display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.5g}")

display_metrics["R2"] = display_metrics["R2"].map(lambda value: f"{value:.4f}")

display_metrics["Fit_percent"] = display_metrics["Fit_percent"].map(lambda value: f"{value:.2f}%")

display_columns = ["R²" if column == "R2" else "Fit" if column == "Fit_percent" else column for column in display_metrics.columns]

figure, axis = plt.subplots(figsize=(16, 5.2))

axis.axis("off")

table = axis.table(

    cellText=display_metrics.values,

    colLabels=display_columns,

    loc="center",

    cellLoc="center",

    colLoc="center",

    colWidths=[0.24, 0.12, 0.07, 0.10, 0.10, 0.10, 0.08, 0.09],

)

table.auto_set_font_size(False)

table.set_fontsize(10)

table.scale(1.0, 2.05)

for (row_number, column_number), cell in table.get_celld().items():

    cell.set_edgecolor("#cfd8dc")

    cell.set_linewidth(0.8)

    if row_number == 0:

        cell.set_facecolor(BLUE)

        cell.set_text_props(color="white", weight="bold")

    else:

        cell.set_facecolor(LIGHT_GRAY if row_number % 2 == 0 else "white")

axis.set_title(

    "Test 2 — held-out and complete-record performance",

    fontsize=15,

    pad=18,

)

axis.text(

    0.5,

    0.035,

    "Internal held-out 127 mA: distributed blocks excluded from fitting.  "
    "Complete 127 mA record*: entire series, including fitted regions; descriptive only.\n"
    "Higher R² and Fit are better; lower MSE, RMSE, and MAE are better.",

    transform=axis.transAxes,

    ha="center",

    fontsize=10,

    color=GRAY,

)

figure.tight_layout()

figure.savefig(FIGURES_FOLDER / "07_metrics.png", dpi=180, bbox_inches="tight")

plt.close(figure)


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
