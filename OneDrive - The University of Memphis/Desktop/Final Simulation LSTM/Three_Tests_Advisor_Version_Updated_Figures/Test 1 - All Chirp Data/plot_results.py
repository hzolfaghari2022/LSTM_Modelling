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


from config import FINAL_TEST_SHEET, MIXED_SHEET, TRAIN_ONLY_SHEETS

from data_utils import find_workbook, load_sheet


HERE = Path(__file__).resolve().parent

RESULTS_FOLDER = HERE / "ResultsData"

FIGURES_FOLDER = HERE / "FiguresResults"

FIGURES_FOLDER.mkdir(exist_ok=True)


results = np.load(RESULTS_FOLDER / "simulation_results.npz")

metrics = pd.read_csv(RESULTS_FOLDER / "metrics.csv")

split_table = pd.read_csv(RESULTS_FOLDER / "data_split.csv")


development_time = results["development_test_time"]

development_measured = results["development_measured"]

development_prediction = results["development_prediction"]

final_time = results["final_test_time"]

final_measured = results["final_measured"]

final_prediction = results["final_prediction"]

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


    if len(time) < 2:

        plt.plot(time, values, **kwargs)

        return

    gap_threshold = 3.0 * np.median(np.diff(time))

    segment_starts = np.r_[0, np.where(np.diff(time) > gap_threshold)[0] + 1]

    segment_stops = np.r_[segment_starts[1:], len(time)]

    for segment_number, (start, stop) in enumerate(

        zip(segment_starts, segment_stops)

    ):

        segment_kwargs = dict(kwargs)

        if segment_number > 0:

            segment_kwargs.pop("label", None)

        plt.plot(time[start:stop], values[start:stop], **segment_kwargs)


workbook = find_workbook(HERE)

for sheet_number, sheet in enumerate(TRAIN_ONLY_SHEETS + [MIXED_SHEET], 1):

    time, current, _ = load_sheet(workbook, sheet)

    rows = split_table[split_table["sheet"] == sheet]

    plt.figure(figsize=(12, 4))

    used_labels = set()

    for _, row in rows.iterrows():

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

    plt.title(f"{sheet}: time/frequency-distributed data roles")

    plt.legend()

    finish_figure(f"01_{sheet_number}_data_split_{sheet}.png")


time, current, _ = load_sheet(workbook, FINAL_TEST_SHEET)

plt.figure(figsize=(12, 4))

plt.plot(time, current, color=PURPLE, label="Pure test")

plt.xlabel("Time (s)")

plt.ylabel("Coil current (A)")

plt.title(f"{FINAL_TEST_SHEET}: complete untouched final test")

plt.legend()

finish_figure("02_untouched_final_test_input.png")


epochs = np.arange(1, len(training_history) + 1)

plt.figure(figsize=(9, 5))

plt.semilogy(epochs, training_history, label="Training", color=BLUE)

plt.semilogy(epochs, validation_history, label="Validation", color=ORANGE)

plt.axvline(best_epoch, color=GREEN, linestyle=":", label=f"Best epoch: {best_epoch}")

plt.xlabel("Epoch")

plt.ylabel("Normalized loss")

plt.title("Model convergence")

plt.legend(frameon=False)

finish_figure("03_training_validation_loss.png")


for index, short_name, display_name, unit in OUTPUTS:

    plt.figure(figsize=(12, 5))

    plot_discontinuous(

        development_time,

        development_measured[:, index],

        color=BLUE,

        label="Measured",

        linewidth=1.4,

    )

    plot_discontinuous(

        development_time,

        development_prediction[:, index],

        color=ORANGE,

        linestyle="--",

        label="Predicted",

        linewidth=1.1,

    )

    plt.xlabel("Time (s)")

    plt.ylabel(f"{display_name} ({unit})")

    plt.title(f"{MIXED_SHEET} distributed test: {short_name}")

    plt.legend()

    finish_figure(f"04_internal_test_{short_name}.png")


for index, short_name, display_name, unit in OUTPUTS:

    plt.figure(figsize=(12, 5))

    plt.plot(final_time, final_measured[:, index], color=BLUE, label="Measured", linewidth=1.4)

    plt.plot(final_time, final_prediction[:, index], color=ORANGE, linestyle="--", label="Predicted", linewidth=1.1)

    plt.xlabel("Time (s)")

    plt.ylabel(f"{display_name} ({unit})")

    plt.title(f"{FINAL_TEST_SHEET} pure test: {short_name}")

    plt.legend()

    finish_figure(f"05_final_test_{short_name}.png")


for index, short_name, display_name, unit in OUTPUTS:

    error = final_measured[:, index] - final_prediction[:, index]

    plt.figure(figsize=(12, 4))

    plt.plot(final_time, error, color=RED)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Time (s)")

    plt.ylabel(f"{display_name} error ({unit})")

    plt.title(f"Pure-test {short_name} prediction error")

    finish_figure(f"06_final_test_{short_name}_error.png")


    minimum = min(final_measured[:, index].min(), final_prediction[:, index].min())

    maximum = max(final_measured[:, index].max(), final_prediction[:, index].max())

    plt.figure(figsize=(6, 6))

    plt.scatter(final_measured[:, index], final_prediction[:, index], color=BLUE, s=12, alpha=0.45)

    plt.plot([minimum, maximum], [minimum, maximum], color=GRAY, linestyle="--", label="Ideal prediction")

    plt.xlabel(f"Measured {display_name} ({unit})")

    plt.ylabel(f"Predicted {display_name} ({unit})")

    plt.title(f"Pure-test {display_name}: parity")

    plt.legend()

    finish_figure(f"07_final_test_{short_name}_parity.png")


metric_ranges = [
    np.ptp(development_measured[:, 0]),
    np.ptp(development_measured[:, 1]),
    np.ptp(final_measured[:, 0]),
    np.ptp(final_measured[:, 1]),
]
metric_samples = [
    len(development_time),
    len(development_time),
    len(final_time),
    len(final_time),
]

display_metrics = metrics.copy()
display_metrics["Evaluation"] = display_metrics["Evaluation"].replace(
    {
        "DC_Offset_127mA distributed internal test": "Internal held-out\n127 mA",
        "DC_Offset_147mA untouched pure test": "Untouched pure test\n147 mA",
    }
)
display_metrics["Output"] = display_metrics["Output"].replace({"Lorentz force": "Force"})
display_metrics["Samples"] = metric_samples
display_metrics["NRMSE"] = [
    100.0 * rmse / value_range if value_range > 0 else np.nan
    for rmse, value_range in zip(metrics["RMSE"], metric_ranges)
]

table_rows = []
for row in display_metrics.itertuples():
    table_rows.append(
        [
            row.Evaluation,
            row.Output,
            f"{row.Samples:,}",
            f"{row.R2:.4f}",
            f"{row.MSE:.3e}",
            f"{row.RMSE:.5f} {row.Unit}",
            f"{row.MAE:.5f} {row.Unit}",
            f"{row.NRMSE:.2f}%",
            f"{row.Fit_percent:.2f}%",
        ]
    )

figure, axis = plt.subplots(figsize=(16, 5.4))
axis.axis("off")
axis.set_title("Test 1 — held-out prediction performance", fontsize=16, pad=18, weight="bold")
table = axis.table(
    cellText=table_rows,
    colLabels=["Evaluation", "Output", "Samples", "R²", "MSE", "RMSE", "MAE", "NRMSE", "Fit"],
    loc="center",
    cellLoc="center",
    colLoc="center",
    colWidths=[0.20, 0.10, 0.08, 0.075, 0.11, 0.125, 0.125, 0.10, 0.09],
)
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1.0, 2.15)

for (row_number, column_number), cell in table.get_celld().items():
    cell.set_edgecolor("#c5d2dc")
    cell.set_linewidth(0.8)
    if row_number == 0:
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#f7fafc" if row_number % 2 else LIGHT_GRAY)

for row_number, result in enumerate(metrics.itertuples(), start=1):
    r2_cell = table[(row_number, 3)]
    r2_cell.set_facecolor("#c8e6c9" if result.R2 >= 0.90 else "#fff3cd" if result.R2 >= 0.80 else "#ffcdd2")
    r2_cell.set_text_props(weight="bold")

axis.text(
    0.5,
    0.025,
    "Higher R² and Fit are better; lower MSE, RMSE, MAE, and NRMSE are better.",
    transform=axis.transAxes,
    ha="center",
    fontsize=10,
    color=GRAY,
)
figure.tight_layout()
figure.savefig(FIGURES_FOLDER / "08_all_test_metrics.png", dpi=180, bbox_inches="tight")
plt.close(figure)


region_rows = []

for region_name, indices in zip(

    ["Beginning / lower frequency", "Middle / medium frequency", "End / higher frequency"],

    np.array_split(np.arange(len(final_time)), 3),

):

    for index, _, display_name, unit in OUTPUTS:

        error = final_measured[indices, index] - final_prediction[indices, index]

        denominator = np.sum(

            (final_measured[indices, index] - final_measured[indices, index].mean()) ** 2

        )

        r2 = 1 - np.sum(error ** 2) / denominator if denominator > 0 else np.nan

        region_rows.append(

            {

                "Region": region_name,

                "Output": display_name,

                "Unit": unit,

                "RMSE": float(np.sqrt(np.mean(error ** 2))),

                "R2": float(r2),

            }

        )


region_metrics = pd.DataFrame(region_rows)

region_metrics.to_csv(RESULTS_FOLDER / "frequency_region_metrics.csv", index=False)


report_folder = HERE.parent / "Overleaf_Report" / "figures" / "Test1"

report_folder.mkdir(parents=True, exist_ok=True)

for source_name, report_name in {

    "03_training_validation_loss.png": "test1_convergence.png",

    "05_final_test_displacement.png": "test1_pure_test_displacement.png",

    "05_final_test_force.png": "test1_pure_test_force.png",

    "08_all_test_metrics.png": "test1_metrics.png",

}.items():

    shutil.copy2(FIGURES_FOLDER / source_name, report_folder / report_name)


print("\nFigures saved in:", FIGURES_FOLDER)

for figure_file in sorted(FIGURES_FOLDER.glob("*.png")):

    print(" -", figure_file.name)
