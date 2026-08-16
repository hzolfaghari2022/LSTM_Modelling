from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "ResultsData"
FIGURES = HERE / "FiguresResults"
FIGURES.mkdir(exist_ok=True)

BLUE = "#1769aa"
ORANGE = "#ef6c00"

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


for source_name, slide_name in {
    "02_untouched_final_test_input.png": "02_final_test_input_147mA.png",
    "05_final_test_displacement.png": "04_displacement_tracking_full.png",
    "05_final_test_force.png": "04_force_tracking_full.png",
    "06_final_test_displacement_error.png": "06_displacement_error_vs_time.png",
    "06_final_test_force_error.png": "06_force_error_vs_time.png",
}.items():
    shutil.copy2(FIGURES / source_name, FIGURES / slide_name)


results = np.load(RESULTS / "simulation_results.npz")
time = results["final_test_time"]
measured = results["final_measured"]
predicted = results["final_prediction"]
zoom_start = int(0.80 * len(time))


for index, short_name, display_name, unit in (
    (0, "displacement", "Displacement", "mm"),
    (1, "force", "Lorentz force", "N"),
):
    plt.figure(figsize=(12, 5))
    plt.plot(
        time[zoom_start:],
        measured[zoom_start:, index],
        color=BLUE,
        label="Measured",
        linewidth=1.4,
    )
    plt.plot(
        time[zoom_start:],
        predicted[zoom_start:, index],
        color=ORANGE,
        linestyle="--",
        label="Predicted",
        linewidth=1.1,
    )
    plt.xlabel("Time (s)")
    plt.ylabel(f"{display_name} ({unit})")
    plt.title(f"High-frequency zoom: {short_name}")
    plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"05_{short_name}_high_frequency_zoom.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


print("Slide figures updated in:", FIGURES)
