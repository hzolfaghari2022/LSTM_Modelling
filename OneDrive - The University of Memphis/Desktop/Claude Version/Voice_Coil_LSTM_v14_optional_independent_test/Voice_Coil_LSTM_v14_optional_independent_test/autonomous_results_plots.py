"""
Figure generation for the actuator system identification study.

Every figure is written to FiguresResults. The figures are grouped so that
they answer four questions in order:

    what data exists and how was it split        figures 01 and 02
    did the training converge                    figure 03
    does the model reproduce held out blocks     figures 04 and 05
    does the model generalise to unseen records  figures 06 onwards
"""

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
TEAL = "#00838f"

ROLE_COLORS = {
    "training": "#8fbf8f",
    "validation": "#f2c14e",
    "test": "#e8836b",
    "pure_test": "#b07bc4",
    "unused_short_remainder": "#cfd8dc",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#cfd8dc",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,
    }
)


HERE = Path(__file__).resolve().parent
RESULTS_FOLDER = Path(
    os.environ.get("DLSTM_RESULTS_FOLDER", str(HERE / "ResultsData"))
).resolve()
FIGURES_FOLDER = Path(
    os.environ.get("DLSTM_FIGURES_FOLDER", str(HERE / "FiguresResults"))
).resolve()

def prepare_figure_folder(folder: Path):
    """
    Empty the figures folder without removing the folder itself.

    Deleting the directory outright fails on Windows whenever something holds
    a handle on it, which OneDrive does routinely while it syncs, and so does
    an open image viewer or an Explorer window. The directory is therefore
    kept and only its contents are cleared, and a file that refuses to be
    deleted is reported rather than aborting the whole plotting stage.
    """
    folder.mkdir(parents=True, exist_ok=True)

    locked = []
    for item in folder.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
        except OSError:
            locked.append(item.name)

    if locked:
        print("These old figures could not be deleted and will be overwritten "
              "in place if possible:")
        for name in locked:
            print(f"  {name}")
        print("Close any image viewer or Explorer window on FiguresResults, "
              "or pause OneDrive syncing, if a figure fails to save.")


prepare_figure_folder(FIGURES_FOLDER)


archive = np.load(RESULTS_FOLDER / "simulation_results.npz")

with open(RESULTS_FOLDER / "manifest.json", encoding="utf-8") as handle:
    manifest = json.load(handle)

metrics = pd.read_csv(RESULTS_FOLDER / "metrics.csv")
inventory = pd.read_csv(RESULTS_FOLDER / "record_inventory.csv")
split_table = pd.read_csv(RESULTS_FOLDER / "data_split.csv")

PURE_TEST_RECORDS = manifest["pure_test_records"]
DEVELOPMENT_RECORDS = manifest["development_records"]
EXTRA_STEP_RECORDS = manifest.get("extra_step_records", [])
EXTRA_ZERO_RECORDS = manifest.get("extra_zero_input_records", [])
SAMPLE_RATE = float(manifest["sample_rate_hz"])

EVALUATION_KIND = {item["name"]: item["kind"] for item in manifest["evaluations"]}
HAS_REFERENCE = {item["name"]: item["has_reference"] for item in manifest["evaluations"]}


def evaluation(name):
    return {
        "time": archive[f"eval__{name}__time"],
        "current": archive[f"eval__{name}__current"],
        "measured": archive[f"eval__{name}__measured"],
        "predicted": archive[f"eval__{name}__predicted"],
        "baseline": archive[f"eval__{name}__baseline"],
    }


def record_signals(name):
    return {
        "time": archive[f"record__{name}__time"],
        "current": archive[f"record__{name}__current"],
        "outputs": archive[f"record__{name}__outputs"],
    }


def metric_value(name, output, column):
    row = metrics[(metrics["Evaluation"] == name) & (metrics["Output"] == output)]
    if row.empty:
        return np.nan
    return float(row.iloc[0][column])


def format_r2(value):
    """R2 is meaningless when the reference signal is flat."""
    if value is None or not np.isfinite(value):
        return "R2 not applicable (flat reference)"
    return f"R2 = {value:.4f}"


FAILED_FIGURES = []


def finish(figure, file_name):
    """
    Save one figure, tolerating a file that is locked by another program.

    A single figure left open in a viewer should not cost the whole run, so a
    failure here is recorded and reported at the end instead of raising.
    """
    target = FIGURES_FOLDER / file_name
    try:
        figure.savefig(target, dpi=170, bbox_inches="tight")
        print("  wrote", file_name, flush=True)
    except OSError as error:
        fallback = FIGURES_FOLDER / f"{target.stem}__new{target.suffix}"
        try:
            figure.savefig(fallback, dpi=170, bbox_inches="tight")
            print(f"  {file_name} was locked, wrote {fallback.name} instead",
                  flush=True)
        except OSError:
            FAILED_FIGURES.append((file_name, str(error)))
            print(f"  could not write {file_name}: {error}", flush=True)
    finally:
        plt.close(figure)


def plot_with_gaps(axis, time, values, **kwargs):
    """
    Draw a signal that may contain jumps in time.

    The internal test blocks are scattered through a record, so a plain line
    plot would draw long straight segments across the gaps.
    """
    time = np.asarray(time)
    values = np.asarray(values)
    if len(time) < 2:
        axis.plot(time, values, **kwargs)
        return
    steps = np.diff(time)
    typical = np.median(steps)
    break_points = np.where(steps > 4.0 * max(typical, 1e-9))[0]
    segments = np.split(np.arange(len(time)), break_points + 1)
    label = kwargs.pop("label", None)
    for position, segment in enumerate(segments):
        axis.plot(
            time[segment],
            values[segment],
            label=label if position == 0 else None,
            **kwargs,
        )


# ======================================================================
# 01  Record inventory
# ======================================================================

print("Figures:", flush=True)

order = list(inventory["record"])
columns = 4
rows = int(np.ceil(len(order) / columns))

figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 2.7 * rows))
axes = np.atleast_1d(axes).ravel()

for position, name in enumerate(order):
    axis = axes[position]
    info = inventory[inventory["record"] == name].iloc[0]
    signals = record_signals(name)
    time = signals["time"]
    displacement = signals["outputs"][:, 0]

    axis.plot(time, displacement, color=BLUE, linewidth=1.0, label="displacement")
    axis.set_ylabel("x [mm]", color=BLUE)
    axis.tick_params(axis="y", labelcolor=BLUE)

    twin = axis.twinx()
    twin.plot(time, signals["current"], color=ORANGE, linewidth=0.8, alpha=0.85)
    twin.set_ylabel("I [A]", color=ORANGE)
    twin.tick_params(axis="y", labelcolor=ORANGE)
    twin.spines["top"].set_visible(False)

    pure = info["usage"] == "pure_test"
    axis.set_title(
        f"{name}\n{info['load_mass_g']:g} g load, "
        f"{'PURE TEST' if pure else 'development'}",
        color=RED if pure else GRAY,
        fontsize=10,
    )
    axis.set_xlabel("time [s]")
    axis.grid(True)

for extra in range(len(order), len(axes)):
    axes[extra].axis("off")

figure.suptitle(
    "Figure 1  MEASURED DATA ONLY — no LSTM prediction. Complete inventory "
    "(blue: displacement, orange: coil current)",
    fontsize=15,
    y=1.002,
)
figure.tight_layout()
finish(figure, "01_record_inventory.png")


# ======================================================================
# 02  Split map
# ======================================================================

development = [name for name in order if name in DEVELOPMENT_RECORDS]
pure = [name for name in order if name in PURE_TEST_RECORDS]
ordered = development + pure

figure, axis = plt.subplots(figsize=(14, 0.42 * len(ordered) + 2.4))

for row_position, name in enumerate(ordered):
    blocks = split_table[split_table["record"] == name]
    for _, block in blocks.iterrows():
        start = block["start_time_s"]
        width = block["end_time_s"] - block["start_time_s"]
        axis.barh(
            row_position,
            width,
            left=start,
            height=0.68,
            color=ROLE_COLORS.get(block["role"], LIGHT_GRAY),
            edgecolor="white",
            linewidth=0.4,
        )

axis.set_yticks(range(len(ordered)))
axis.set_yticklabels(
    [
        f"{name}  ({'PURE TEST' if name in PURE_TEST_RECORDS else 'dev'})"
        for name in ordered
    ]
)
axis.invert_yaxis()
axis.set_xlabel("time within the record [s]")
axis.grid(True, axis="x")

handles = [
    plt.Rectangle((0, 0), 1, 1, color=ROLE_COLORS[role])
    for role in ["training", "validation", "test", "pure_test"]
]
axis.legend(
    handles,
    ["training", "validation", "internal test", "pure test (never touched)"],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=4,
    frameon=False,
)
axis.set_title(
    "Figure 2  Data split. Development records are cut into contiguous blocks, "
    "pure test records are held out completely."
)
figure.tight_layout()
finish(figure, "02_data_split_map.png")


# ======================================================================
# 03  Learning curves
# ======================================================================

training_history = archive["training_history"]
validation_history = archive["validation_history"]
fine_tune_history = archive["fine_tune_history"]
best_epoch = int(archive["best_epoch"])

figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))

epochs = np.arange(1, len(training_history) + 1)
axes[0].plot(epochs, training_history, color=BLUE, marker="o", markersize=3,
             label="training")
axes[0].plot(epochs, validation_history, color=ORANGE, marker="s", markersize=3,
             label="validation")
if 1 <= best_epoch <= len(validation_history):
    axes[0].axvline(best_epoch, color=GREEN, linestyle="--", linewidth=1.2)
    axes[0].annotate(
        f"best epoch {best_epoch}",
        xy=(best_epoch, validation_history[best_epoch - 1]),
        xytext=(8, 14),
        textcoords="offset points",
        color=GREEN,
    )
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("weighted MSE on normalised residuals")
axes[0].set_title("Training and validation loss")
axes[0].grid(True, which="both")
axes[0].legend(frameon=False)

if len(fine_tune_history) > 0:
    axes[1].plot(
        np.arange(1, len(fine_tune_history) + 1),
        fine_tune_history,
        color=PURPLE,
        marker="o",
        markersize=4,
    )
    axes[1].set_xlabel("fine tune epoch")
    axes[1].set_ylabel("development loss")
    axes[1].set_title("Final fine tune on training plus validation")
    axes[1].grid(True)
else:
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, "fine tune disabled", ha="center", va="center")

figure.suptitle("Figure 3  Convergence", fontsize=14)
figure.tight_layout()
finish(figure, "03_learning_curves.png")


# ======================================================================
# 04 and 05  Internal held out blocks
# ======================================================================

internal_names = [
    item["name"]
    for item in manifest["evaluations"]
    if item["kind"] == "internal_test"
]

for figure_index, (column, label, unit, color) in enumerate(
    [(0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN)]
):
    columns = 3
    rows = int(np.ceil(len(internal_names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()

    for position, name in enumerate(internal_names):
        axis = axes[position]
        result = evaluation(name)
        plot_with_gaps(
            axis, result["time"], result["measured"][:, column],
            color=GRAY, linewidth=1.6, label="COMSOL",
        )
        plot_with_gaps(
            axis, result["time"], result["predicted"][:, column],
            color=color, linewidth=1.0, linestyle="--",
            label="grey-box + LSTM residual",
        )
        r2 = metric_value(name, label, "R2")
        axis.set_title(f"{name}\n{format_r2(r2)}", fontsize=10)
        axis.set_xlabel("time [s]")
        axis.set_ylabel(f"{label} [{unit}]")
        axis.grid(True)
        if position == 0:
            axis.legend(frameon=False, fontsize=8)

    for extra in range(len(internal_names), len(axes)):
        axes[extra].axis("off")

    pooled_r2 = metric_value("ALL_internal_test_blocks", label, "R2")
    figure.suptitle(
        f"Figure {4 + figure_index}  {label} on the internal held out blocks. "
        f"Pooled {format_r2(pooled_r2)}",
        fontsize=14,
        y=1.002,
    )
    figure.tight_layout()
    finish(
        figure,
        f"0{4 + figure_index}_internal_test_"
        f"{'displacement' if column == 0 else 'force'}.png",
    )


# ======================================================================
# 06  Pure test records, one detailed figure each
# ======================================================================

for position, name in enumerate(PURE_TEST_RECORDS, start=1):
    result = evaluation(name)
    info = inventory[inventory["record"] == name].iloc[0]

    figure, axes = plt.subplots(
        4, 1, figsize=(13, 11), sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.3, 1.3, 0.9]},
    )

    axes[0].plot(result["time"], result["current"], color=ORANGE, linewidth=1.0)
    axes[0].set_ylabel("current [A]")
    axes[0].set_title("Input, coil current")
    axes[0].grid(True)

    for axis_index, (column, label, unit, color) in enumerate(
        [(0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN)],
        start=1,
    ):
        axis = axes[axis_index]
        axis.plot(result["time"], result["measured"][:, column],
                  color=GRAY, linewidth=1.8, label="COMSOL reference")
        axis.plot(result["time"], result["predicted"][:, column],
                  color=color, linewidth=1.1, linestyle="--",
                  label="grey-box + LSTM residual")
        axis.plot(result["time"], result["baseline"][:, column],
                  color=RED, linewidth=0.8, alpha=0.55, linestyle=":",
                  label="grey-box physics only")
        r2 = metric_value(name, label, "R2")
        rmse = metric_value(name, label, "RMSE")
        axis.set_ylabel(f"{label} [{unit}]")
        axis.set_title(f"{label}:  {format_r2(r2)},  RMSE = {rmse:.5f} {unit}")
        axis.grid(True)
        axis.legend(frameon=False, loc="best", fontsize=9)

    axes[3].plot(
        result["time"],
        result["measured"][:, 0] - result["predicted"][:, 0],
        color=BLUE, linewidth=0.9, label="displacement error [mm]",
    )
    twin = axes[3].twinx()
    twin.plot(
        result["time"],
        result["measured"][:, 1] - result["predicted"][:, 1],
        color=GREEN, linewidth=0.9, label="force error [N]",
    )
    twin.set_ylabel("force error [N]", color=GREEN)
    twin.tick_params(axis="y", labelcolor=GREEN)
    twin.spines["top"].set_visible(False)
    axes[3].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[3].set_ylabel("displacement error [mm]", color=BLUE)
    axes[3].tick_params(axis="y", labelcolor=BLUE)
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("Prediction error")
    axes[3].grid(True)

    tag = ""
    if name in EXTRA_STEP_RECORDS:
        tag = "   [extra validation signal 1: step input]"
    if name in EXTRA_ZERO_RECORDS:
        tag = "   [extra validation signal 2: zero input]"

    figure.suptitle(
        f"Figure 6.{position}  Pure test record {name}{tag}\n"
        f"{info['description']}, load mass {info['load_mass_g']:g} g, "
        f"total mass {info['total_mass_g']:g} g",
        fontsize=13,
        y=0.998,
    )
    figure.tight_layout()
    finish(figure, f"06_{position}_pure_test_{name}.png")


# ======================================================================
# 07  Parity plots for the pure test records
# ======================================================================

columns = 4
rows = int(np.ceil(len(PURE_TEST_RECORDS) / columns))

for column, label, unit, color in [
    (0, "Displacement", "mm", BLUE),
    (1, "Lorentz force", "N", GREEN),
]:
    figure, axes = plt.subplots(rows, columns, figsize=(3.7 * columns, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for position, name in enumerate(PURE_TEST_RECORDS):
        axis = axes[position]
        result = evaluation(name)
        measured = result["measured"][:, column]
        predicted = result["predicted"][:, column]

        low = float(min(measured.min(), predicted.min()))
        high = float(max(measured.max(), predicted.max()))
        span = high - low
        if span <= 0:
            span = 1.0
        low -= 0.05 * span
        high += 0.05 * span

        axis.plot([low, high], [low, high], color=GRAY, linewidth=1.0)
        axis.scatter(measured, predicted, s=3, alpha=0.35, color=color,
                     edgecolors="none")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        r2 = metric_value(name, label, "R2")
        axis.set_title(f"{name}\n{format_r2(r2)}", fontsize=9)
        axis.set_xlabel(f"COMSOL [{unit}]")
        axis.set_ylabel(f"model [{unit}]")
        axis.grid(True)

    for extra in range(len(PURE_TEST_RECORDS), len(axes)):
        axes[extra].axis("off")

    suffix = "displacement" if column == 0 else "force"
    figure.suptitle(
        f"Figure 7  Parity plots, {label.lower()} on the pure test records",
        fontsize=14,
        y=1.002,
    )
    figure.tight_layout()
    finish(figure, f"07_parity_{suffix}.png")


# ======================================================================
# 08  Metric summary
# ======================================================================

summary = metrics[metrics["Kind"].isin(["internal_test", "pure_test"])].copy()
names = list(dict.fromkeys(summary["Evaluation"]))

R2_FLOOR = -0.5

figure, axes = plt.subplots(2, 1, figsize=(max(12, 0.8 * len(names)), 9.5))

positions = np.arange(len(names))
width = 0.38

# ---- upper panel, R2 with a display floor -----------------------------
axis = axes[0]
for offset, output, color in [
    (-width / 2, "Displacement", BLUE),
    (width / 2, "Lorentz force", GREEN),
]:
    raw = np.array([metric_value(n, output, "R2") for n in names], dtype=float)
    shown = np.clip(raw, R2_FLOOR, None)
    finite = np.isfinite(raw)
    axis.bar(positions[finite] + offset, shown[finite], width,
             color=color, label=output)
    for position, value, drawn in zip(positions[finite], raw[finite], shown[finite]):
        if value < R2_FLOOR:
            axis.annotate(
                f"{value:.1f}",
                xy=(position + offset, R2_FLOOR),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=RED,
                rotation=90,
            )
    for position, value in zip(positions[~finite], raw[~finite]):
        axis.annotate(
            "n/a",
            xy=(position + offset, 0.0),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=GRAY,
        )

axis.axhline(1.0, color=GRAY, linestyle="--", linewidth=0.9)
axis.axhline(0.0, color=RED, linestyle="--", linewidth=0.9)
axis.set_ylim(R2_FLOOR - 0.12, 1.08)
axis.set_ylabel("R2")
axis.set_title(
    "Coefficient of determination R2, 1.0 is perfect and 0.0 means no better "
    f"than predicting the mean. Bars are clipped at {R2_FLOOR:g}, the true "
    "value is printed underneath. 'n/a' marks a flat reference signal."
)
axis.grid(True, axis="y")
axis.legend(frameon=False)

# ---- lower panel, absolute RMSE on a log scale ------------------------
axis = axes[1]
for offset, output, color in [
    (-width / 2, "Displacement", BLUE),
    (width / 2, "Lorentz force", GREEN),
]:
    values = np.array([metric_value(n, output, "RMSE") for n in names], dtype=float)
    unit = "mm" if output == "Displacement" else "N"
    axis.bar(positions + offset, values, width, color=color,
             label=f"{output} [{unit}]")

axis.set_yscale("log")
axis.set_ylabel("RMSE (log scale)")
axis.set_title("Absolute RMSE. This stays meaningful even where R2 does not.")
axis.grid(True, axis="y", which="both")
axis.legend(frameon=False)

for axis in axes:
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{n}\n{'PURE TEST' if n in PURE_TEST_RECORDS else 'internal'}"
            for n in names
        ],
        rotation=60,
        ha="right",
        fontsize=8,
    )

figure.suptitle("Figure 8  Accuracy summary across every evaluation set", fontsize=14)
figure.tight_layout()
finish(figure, "08_metric_summary.png")


# ======================================================================
# 09  Step response validation
# ======================================================================

step_names = [
    name for name in PURE_TEST_RECORDS
    if inventory[inventory["record"] == name].iloc[0]["family"] == "step"
]

if step_names:
    figure, axes = plt.subplots(
        2, len(step_names), figsize=(6.2 * len(step_names), 8), squeeze=False
    )
    for position, name in enumerate(step_names):
        result = evaluation(name)
        for row, (column, label, unit, color) in enumerate(
            [(0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN)]
        ):
            axis = axes[row][position]
            axis.plot(result["time"], result["measured"][:, column],
                      color=GRAY, linewidth=2.0, label="COMSOL reference")
            axis.plot(result["time"], result["predicted"][:, column],
                      color=color, linewidth=1.2, linestyle="--",
                      label="grey-box + LSTM residual")
            r2 = metric_value(name, label, "R2")
            axis.set_title(f"{name}\n{label}, {format_r2(r2)}", fontsize=11)
            axis.set_xlabel("time [s]")
            axis.set_ylabel(f"{label} [{unit}]")
            axis.grid(True)
            axis.legend(frameon=False, fontsize=9)

    figure.suptitle(
        "Figure 9  Untouched step-input records: transient shape, damping, "
        "and settled-value validation.",
        fontsize=13,
    )
    figure.tight_layout()
    finish(figure, "09_step_input_validation.png")


# ======================================================================
# 10  Zero input validation
# ======================================================================

zero_names = [
    name for name in PURE_TEST_RECORDS
    if inventory[inventory["record"] == name].iloc[0]["family"] == "zero_input"
]

if zero_names:
    figure, axes = plt.subplots(
        2, len(zero_names), figsize=(6.2 * len(zero_names), 8), squeeze=False
    )
    for position, name in enumerate(zero_names):
        result = evaluation(name)
        for row, (column, label, unit, color) in enumerate(
            [(0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN)]
        ):
            axis = axes[row][position]
            axis.plot(result["time"], result["measured"][:, column],
                      color=GRAY, linewidth=2.0, label="COMSOL reference")
            axis.plot(result["time"], result["predicted"][:, column],
                      color=color, linewidth=1.2, linestyle="--",
                      label="grey-box + LSTM residual")
            settled_measured = float(np.mean(result["measured"][-50:, column]))
            settled_model = float(np.mean(result["predicted"][-50:, column]))
            rmse = metric_value(name, label, "RMSE")
            r2 = metric_value(name, label, "R2")
            axis.set_title(
                f"{name}\n{label}, {format_r2(r2)}, RMSE = {rmse:.5f} {unit}\n"
                f"settled COMSOL = {settled_measured:.4f} {unit}, "
                f"settled model = {settled_model:.4f} {unit}",
                fontsize=10,
            )
            axis.set_xlabel("time [s]")
            axis.set_ylabel(f"{label} [{unit}]")
            axis.grid(True)
            axis.legend(frameon=False, fontsize=9)

    figure.suptitle(
        "Figure 10  Zero input validation. The coil current is identically "
        "zero; displacement comes from the initialized grey-box mechanics "
        "and Lorentz force is constrained to zero.",
        fontsize=13,
    )
    figure.tight_layout()
    finish(figure, "10_zero_input_validation.png")


# ======================================================================
# 11  Frequency content of the untouched chirp
# ======================================================================

chirp_candidates = [
    name for name in PURE_TEST_RECORDS
    if inventory[inventory["record"] == name].iloc[0]["family"]
    in ("dc_plus_chirp", "chirp")
]
chirp_name = chirp_candidates[0] if chirp_candidates else None

if chirp_name is not None:
    result = evaluation(chirp_name)
    time = result["time"]
    step = float(np.median(np.diff(time)))

    figure, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Zoom on a one second slice in the middle of the record, wherever that
    # falls, so this works for any chirp length the workbook happens to hold.
    middle = 0.5 * (float(time[0]) + float(time[-1]))
    zoom = (time >= middle - 0.5) & (time <= middle + 0.5)
    if zoom.sum() < 10:
        zoom = np.ones_like(time, dtype=bool)
    axes[0][0].plot(time[zoom], result["measured"][zoom, 0], color=GRAY,
                    linewidth=1.8, label="COMSOL reference")
    axes[0][0].plot(time[zoom], result["predicted"][zoom, 0], color=BLUE,
                    linewidth=1.1, linestyle="--",
                    label="grey-box + LSTM residual")
    axes[0][0].set_title("Displacement, one second zoom around mid chirp")
    axes[0][0].set_xlabel("time [s]")
    axes[0][0].set_ylabel("displacement [mm]")
    axes[0][0].grid(True)
    axes[0][0].legend(frameon=False)

    axes[0][1].plot(time[zoom], result["measured"][zoom, 1], color=GRAY,
                    linewidth=1.8, label="COMSOL reference")
    axes[0][1].plot(time[zoom], result["predicted"][zoom, 1], color=GREEN,
                    linewidth=1.1, linestyle="--",
                    label="grey-box + LSTM residual")
    axes[0][1].set_title("Lorentz force, same zoom")
    axes[0][1].set_xlabel("time [s]")
    axes[0][1].set_ylabel("force [N]")
    axes[0][1].grid(True)
    axes[0][1].legend(frameon=False)

    for axis_index, (column, label, unit, color) in enumerate(
        [(0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN)]
    ):
        signal_measured = result["measured"][:, column]
        signal_model = result["predicted"][:, column]
        window = np.hanning(len(signal_measured))
        frequencies = np.fft.rfftfreq(len(signal_measured), d=step)
        spectrum_measured = np.abs(
            np.fft.rfft((signal_measured - signal_measured.mean()) * window)
        )
        spectrum_model = np.abs(
            np.fft.rfft((signal_model - signal_model.mean()) * window)
        )
        scale = 2.0 / np.sum(window)

        axis = axes[1][axis_index]
        axis.semilogy(frequencies, spectrum_measured * scale + 1e-12,
                      color=GRAY, linewidth=1.4, label="COMSOL reference")
        axis.semilogy(frequencies, spectrum_model * scale + 1e-12,
                      color=color, linewidth=1.0, linestyle="--",
                      label="grey-box + LSTM residual")
        axis.set_xlim(0, 40)
        axis.set_xlabel("frequency [Hz]")
        axis.set_ylabel(f"amplitude [{unit}]")
        axis.set_title(f"{label} spectrum over the chirp band")
        axis.grid(True, which="both")
        axis.legend(frameon=False)

    figure.suptitle(
        f"Figure 11  Untouched chirp {chirp_name}. Time domain detail and "
        "frequency "
        "content, which shows whether the resonance is placed correctly.",
        fontsize=13,
    )
    figure.tight_layout()
    finish(figure, "11_untouched_chirp_detail.png")


# ======================================================================
# 12  Synthetic probe signals
# ======================================================================

probe_names = [
    item["name"] for item in manifest["evaluations"]
    if item["kind"] == "synthetic_probe"
]

if probe_names:
    step_probes = [n for n in probe_names if "Step" in n]
    zero_probes = [n for n in probe_names if "Zero" in n]

    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    palette = [BLUE, ORANGE, PURPLE, TEAL, RED, GREEN]

    for group_index, (group, group_title) in enumerate(
        [(step_probes, "Synthetic step input"),
         (zero_probes, "Synthetic zero input")]
    ):
        for column, label, unit in [(0, "displacement", "mm"), (1, "force", "N")]:
            axis = axes[column][group_index]
            for position, name in enumerate(group):
                result = evaluation(name)
                axis.plot(
                    result["time"],
                    result["predicted"][:, column],
                    color=palette[position % len(palette)],
                    linewidth=1.4,
                    label=name.replace("Synthetic_", ""),
                )
            axis.axhline(0.0, color=GRAY, linewidth=0.8)
            axis.set_xlabel("time [s]")
            axis.set_ylabel(f"{label} [{unit}]")
            axis.set_title(f"{group_title}, predicted {label}")
            axis.grid(True)
            axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Figure 12  Generated Load-2 zero and ideal 150 mA step inputs, "
        "initialized only with measured x0, velocity0, and force0. There is "
        "no generated-signal ground truth; this is a physical sanity check.",
        fontsize=12,
    )
    figure.tight_layout()
    finish(figure, "12_synthetic_probe_response.png")


# ======================================================================
# 13  Printable metric table
# ======================================================================

table = metrics[metrics["Kind"].isin(["internal_test", "internal_test_pooled",
                                      "pure_test"])].copy()
table = table[["Evaluation", "Kind", "Output", "Unit", "Samples",
               "RMSE", "MAE", "MaxAbsError", "NRMSE_percent", "R2", "Fit_percent"]]

figure, axis = plt.subplots(figsize=(16, 0.34 * len(table) + 2.0))
axis.axis("off")

def cell(value, digits):
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


cell_text = []
for _, row in table.iterrows():
    cell_text.append(
        [
            row["Evaluation"],
            row["Kind"],
            row["Output"],
            cell(row["RMSE"], 5),
            cell(row["MAE"], 5),
            cell(row["MaxAbsError"], 5),
            cell(row["NRMSE_percent"], 2),
            cell(row["R2"], 4),
            cell(row["Fit_percent"], 2),
        ]
    )

rendered = axis.table(
    cellText=cell_text,
    colLabels=["evaluation", "kind", "output", "RMSE", "MAE", "max abs error",
               "NRMSE %", "R2", "fit %"],
    loc="center",
    cellLoc="center",
)
rendered.auto_set_font_size(False)
rendered.set_fontsize(8)
rendered.scale(1.0, 1.25)

for (row_index, _), cell in rendered.get_celld().items():
    if row_index == 0:
        cell.set_facecolor(GRAY)
        cell.set_text_props(color="white", weight="bold")
    else:
        name = cell_text[row_index - 1][0]
        if name in PURE_TEST_RECORDS:
            cell.set_facecolor("#f3e5f5")
        else:
            cell.set_facecolor("#eceff1" if row_index % 2 else "white")

axis.set_title("Figure 13  Full metric table. Purple rows are pure test records.",
               fontsize=13, pad=18)
figure.tight_layout()
finish(figure, "13_metric_table.png")

if FAILED_FIGURES:
    print()
    print(f"{len(FAILED_FIGURES)} figure(s) could not be written:")
    for name, reason in FAILED_FIGURES:
        print(f"  {name}: {reason}")
    print("This is almost always a file lock. Close anything viewing "
          "FiguresResults, pause OneDrive syncing, and rerun:")
    print("    python plot_results.py")
    print("The results themselves are already saved in ResultsData, so the "
          "training does not need to be repeated.")
else:
    print(f"\nAll figures are in: {FIGURES_FOLDER}")
