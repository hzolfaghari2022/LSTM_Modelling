"""Complete figure report for the causal one-step measured-feedback model.

This module contains plotting and post-training metric aggregation only.  It
does not fit, modify, or select the model.  ``main.py`` calls
``generate_complete_figures`` after the checkpoint and all predictions have
already been frozen.
"""

import json
import os
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

# Measured-versus-predicted comparison style.  The two curves are often
# almost identical, so color alone is not enough to separate them.  Measured
# data are solid black and slightly thicker.  Predictions are vivid,
# long-dashed, drawn on top, and carry sparse white-centred markers.
MEASURED_COLOR = "#111111"
DISPLACEMENT_PREDICTION_COLOR = "#d81b60"
FORCE_PREDICTION_COLOR = "#0072b2"
PREDICTION_MARKER_COUNT = 18

ROLE_COLORS = {
    "one_step_training": "#81c784",
    "one_step_validation": "#fbc02d",
    "one_step_internal_test": "#64b5f6",
    "one_step_pure_test": "#ce93d8",
}
ROLE_LABELS = {
    "one_step_training": "Training",
    "one_step_validation": "Validation",
    "one_step_internal_test": "Internal test",
    "one_step_pure_test": "Pure test",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.30,
    }
)


def _finish(figure, folder, file_name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / file_name, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print("  wrote", file_name, flush=True)


def _plot_measured_prediction(
    axis,
    time,
    measured,
    predicted,
    prediction_color,
    prediction_label="LSTM prediction (dashed + markers)",
):
    """Draw an overlapping comparison that remains readable in print."""
    time = np.asarray(time)
    marker_spacing = max(1, int(np.ceil(len(time) / PREDICTION_MARKER_COUNT)))
    axis.plot(
        time,
        measured,
        color=MEASURED_COLOR,
        lw=2.15,
        ls="-",
        alpha=0.92,
        label="Measured (solid black)",
        zorder=2,
    )
    axis.plot(
        time,
        predicted,
        color=prediction_color,
        lw=1.65,
        ls=(0, (7, 3)),
        marker="o",
        markersize=3.2,
        markevery=marker_spacing,
        markerfacecolor="white",
        markeredgecolor=prediction_color,
        markeredgewidth=0.9,
        label=prediction_label,
        zorder=3,
    )


def _evaluation_map(evaluations):
    return {(item["name"], item["kind"]): item for item in evaluations}


def _metric(metrics, name, kind, output, column):
    row = metrics[
        (metrics["Evaluation"] == name)
        & (metrics["Kind"] == kind)
        & (metrics["Output"] == output)
    ]
    return np.nan if row.empty else float(row.iloc[0][column])


def _record_role_label(data, name):
    if name in data.get("independent_test_names", []):
        return "INDEPENDENT TEST"
    if name in data["pure_test_names"]:
        return "PURE TEST"
    return "development"


def _value(value, digits=4):
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _shade_roles(axis, record_name, split_table):
    for block in split_table[split_table["record"] == record_name].itertuples():
        kind = {
            "training": "one_step_training",
            "validation": "one_step_validation",
            "test": "one_step_internal_test",
            "pure_test": "one_step_pure_test",
        }[block.role]
        axis.axvspan(
            block.start_time_s,
            block.end_time_s,
            color=ROLE_COLORS[kind],
            alpha=0.10,
            linewidth=0,
        )


def _combined_development(name, lookup):
    pieces = [
        lookup[(name, kind)]
        for kind in (
            "one_step_training",
            "one_step_validation",
            "one_step_internal_test",
        )
    ]
    order = np.argsort(np.concatenate([item["time"] for item in pieces]))
    return {
        field: np.concatenate([item[field] for item in pieces], axis=0)[order]
        for field in ("time", "current", "measured", "predicted")
    }


def _calculate_metrics(measured, predicted, negligible_range):
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = measured - predicted
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    maximum = float(np.max(np.abs(error)))
    signal_range = float(measured.max() - measured.min())
    centred = measured - measured.mean()
    variance = float(np.sum(centred**2))
    if signal_range < negligible_range or variance <= 0.0:
        r2 = fit = nrmse = np.nan
    else:
        r2 = 1.0 - float(np.sum(error**2)) / variance
        fit = 100.0 * (
            1.0 - float(np.linalg.norm(error)) / float(np.linalg.norm(centred))
        )
        nrmse = 100.0 * rmse / signal_range
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MaxAbsError": maximum,
        "NRMSE_percent": nrmse,
        "R2": r2,
        "Fit_percent": fit,
    }


def _plot_inventory(records, data, folder):
    columns = 4
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 2.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, record in zip(axes, records):
        first = record["pad"]
        time = record["time"][first:]
        displacement = record["outputs"][first:, 0]
        current = record["current"][first:]
        axis.plot(time, displacement, color=BLUE, lw=0.9)
        twin = axis.twinx()
        twin.plot(time, current, color=ORANGE, lw=0.65, alpha=0.85)
        pure = record["name"] in data["pure_test_names"]
        role_label = _record_role_label(data, record["name"])
        axis.set_title(
            f"{record['name']}\n{role_label}",
            color=PURPLE if pure else GRAY,
            fontsize=9,
        )
        axis.set_xlabel("time [s]")
        axis.set_ylabel("x [mm]", color=BLUE)
        twin.set_ylabel("I [A]", color=ORANGE)
        axis.grid(True)
    for axis in axes[len(records) :]:
        axis.axis("off")
    figure.suptitle(
        f"Figure 1  Complete measured-data inventory: {len(records)} unique records\n"
        "Purple titles are whole untouched pure or independent-test records",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    _finish(figure, folder, "01_complete_record_inventory.png")


def _plot_split(data, folder):
    table = data["split_table"]
    order = data["development_names"] + data["pure_test_names"]
    role_to_kind = {
        "training": "one_step_training",
        "validation": "one_step_validation",
        "test": "one_step_internal_test",
        "pure_test": "one_step_pure_test",
    }
    figure, axis = plt.subplots(figsize=(14, 0.44 * len(order) + 2.4))
    for row_number, name in enumerate(order):
        for block in table[table["record"] == name].itertuples():
            kind = role_to_kind[block.role]
            axis.barh(
                row_number,
                block.end_time_s - block.start_time_s,
                left=block.start_time_s,
                height=0.68,
                color=ROLE_COLORS[kind],
                edgecolor="white",
                linewidth=0.3,
            )
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(
        [
            f"{name}  ({_record_role_label(data, name)})"
            for name in order
        ]
    )
    axis.invert_yaxis()
    axis.set_xlabel("time within record [s]")
    axis.set_title(
        "Figure 2  Data split: every development record contributes training, "
        "validation, and internal-test blocks"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ROLE_COLORS[kind])
        for kind in ROLE_LABELS
    ]
    axis.legend(
        handles,
        list(ROLE_LABELS.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
    )
    axis.grid(True, axis="x")
    figure.tight_layout()
    _finish(figure, folder, "02_data_split_map.png")


def _plot_learning(history, best_loss, residual_trust, folder):
    frame = pd.DataFrame(history, columns=["Epoch", "TrainingLoss", "ValidationLoss"])
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.plot(frame["Epoch"], frame["TrainingLoss"], marker="o", label="training")
    axis.plot(frame["Epoch"], frame["ValidationLoss"], marker="s", label="validation")
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalised displacement-residual MSE")
    axis.set_title(
        "Figure 3  One-step LSTM convergence\n"
        f"best validation loss {best_loss:.5g}; validation trust {residual_trust:.3f}"
    )
    axis.grid(True, which="both")
    axis.legend(frameon=False)
    figure.tight_layout()
    _finish(figure, folder, "03_learning_curves.png")


def _plot_development_grids(data, lookup, metrics, folder):
    for figure_number, column, label, unit, color in (
        (4, 0, "Displacement", "mm", BLUE),
        (5, 1, "Lorentz force", "N", GREEN),
    ):
        names = data["development_names"]
        columns = 3
        rows = int(np.ceil(len(names) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(5.7 * columns, 3.0 * rows))
        axes = np.asarray(axes).reshape(-1)
        for axis, name in zip(axes, names):
            result = _combined_development(name, lookup)
            _shade_roles(axis, name, data["split_table"])
            prediction_color = (
                DISPLACEMENT_PREDICTION_COLOR
                if column == 0
                else FORCE_PREDICTION_COLOR
            )
            _plot_measured_prediction(
                axis,
                result["time"],
                result["measured"][:, column],
                result["predicted"][:, column],
                prediction_color,
            )
            role_fits = [
                _metric(metrics, name, kind, label, "Fit_percent")
                for kind in (
                    "one_step_training",
                    "one_step_validation",
                    "one_step_internal_test",
                )
            ]
            fit_text = "/".join(_value(value, 1) for value in role_fits)
            axis.set_title(f"{name}\nfit T/V/I = {fit_text}%", fontsize=9)
            axis.set_xlabel("time [s]")
            axis.set_ylabel(f"{label} [{unit}]")
            axis.grid(True)
            axis.legend(frameon=False, fontsize=7)
        for axis in axes[len(names) :]:
            axis.axis("off")
        figure.suptitle(
            f"Figure {figure_number}  Measured versus predicted {label.lower()} "
            "for every development record\n"
            "background: green training, yellow validation, blue internal test",
            fontsize=14,
        )
        figure.tight_layout(rect=[0, 0, 1, 0.975])
        suffix = "displacement" if column == 0 else "force"
        _finish(figure, folder, f"0{figure_number}_all_development_{suffix}.png")


def _plot_pure_details(data, records_by_name, lookup, metrics, folder):
    for position, name in enumerate(data["pure_test_names"], start=1):
        result = lookup[(name, "one_step_pure_test")]
        record = records_by_name[name]
        figure, axes = plt.subplots(
            4, 1, figsize=(13, 11), sharex=True,
            gridspec_kw={"height_ratios": [0.65, 1.25, 1.25, 0.9]},
        )
        axes[0].plot(result["time"], result["current"], color=ORANGE, lw=0.9)
        axes[0].set_ylabel("current [A]")
        axes[0].set_title("Input")
        for axis_index, (column, label, unit, color) in enumerate(
            ((0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN)),
            start=1,
        ):
            axis = axes[axis_index]
            prediction_color = (
                DISPLACEMENT_PREDICTION_COLOR
                if column == 0
                else FORCE_PREDICTION_COLOR
            )
            _plot_measured_prediction(
                axis,
                result["time"],
                result["measured"][:, column],
                result["predicted"][:, column],
                prediction_color,
                prediction_label="One-step LSTM prediction (dashed + markers)",
            )
            fit = _metric(metrics, name, "one_step_pure_test", label, "Fit_percent")
            rmse = _metric(metrics, name, "one_step_pure_test", label, "RMSE")
            axis.set_ylabel(f"{label} [{unit}]")
            axis.set_title(f"{label}: fit {_value(fit, 2)}%, RMSE {_value(rmse, 6)} {unit}")
            axis.grid(True)
            axis.legend(frameon=False)
        axes[3].plot(result["time"], result["measured"][:, 0] - result["predicted"][:, 0], color=BLUE, lw=0.85, label="displacement error [mm]")
        twin = axes[3].twinx()
        twin.plot(result["time"], result["measured"][:, 1] - result["predicted"][:, 1], color=GREEN, lw=0.85, label="force error [N]")
        axes[3].axhline(0, color=GRAY, lw=0.7)
        axes[3].set_ylabel("displacement error [mm]", color=BLUE)
        twin.set_ylabel("force error [N]", color=GREEN)
        axes[3].set_xlabel("time [s]")
        axes[3].set_title("Prediction error")
        axes[3].grid(True)
        figure.suptitle(
            f"Figure 6.{position}  Untouched pure-test record {name}\n"
            f"load {record['load_mass_g']:g} g; causal one-step measured feedback",
            fontsize=13,
        )
        figure.tight_layout(rect=[0, 0, 1, 0.97])
        _finish(figure, folder, f"06_{position}_pure_test_{name}.png")


def _plot_parity(data, lookup, metrics, folder):
    names = data["pure_test_names"]
    for column, label, unit, color, suffix in (
        (0, "Displacement", "mm", BLUE, "displacement"),
        (1, "Lorentz force", "N", GREEN, "force"),
    ):
        columns = 3
        rows = int(np.ceil(len(names) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 4.1 * rows))
        axes = np.asarray(axes).reshape(-1)
        for axis, name in zip(axes, names):
            result = lookup[(name, "one_step_pure_test")]
            measured = result["measured"][:, column]
            predicted = result["predicted"][:, column]
            low = float(min(measured.min(), predicted.min()))
            high = float(max(measured.max(), predicted.max()))
            span = max(high - low, 1e-9)
            low, high = low - 0.05 * span, high + 0.05 * span
            axis.plot([low, high], [low, high], color=GRAY, lw=1.0)
            axis.scatter(measured, predicted, s=3, alpha=0.30, color=color, edgecolors="none")
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_aspect("equal", adjustable="box")
            fit = _metric(metrics, name, "one_step_pure_test", label, "Fit_percent")
            axis.set_title(f"{name}\nfit {_value(fit, 2)}%", fontsize=9)
            axis.set_xlabel(f"measured [{unit}]")
            axis.set_ylabel(f"predicted [{unit}]")
            axis.grid(True)
        for axis in axes[len(names) :]:
            axis.axis("off")
        figure.suptitle(
            f"Figure 7  Pure/independent-test parity: {label.lower()}",
            fontsize=14,
        )
        figure.tight_layout()
        _finish(figure, folder, f"07_parity_{suffix}.png")


def _plot_metric_summary(data, metrics, folder):
    summary = metrics[metrics["Kind"].isin(["one_step_internal_test", "one_step_pure_test"])]
    names = data["development_names"] + data["pure_test_names"]
    positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(14, 0.72 * len(names)), 6.5))
    for offset, output, color in ((-width / 2, "Displacement", BLUE), (width / 2, "Lorentz force", GREEN)):
        values = []
        for name in names:
            kind = "one_step_pure_test" if name in data["pure_test_names"] else "one_step_internal_test"
            values.append(_metric(summary, name, kind, output, "Fit_percent"))
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        axis.bar(positions[finite] + offset, values[finite], width, color=color, label=output)
        for position in positions[~finite]:
            axis.text(position + offset, 94.7, "n/a", rotation=90, ha="center", va="top", fontsize=7)
    axis.axhline(95, color=RED, ls="--", lw=1.0, label="95% target")
    axis.set_ylim(90, 100.3)
    axis.set_ylabel("fit [%]")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{name}\n"
            f"{'IDPD' if name in data.get('independent_test_names', []) else 'PURE' if name in data['pure_test_names'] else 'internal'}"
            for name in names
        ],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    axis.set_title("Figure 8  Independent evaluation summary for every record")
    axis.grid(True, axis="y")
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    _finish(figure, folder, "08_metric_summary.png")


def _plot_family(data, records_by_name, lookup, metrics, family, number, folder):
    names = [
        name for name in data["pure_test_names"]
        if records_by_name[name]["family"] == family
    ]
    if not names:
        return
    figure, axes = plt.subplots(2, len(names), figsize=(6.4 * len(names), 7.7), squeeze=False)
    for position, name in enumerate(names):
        result = lookup[(name, "one_step_pure_test")]
        for row, (column, output, unit, color) in enumerate(((0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN))):
            axis = axes[row, position]
            prediction_color = (
                DISPLACEMENT_PREDICTION_COLOR
                if column == 0
                else FORCE_PREDICTION_COLOR
            )
            _plot_measured_prediction(
                axis,
                result["time"],
                result["measured"][:, column],
                result["predicted"][:, column],
                prediction_color,
            )
            fit = _metric(metrics, name, "one_step_pure_test", output, "Fit_percent")
            axis.set_title(f"{name}\n{output}, fit {_value(fit, 2)}%", fontsize=10)
            axis.set_xlabel("time [s]")
            axis.set_ylabel(f"{output} [{unit}]")
            axis.grid(True)
            axis.legend(frameon=False)
    title = "Step-input validation" if family == "step" else "Zero-input validation"
    figure.suptitle(f"Figure {number}  {title}: measured versus prediction", fontsize=14)
    figure.tight_layout()
    _finish(figure, folder, f"{number:02d}_{family}_validation.png")


def _plot_chirp(data, records_by_name, lookup, folder):
    candidates = [
        name for name in data["pure_test_names"]
        if records_by_name[name]["family"] in ("dc_plus_chirp", "chirp")
    ]
    if not candidates:
        return
    name = candidates[0]
    result = lookup[(name, "one_step_pure_test")]
    time = result["time"]
    step = float(np.median(np.diff(time)))
    middle = 0.5 * (time[0] + time[-1])
    zoom = (time >= middle - 0.5) & (time <= middle + 0.5)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    for column, label, unit, color in ((0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN)):
        prediction_color = (
            DISPLACEMENT_PREDICTION_COLOR
            if column == 0
            else FORCE_PREDICTION_COLOR
        )
        _plot_measured_prediction(
            axes[0, column],
            time[zoom],
            result["measured"][zoom, column],
            result["predicted"][zoom, column],
            prediction_color,
        )
        axes[0, column].set_title(f"{label}: one-second mid-chirp zoom")
        axes[0, column].set_xlabel("time [s]")
        axes[0, column].set_ylabel(f"{label} [{unit}]")
        axes[0, column].grid(True)
        axes[0, column].legend(frameon=False)
        window = np.hanning(len(time))
        frequencies = np.fft.rfftfreq(len(time), d=step)
        for signal, line_color, style, line_label in (
            (result["measured"][:, column], MEASURED_COLOR, "-", "Measured (solid black)"),
            (result["predicted"][:, column], prediction_color, (0, (7, 3)), "LSTM prediction (dashed)"),
        ):
            spectrum = np.abs(np.fft.rfft((signal - signal.mean()) * window))
            axes[1, column].semilogy(frequencies, spectrum * 2.0 / np.sum(window) + 1e-12, color=line_color, ls=style, lw=1.1, label=line_label)
        axes[1, column].set_xlim(0, 40)
        axes[1, column].set_xlabel("frequency [Hz]")
        axes[1, column].set_ylabel(f"amplitude [{unit}]")
        axes[1, column].set_title(f"{label} frequency content")
        axes[1, column].grid(True, which="both")
        axes[1, column].legend(frameon=False)
    figure.suptitle(f"Figure 11  Untouched chirp detail: {name}", fontsize=14)
    figure.tight_layout()
    _finish(figure, folder, "11_untouched_chirp_detail.png")


def _diagnostic_value(value):
    """Compact value formatting for error-figure titles."""
    value = float(value)
    if value == 0.0:
        return "0"
    return f"{value:.3e}" if abs(value) < 0.01 else f"{value:.5f}"


def _plot_prediction_error_diagnostics(data, lookup, folder):
    """Show small prediction differences directly instead of hiding overlap."""
    specifications = (
        (
            15,
            data["development_names"],
            "development",
            0,
            "Displacement",
            "mm",
            DISPLACEMENT_PREDICTION_COLOR,
            "development_displacement_error",
        ),
        (
            16,
            data["development_names"],
            "development",
            1,
            "Lorentz force",
            "N",
            FORCE_PREDICTION_COLOR,
            "development_force_error",
        ),
        (
            17,
            data["pure_test_names"],
            "pure/independent test",
            0,
            "Displacement",
            "mm",
            DISPLACEMENT_PREDICTION_COLOR,
            "pure_test_displacement_error",
        ),
        (
            18,
            data["pure_test_names"],
            "pure/independent test",
            1,
            "Lorentz force",
            "N",
            FORCE_PREDICTION_COLOR,
            "pure_test_force_error",
        ),
    )

    for number, names, role, column, output, unit, color, suffix in specifications:
        columns = 3
        rows = int(np.ceil(len(names) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.7 * columns, 3.15 * rows),
            squeeze=False,
        )
        axes = axes.reshape(-1)

        for axis, name in zip(axes, names):
            result = (
                _combined_development(name, lookup)
                if role == "development"
                else lookup[(name, "one_step_pure_test")]
            )
            time = np.asarray(result["time"], dtype=np.float64)
            error = (
                np.asarray(result["measured"][:, column], dtype=np.float64)
                - np.asarray(result["predicted"][:, column], dtype=np.float64)
            )
            if role == "development":
                _shade_roles(axis, name, data["split_table"])

            maximum_index = int(np.argmax(np.abs(error)))
            maximum_error = float(np.abs(error[maximum_index]))
            rmse = float(np.sqrt(np.mean(error**2)))
            mae = float(np.mean(np.abs(error)))
            symmetric_limit = max(1.18 * maximum_error, 1e-12)

            axis.axhline(0.0, color=MEASURED_COLOR, lw=0.8, zorder=1)
            axis.fill_between(
                time,
                0.0,
                error,
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=2,
            )
            axis.plot(time, error, color=color, lw=1.05, zorder=3)
            axis.scatter(
                [time[maximum_index]],
                [error[maximum_index]],
                color=RED,
                edgecolor="white",
                linewidth=0.7,
                s=34,
                zorder=4,
            )
            axis.set_ylim(-symmetric_limit, symmetric_limit)
            axis.set_title(
                f"{name}\n"
                f"RMSE {_diagnostic_value(rmse)}, "
                f"MAE {_diagnostic_value(mae)}, "
                f"max |e| {_diagnostic_value(maximum_error)} {unit} "
                f"at {time[maximum_index]:.4f} s",
                fontsize=8.5,
            )
            axis.set_xlabel("time [s]")
            axis.set_ylabel(f"measured - predicted [{unit}]")
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.grid(True)

        for axis in axes[len(names) :]:
            axis.axis("off")

        background_note = (
            "Pastel background: training, validation, and internal-test blocks."
            if role == "development"
            else ""
        )
        title_text = (
            f"Figure {number}  {output} prediction error: {role} records\n"
            "e(t) = measured - predicted; red dot = maximum |e|. "
            "Each panel has a symmetric y-scale around zero."
        )
        if background_note:
            title_text += f"\n{background_note}"
        figure.suptitle(
            title_text,
            fontsize=12.5,
            y=0.995,
        )
        figure.tight_layout(
            rect=[0, 0, 1, 0.90 if background_note else 0.94]
        )
        _finish(figure, folder, f"{number:02d}_{suffix}.png")


def _pooled_role_metrics(evaluations):
    rows = []
    for kind in ROLE_LABELS:
        selected = [item for item in evaluations if item["kind"] == kind]
        for column, output, unit, threshold in (
            (0, "Displacement", "mm", 2e-3),
            (1, "Lorentz force", "N", 2e-3),
        ):
            measured = np.concatenate([item["measured"][:, column] for item in selected])
            predicted = np.concatenate([item["predicted"][:, column] for item in selected])
            rows.append(
                {
                    "Kind": kind,
                    "Data role": ROLE_LABELS[kind],
                    "Output": output,
                    "Unit": unit,
                    "Samples": len(measured),
                    **_calculate_metrics(measured, predicted, threshold),
                }
            )
    return pd.DataFrame(rows)


def _plot_tables(metrics, pooled, results_folder, figures_folder):
    pooled.to_csv(Path(results_folder) / "one_step_pooled_role_metrics.csv", index=False)
    figure, axis = plt.subplots(figsize=(14, 5.4))
    axis.axis("off")
    cell_text = [
        [
            row["Data role"], row["Output"], str(int(row["Samples"])),
            _value(row["RMSE"], 6), _value(row["MAE"], 6),
            _value(row["MaxAbsError"], 6), _value(row["R2"], 4),
            _value(row["Fit_percent"], 2),
        ]
        for _, row in pooled.iterrows()
    ]
    table = axis.table(
        cellText=cell_text,
        colLabels=["data role", "output", "samples", "RMSE", "MAE", "max abs error", "R2", "fit %"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(GRAY)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#eceff1")
    axis.set_title("Figure 12  Pooled performance for every data role", fontsize=14, pad=16)
    figure.tight_layout()
    _finish(figure, figures_folder, "12_pooled_role_metric_table.png")

    ordered = metrics.copy()
    kind_order = {kind: index for index, kind in enumerate(ROLE_LABELS)}
    ordered["_kind_order"] = ordered["Kind"].map(kind_order)
    ordered = ordered.sort_values(["_kind_order", "Evaluation", "Output"])
    cell_text = []
    for _, row in ordered.iterrows():
        role_label = ROLE_LABELS[row["Kind"]]
        if row.get("Test_origin") == "Test_idpd.xlsx":
            role_label = "Independent test"
        cell_text.append(
            [
                row["Evaluation"], role_label, row["Output"],
                _value(row["RMSE"], 6), _value(row["MAE"], 6),
                _value(row["MaxAbsError"], 6), _value(row["R2"], 4),
                _value(row["Fit_percent"], 2),
            ]
        )
    figure, axis = plt.subplots(figsize=(17, 0.31 * len(cell_text) + 2.1))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=["record", "data role", "output", "RMSE", "MAE", "max abs error", "R2", "fit %"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.3)
    table.scale(1.0, 1.16)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(GRAY)
            cell.set_text_props(color="white", weight="bold")
        elif cell_text[row - 1][1] in ("Pure test", "Independent test"):
            cell.set_facecolor("#f3e5f5")
        elif row % 2:
            cell.set_facecolor("#eceff1")
    axis.set_title(
        "Figure 13  Complete metric table: training, validation, internal, pure, and optional independent tests",
        fontsize=14,
        pad=18,
    )
    figure.tight_layout()
    _finish(figure, figures_folder, "13_full_metric_table.png")


def _plot_explicit_zero_input_test(
    data, records_by_name, lookup, metrics, results_folder, figures_folder
):
    """Add one explicit Load-2 I(t)=0 test without modifying the model."""
    candidates = [
        name
        for name in data["pure_test_names"]
        if records_by_name[name]["family"] == "zero_input"
    ]
    if not candidates:
        raise RuntimeError(
            "The explicit zero-input test requires an untouched zero-input record."
        )
    # The configured reference-mass zero-input record is Load 2 in this data set.
    name = candidates[0]
    result = lookup[(name, "one_step_pure_test")]
    if float(np.max(np.abs(result["current"]))) > 1e-7:
        raise RuntimeError(f"{name} is not a true zero-current record.")

    zero_metrics = metrics[
        (metrics["Evaluation"] == name)
        & (metrics["Kind"] == "one_step_pure_test")
    ].copy()
    zero_metrics.insert(0, "Test", "explicit_zero_input")
    zero_metrics.to_csv(
        Path(results_folder) / "explicit_zero_input_test_metrics.csv", index=False
    )

    time = result["time"]
    measured = result["measured"]
    predicted = result["predicted"]
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(13, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [0.60, 1.25, 1.25, 0.95]},
    )
    axes[0].plot(time, result["current"], color=ORANGE, lw=1.2)
    axes[0].axhline(0.0, color=GRAY, lw=0.7)
    axes[0].set_ylabel("current [A]")
    axes[0].set_title("Applied test input: I(t) = 0 A for the entire record")
    axes[0].grid(True)

    for axis_index, (column, output, unit, color) in enumerate(
        ((0, "Displacement", "mm", BLUE), (1, "Lorentz force", "N", GREEN)),
        start=1,
    ):
        axis = axes[axis_index]
        prediction_color = (
            DISPLACEMENT_PREDICTION_COLOR
            if column == 0
            else FORCE_PREDICTION_COLOR
        )
        _plot_measured_prediction(
            axis,
            time,
            measured[:, column],
            predicted[:, column],
            prediction_color,
            prediction_label="One-step LSTM prediction (dashed + markers)",
        )
        fit = _metric(
            metrics, name, "one_step_pure_test", output, "Fit_percent"
        )
        rmse = _metric(metrics, name, "one_step_pure_test", output, "RMSE")
        fit_text = "n/a (flat reference)" if not np.isfinite(fit) else f"{fit:.2f}%"
        axis.set_ylabel(f"{output} [{unit}]")
        axis.set_title(
            f"{output}: fit {fit_text}; RMSE {_value(rmse, 6)} {unit}"
        )
        axis.grid(True)
        axis.legend(frameon=False)

    axes[3].plot(
        time,
        measured[:, 0] - predicted[:, 0],
        color=BLUE,
        lw=0.9,
        label="displacement error [mm]",
    )
    twin = axes[3].twinx()
    twin.plot(
        time,
        measured[:, 1] - predicted[:, 1],
        color=GREEN,
        lw=0.9,
        label="force error [N]",
    )
    axes[3].axhline(0.0, color=GRAY, lw=0.7)
    axes[3].set_ylabel("displacement error [mm]", color=BLUE)
    twin.set_ylabel("force error [N]", color=GREEN)
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("Zero-input prediction errors")
    axes[3].grid(True)

    record = records_by_name[name]
    figure.suptitle(
        "Figure 14  Explicit zero-input system test\n"
        f"{name}, load {record['load_mass_g']:g} g; initialized only from "
        "measured x0, velocity0, and force0; untouched during training",
        fontsize=14,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    _finish(figure, figures_folder, "14_explicit_zero_input_test.png")


def generate_complete_figures(
    data,
    records,
    history,
    best_loss,
    residual_trust,
    evaluations,
    metrics,
    results_folder,
    figures_folder,
):
    """Write the complete figure set after the frozen model is evaluated."""
    figures_folder = Path(figures_folder)
    figures_folder.mkdir(parents=True, exist_ok=True)
    lookup = _evaluation_map(evaluations)
    records_by_name = {record["name"]: record for record in records}

    print("\nBuilding the complete figure report ...", flush=True)
    _plot_inventory(records, data, figures_folder)
    _plot_split(data, figures_folder)
    _plot_learning(history, best_loss, residual_trust, figures_folder)
    _plot_development_grids(data, lookup, metrics, figures_folder)
    _plot_pure_details(data, records_by_name, lookup, metrics, figures_folder)
    _plot_parity(data, lookup, metrics, figures_folder)
    _plot_metric_summary(data, metrics, figures_folder)
    _plot_family(data, records_by_name, lookup, metrics, "step", 9, figures_folder)
    _plot_family(data, records_by_name, lookup, metrics, "zero_input", 10, figures_folder)
    _plot_chirp(data, records_by_name, lookup, figures_folder)
    pooled = _pooled_role_metrics(evaluations)
    _plot_tables(metrics, pooled, results_folder, figures_folder)
    _plot_explicit_zero_input_test(
        data,
        records_by_name,
        lookup,
        metrics,
        results_folder,
        figures_folder,
    )
    _plot_prediction_error_diagnostics(data, lookup, figures_folder)
    print(f"\nAll complete figures are in: {figures_folder}")


def _load_saved_report(results_folder):
    """Load the plotting-only artifacts written by a completed ``main.py`` run."""
    results_folder = Path(results_folder)
    inventory = pd.read_csv(results_folder / "record_inventory.csv")
    split_table = pd.read_csv(results_folder / "data_split.csv")
    metrics = pd.read_csv(results_folder / "one_step_metrics.csv")
    history_frame = pd.read_csv(results_folder / "one_step_training_history.csv")
    with open(
        results_folder / "one_step_plot_metadata.json", encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)

    prediction_folder = results_folder / "one_step_predictions"
    prediction_map_path = results_folder / "one_step_prediction_file_map.csv"
    prediction_lookup = None
    if prediction_map_path.exists():
        prediction_map = pd.read_csv(prediction_map_path)
        required_map_columns = {"File", "Kind", "Evaluation"}
        missing_map_columns = required_map_columns.difference(prediction_map.columns)
        if missing_map_columns:
            raise ValueError(
                f"{prediction_map_path.name} is missing "
                f"{sorted(missing_map_columns)}. Run main.py again."
            )
        prediction_lookup = {
            (str(row.Kind), str(row.Evaluation)): str(row.File)
            for row in prediction_map.itertuples(index=False)
        }
    evaluations = []
    records = []
    for row in inventory.itertuples(index=False):
        kinds = (
            ("one_step_pure_test",)
            if row.usage in ("pure_test", "independent_test")
            else (
                "one_step_training",
                "one_step_validation",
                "one_step_internal_test",
            )
        )
        pieces = []
        for kind in kinds:
            if prediction_lookup is None:
                # Backward compatibility with results from earlier versions.
                file_name = f"{kind}__{row.record}.csv"
            else:
                key = (kind, str(row.record))
                if key not in prediction_lookup:
                    raise FileNotFoundError(
                        f"No prediction-file mapping exists for {kind}, "
                        f"{row.record}. Run main.py again."
                    )
                file_name = prediction_lookup[key]
            path = prediction_folder / file_name
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path.name}. Run main.py once with this updated "
                    "version before rebuilding figures only."
                )
            frame = pd.read_csv(path)
            required = {
                "time_s",
                "current_A",
                "measured_displacement_mm",
                "predicted_displacement_mm",
                "measured_force_N",
                "predicted_force_N",
            }
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"{path.name} is from an older run and lacks "
                    f"{sorted(missing)}. Run main.py once to refresh it."
                )
            item = {
                "name": row.record,
                "kind": kind,
                "time": frame["time_s"].to_numpy(),
                "current": frame["current_A"].to_numpy(),
                "measured": frame[
                    ["measured_displacement_mm", "measured_force_N"]
                ].to_numpy(),
                "predicted": frame[
                    ["predicted_displacement_mm", "predicted_force_N"]
                ].to_numpy(),
            }
            evaluations.append(item)
            pieces.append(item)
        time = np.concatenate([item["time"] for item in pieces])
        order = np.argsort(time)
        records.append(
            {
                "name": row.record,
                "family": row.family,
                "load_mass_g": float(row.load_mass_g),
                "pad": 0,
                "time": time[order],
                "current": np.concatenate(
                    [item["current"] for item in pieces]
                )[order],
                "outputs": np.concatenate(
                    [item["measured"] for item in pieces], axis=0
                )[order],
            }
        )

    data = {
        "split_table": split_table,
        "development_names": inventory.loc[
            inventory["usage"] == "development", "record"
        ].tolist(),
        "pure_test_names": inventory.loc[
            inventory["usage"].isin(["pure_test", "independent_test"]), "record"
        ].tolist(),
        "independent_test_names": inventory.loc[
            inventory["usage"] == "independent_test", "record"
        ].tolist(),
    }
    history = list(history_frame.itertuples(index=False, name=None))
    return (
        data,
        records,
        history,
        float(metadata["best_validation_loss"]),
        float(metadata["displacement_residual_trust"]),
        evaluations,
        metrics,
    )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    results = Path(
        os.environ.get("DLSTM_RESULTS_FOLDER", str(here / "ResultsData"))
    ).resolve()
    figures = Path(
        os.environ.get("DLSTM_FIGURES_FOLDER", str(here / "FiguresResults"))
    ).resolve()
    loaded = _load_saved_report(results)
    generate_complete_figures(
        data=loaded[0],
        records=loaded[1],
        history=loaded[2],
        best_loss=loaded[3],
        residual_trust=loaded[4],
        evaluations=loaded[5],
        metrics=loaded[6],
        results_folder=results,
        figures_folder=figures,
    )
