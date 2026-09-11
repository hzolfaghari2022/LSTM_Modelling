"""Create advisor-ready figures from the completed 13-case comparison."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "ComparisonResults"
OUTPUT = HERE / "ComparisonFigures" / "ReportFigures"
OUTPUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Current",
    "Current change",
    "DC current estimate",
    "Mass ratio",
    "Inverse mass ratio",
    "Elapsed time",
    "Startup indicator",
    "Previous displacement",
    "Previous velocity",
    "Previous acceleration",
    "Previous force",
    "Force change",
    "Displacement baseline",
]

BLUE = "#1769aa"
ORANGE = "#ef6c00"
GREEN = "#2e7d32"
RED = "#c62828"
PURPLE = "#6a1b9a"
GRAY = "#5f6b73"
LIGHT = "#eef3f7"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Garamond", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
    }
)


def load_data():
    required = {
        "all_case_metrics.csv": RESULTS / "all_case_metrics.csv",
        "per_pure_test_comparison.csv": RESULTS / "per_pure_test_comparison.csv",
        "baseline_and_lstm_correction.csv": RESULTS / "baseline_and_lstm_correction.csv",
        "prediction_difference_vs_F13.csv": RESULTS / "prediction_difference_vs_F13.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing comparison output(s): " + ", ".join(missing)
            + ". Run the root main.py first, or use: python main.py --compare-only"
        )

    metrics = pd.read_csv(required["all_case_metrics.csv"])
    pure = pd.read_csv(required["per_pure_test_comparison.csv"])
    correction = pd.read_csv(required["baseline_and_lstm_correction.csv"])
    differences = pd.read_csv(required["prediction_difference_vs_F13.csv"])

    observed = set(pd.to_numeric(metrics["feature_count"]).astype(int))
    expected = set(range(1, 14))
    if observed != expected:
        raise ValueError(
            "The comparison must contain all feature counts F01 through F13; "
            f"found {sorted(observed)}."
        )
    return metrics, pure, correction, differences


def displacement_rows(metrics, role):
    return (
        metrics[
            (metrics["output"] == "Displacement")
            & (metrics["role"] == role)
        ]
        .sort_values("feature_count")
        .reset_index(drop=True)
    )


def save_figure(figure, stem, pdf):
    figure.savefig(OUTPUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    figure.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    pdf.savefig(figure, bbox_inches="tight")


def feature_schedule_figure(pdf):
    matrix = np.tril(np.ones((13, 13), dtype=int)).T
    figure, axis = plt.subplots(figsize=(14, 8.5))
    axis.imshow(matrix, cmap=ListedColormap(["#ffffff", "#cfe8f3"]), aspect="auto")

    for row in range(13):
        for column in range(13):
            if matrix[row, column]:
                symbol = "+" if row == column else "x"
                color = RED if row == column else BLUE
                weight = "bold" if row == column else "normal"
                axis.text(column, row, symbol, ha="center", va="center", color=color,
                          fontsize=13, fontweight=weight)

    axis.set_xticks(range(13), [f"F{count:02d}" for count in range(1, 14)])
    axis.set_yticks(range(13), FEATURES)
    axis.set_xlabel("Simulation case (feature count increases from 1 to 13)")
    axis.set_title(
        "Cumulative Feature-Ablation Design\n"
        "Each configuration retains all preceding features and introduces one additional feature",
        fontweight="bold",
        pad=18,
    )
    axis.set_xticks(np.arange(-0.5, 13, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 13, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.axhspan(6.5, 12.5, color="#fff3e0", alpha=0.24)
    axis.text(
        12.7,
        9.5,
        "Measured-feedback\nfeatures",
        va="center",
        ha="left",
        fontsize=11,
        color=GRAY,
    )
    axis.text(
        12.7,
        3.0,
        "Excitation, load,\nand time features",
        va="center",
        ha="left",
        fontsize=11,
        color=GRAY,
    )
    figure.text(
        0.5,
        0.015,
        "The actuator excitation is current; the remaining LSTM channels are causal transformations, configuration descriptors, or measured-feedback quantities.",
        ha="center",
        fontsize=10.5,
        color=GRAY,
    )
    figure.subplots_adjust(left=0.23, right=0.88, bottom=0.12, top=0.86)
    save_figure(figure, "01_feature_ablation_design", pdf)
    plt.close(figure)


def accuracy_dashboard(metrics, pure, pdf):
    validation = displacement_rows(metrics, "validation")
    pure_pooled = displacement_rows(metrics, "pure_test")
    worst = pure.loc[
        pure.groupby("feature_count")["Fit_percent"].idxmin()
    ].sort_values("feature_count")
    cases = validation["feature_count"].to_numpy()

    validation_values = validation["RMSE"].to_numpy(dtype=float)
    adjacent_improvements = 100.0 * (
        validation_values[:-1] - validation_values[1:]
    ) / validation_values[:-1]
    improvement_index = int(np.argmax(adjacent_improvements))
    improvement_from = improvement_index + 1
    improvement_to = improvement_index + 2
    improvement_percent = float(adjacent_improvements[improvement_index])
    improvement_feature = FEATURES[improvement_to - 1]
    improvement_rmse = float(validation_values[improvement_to - 1])
    validation_best = validation.loc[validation["RMSE"].idxmin()]
    pure_best = pure_pooled.loc[pure_pooled["RMSE"].idxmin()]
    validation_best_count = int(validation_best["feature_count"])
    pure_best_count = int(pure_best["feature_count"])
    validation_best_rmse = float(validation_best["RMSE"])
    pure_best_rmse = float(pure_best["RMSE"])
    selected_pure_rmse = float(
        pure_pooled[pure_pooled["feature_count"] == validation_best_count]["RMSE"].iloc[0]
    )
    pure_penalty = 100.0 * (selected_pure_rmse - pure_best_rmse) / pure_best_rmse
    selected_worst = worst[worst["feature_count"] == validation_best_count].iloc[0]
    selected_worst_fit = float(selected_worst["Fit_percent"])
    selected_worst_name = str(selected_worst["record"]).replace("_", " ")
    if validation_best_count == pure_best_count:
        generalization_text = (
            f"F{validation_best_count:02d} gives the lowest\n"
            "validation and pure-test RMSE"
        )
    else:
        generalization_text = (
            f"F{validation_best_count:02d} pure-test RMSE is\n"
            f"{pure_penalty:.1f}% higher than F{pure_best_count:02d}"
        )

    figure = plt.figure(figsize=(16, 9))
    grid = figure.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 0.95], hspace=0.34, wspace=0.32)
    rmse_axis = figure.add_subplot(grid[0, :2])
    fit_axis = figure.add_subplot(grid[1, :2])
    card_axis = figure.add_subplot(grid[:, 2])

    rmse_axis.semilogy(cases, validation["RMSE"], "o-", lw=2.4, ms=7,
                       color=BLUE, label="Validation RMSE")
    rmse_axis.semilogy(cases, pure_pooled["RMSE"], "s--", lw=2.2, ms=7,
                       color=ORANGE, label="Pure-test RMSE")
    rmse_axis.axvline(improvement_to, color=GREEN, lw=1.3, ls=":")
    rmse_axis.annotate(
        f"F{improvement_to:02d} adds {improvement_feature.lower()}\n"
        f"Validation RMSE decreases by {improvement_percent:.1f}%",
        xy=(improvement_to, improvement_rmse),
        xytext=(max(1.0, improvement_to - 2.7), improvement_rmse * 1.9),
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
        color=GREEN, fontweight="bold",
    )
    rmse_axis.set_ylabel("Pooled displacement RMSE [mm] (log scale)")
    rmse_axis.set_xticks(cases)
    rmse_axis.grid(True, which="both")
    rmse_axis.legend(frameon=False, ncol=2, loc="upper right")
    rmse_axis.set_title("Displacement RMSE Across Feature Configurations", fontweight="bold")

    fit_axis.plot(cases, worst["Fit_percent"], "o-", lw=2.3, ms=7, color=PURPLE)
    fit_axis.axhline(95.0, color=RED, lw=1.4, ls="--", label="95% acceptance threshold")
    fit_axis.scatter([validation_best_count], [selected_worst_fit],
                     s=110, color=RED, zorder=5)
    fit_axis.annotate(
        f"F{validation_best_count:02d} worst case:\n{selected_worst_name}, {selected_worst_fit:.2f}% fit",
        xy=(validation_best_count, selected_worst_fit),
        xytext=(max(1.0, validation_best_count - 3.9), max(95.4, selected_worst_fit - 0.75)),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
        color=RED,
    )
    fit_axis.set_ylim(94.7, 100.0)
    fit_axis.set_xticks(cases)
    fit_axis.set_xlabel("Number of retained features")
    fit_axis.set_ylabel("Worst pure-test displacement fit [%]")
    fit_axis.grid(True)
    fit_axis.legend(frameon=False, loc="lower left")
    fit_axis.set_title("Worst-Case Pure-Test Displacement Fit", fontweight="bold")

    card_axis.axis("off")
    card_axis.set_xlim(0, 1)
    card_axis.set_ylim(0, 1)
    card_axis.text(0.04, 0.96, "Summary", fontsize=17, fontweight="bold", va="top")

    cards = [
        (0.73, GREEN, "Largest improvement", f"F{improvement_from:02d} → F{improvement_to:02d}\n+ {improvement_feature.lower()}\n{improvement_percent:.1f}% lower validation RMSE"),
        (0.50, ORANGE, "Best pure-test RMSE", f"F{pure_best_count:02d}: {pure_best_rmse:.6f} mm\nDescriptive confirmation"),
        (0.27, BLUE, "Best validation RMSE", f"F{validation_best_count:02d}: {validation_best_rmse:.6f} mm\nValidation-selected model"),
        (0.04, RED, "Generalization observation", generalization_text),
    ]
    for y, color, heading, body in cards:
        card_axis.add_patch(
            plt.Rectangle((0.03, y), 0.93, 0.17, facecolor="white", edgecolor=color,
                          lw=2.0, transform=card_axis.transAxes)
        )
        card_axis.text(0.08, y + 0.145, heading, color=color, fontweight="bold",
                       fontsize=12.5, transform=card_axis.transAxes, va="top")
        card_axis.text(0.08, y + 0.103, body, color="#263238", fontsize=11,
                       transform=card_axis.transAxes, va="top", linespacing=1.35)

    figure.suptitle(
        "Cumulative LSTM Feature-Ablation Results",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5, 0.012,
        "All cases use the same split and one random seed. Feature selection must use validation data; pure tests are final confirmation. Force is excluded because it is computed by a separate causal rule.",
        ha="center", color=GRAY, fontsize=10.5,
    )
    figure.subplots_adjust(top=0.90, bottom=0.09, left=0.07, right=0.98)
    save_figure(figure, "02_feature_ablation_results", pdf)
    plt.close(figure)


def interpretation_figure(metrics, pure, correction, differences, pdf):
    validation = displacement_rows(metrics, "validation")
    pure_pooled = displacement_rows(metrics, "pure_test")
    cases = validation["feature_count"].to_numpy()
    val_best = int(validation.loc[validation["RMSE"].idxmin(), "feature_count"])
    pure_best = int(pure_pooled.loc[pure_pooled["RMSE"].idxmin(), "feature_count"])
    reduced = differences[differences["feature_count"] < 13]
    maximum_difference = float(reduced["prediction_difference_max_abs_mm"].max())
    validation_values = validation["RMSE"].to_numpy(dtype=float)
    adjacent_improvements = 100.0 * (
        validation_values[:-1] - validation_values[1:]
    ) / validation_values[:-1]
    improvement_to = int(np.argmax(adjacent_improvements)) + 2
    improvement_feature = FEATURES[improvement_to - 1].lower()
    if val_best == pure_best:
        validation_pure_text = (
            f"F{val_best:02d} minimizes both validation and pure-test RMSE."
        )
    else:
        validation_pure_text = (
            f"F{val_best:02d} minimizes validation RMSE, whereas "
            f"F{pure_best:02d} minimizes pure-test RMSE."
        )

    figure, axes = plt.subplots(1, 2, figsize=(15, 7.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    axis = axes[0]
    bars = axis.bar(cases, pure_pooled["RMSE"] * 1000.0, color="#a7cbe2", edgecolor=BLUE)
    bars[pure_best - 1].set_facecolor("#ffcc80")
    bars[pure_best - 1].set_edgecolor(ORANGE)
    bars[val_best - 1].set_facecolor("#90caf9")
    bars[val_best - 1].set_edgecolor(BLUE)
    axis.axhline(
        float(correction["pure_test_baseline_RMSE_mm"].iloc[0]) * 1000.0,
        color=GRAY, ls="--", lw=1.8, label="Physical baseline without LSTM",
    )
    axis.set_xticks(cases)
    axis.set_xlabel("Number of retained features")
    axis.set_ylabel("Pure-test displacement RMSE [µm]")
    axis.grid(True, axis="y")
    axis.legend(frameon=False)
    axis.set_title("Pure-Test Displacement Error Across Feature Configurations", fontweight="bold")
    baseline_um = float(correction["pure_test_baseline_RMSE_mm"].iloc[0]) * 1000.0
    upper = max(float(np.max(pure_pooled["RMSE"] * 1000.0)), baseline_um) * 1.12
    axis.set_ylim(0.0, upper)
    axis.annotate(f"Lowest pure-test error\nF{pure_best:02d}",
                  xy=(pure_best, bars[pure_best - 1].get_height()),
                  xytext=(max(1.0, pure_best - 2.9), upper * 0.73),
                  arrowprops=dict(arrowstyle="->", color=ORANGE),
                  color=ORANGE, fontweight="bold")
    axis.annotate(f"Validation-selected\nF{val_best:02d}",
                  xy=(val_best, bars[val_best - 1].get_height()),
                  xytext=(max(1.0, val_best - 1.3), upper * 0.80),
                  arrowprops=dict(arrowstyle="->", color=BLUE),
                  color=BLUE, fontweight="bold")

    note = axes[1]
    note.axis("off")
    note.set_xlim(0, 1)
    note.set_ylim(0, 1)
    note.text(0.02, 0.96, "Principal Observations", fontsize=17, fontweight="bold", va="top")
    statements = [
        (GREEN, "1. Distinct model responses", f"Reduced-feature predictions differ from F13 by as much as {maximum_difference:.5g} mm."),
        (BLUE, "2. Dominant accuracy transition", f"The largest validation improvement occurs when {improvement_feature} is introduced in F{improvement_to:02d}."),
        (ORANGE, "3. Validation and pure-test behavior", validation_pure_text),
        (PURPLE, "4. Pure-test accuracy", f"The lowest worst-case displacement fit is {float(pure.groupby('feature_count')['Fit_percent'].min().min()):.2f}% across all configurations."),
        (RED, "5. Statistical limitation", "The present comparison uses one seed; repeated-seed analysis is required to quantify uncertainty."),
    ]
    y = 0.84
    for color, heading, body in statements:
        note.text(0.03, y, heading, color=color, fontsize=12.5, fontweight="bold", va="top")
        note.text(0.03, y - 0.045, body, color="#263238", fontsize=11.2,
                  va="top", wrap=True, linespacing=1.35)
        y -= 0.165
    note.text(
        0.03, 0.015,
        "Scope: F01 uses current as the sole LSTM correction feature; the hybrid one-step predictor continues to use the displacement-history physical baseline.",
        color=GRAY, fontsize=10.2, va="bottom", wrap=True,
    )

    figure.suptitle("Interpretation of the Cumulative Feature-Ablation Study",
                    fontsize=20, fontweight="bold", y=0.98)
    figure.text(
        0.5, 0.925,
        "Purpose: quantify feature sensitivity, identify influential or redundant inputs, and assess whether a simpler model can preserve predictive accuracy.",
        ha="center", color=GRAY, fontsize=11.2,
    )
    figure.tight_layout(rect=[0, 0.03, 1, 0.88])
    save_figure(figure, "03_feature_ablation_interpretation", pdf)
    plt.close(figure)


def generate_report_figures():
    """Generate the three report figures from the latest comparison CSV files."""
    metrics, pure, correction, differences = load_data()
    with PdfPages(OUTPUT / "V15_ablation_report_figures.pdf") as pdf:
        feature_schedule_figure(pdf)
        accuracy_dashboard(metrics, pure, pdf)
        interpretation_figure(metrics, pure, correction, differences, pdf)
    print("Created report figures in", OUTPUT)
    return OUTPUT


def main():
    generate_report_figures()


if __name__ == "__main__":
    main()
