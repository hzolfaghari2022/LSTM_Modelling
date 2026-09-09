"""Run all 13 standalone cumulative feature-ablation projects and compare them."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SHARED_CODE = HERE / "SharedCode"
sys.path.insert(0, str(SHARED_CODE))

from ablation_spec import ONE_STEP_FEATURE_NAMES, trainable_parameter_count
from plot_style import apply_publication_style


def case_folder(feature_count):
    suffix = "Feature" if feature_count == 1 else "Features"
    return HERE / f"Model_{feature_count:02d}_{suffix}"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="reuse a case only when its summary CSV already exists",
    )
    return parser.parse_args()


def run_case(feature_count, skip_completed=False):
    folder = case_folder(feature_count)
    summary_path = folder / "ResultsData" / "ablation_case_summary.csv"
    if skip_completed and summary_path.exists():
        print(f"\nReusing completed {folder.name}.", flush=True)
        return

    print("\n" + "#" * 96, flush=True)
    print(f"RUNNING {folder.name}", flush=True)
    print("#" * 96, flush=True)
    environment = os.environ.copy()
    # A full all-case run pushes once at the end.  Running a folder directly
    # still retains that folder's normal automatic push behavior.
    environment["DLSTM_SKIP_GITHUB_PUSH"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        [sys.executable, str(folder / "main.py")],
        cwd=folder,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{folder.name} failed with exit code {completed.returncode}. "
            "Its earlier completed results were not used as a replacement."
        )
    if not summary_path.exists():
        raise RuntimeError(f"{folder.name} completed but did not create {summary_path.name}.")


def pooled_row(folder, role, output):
    path = folder / "ResultsData" / "one_step_pooled_role_metrics.csv"
    table = pd.read_csv(path)
    selected = table[(table["Data role"] == role) & (table["Output"] == output)]
    if len(selected) != 1:
        raise RuntimeError(f"Missing unique {role}/{output} row in {path}.")
    return selected.iloc[0]


def worst_pure_displacement(folder):
    path = folder / "ResultsData" / "one_step_metrics.csv"
    table = pd.read_csv(path)
    selected = table[
        (table["Kind"] == "one_step_pure_test")
        & (table["Output"] == "Displacement")
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No pure-test displacement rows found in {path}.")
    selected["Fit_sort"] = selected["Fit_percent"].fillna(-np.inf)
    row = selected.sort_values(["Fit_sort", "RMSE"], ascending=[True, False]).iloc[0]
    return row


def collect_comparison():
    rows = []
    for count in range(13, 0, -1):
        folder = case_folder(count)
        training = pooled_row(folder, "Training", "Displacement")
        validation = pooled_row(folder, "Validation", "Displacement")
        pure = pooled_row(folder, "Pure test", "Displacement")
        validation_force = pooled_row(folder, "Validation", "Lorentz force")
        pure_force = pooled_row(folder, "Pure test", "Lorentz force")
        worst = worst_pure_displacement(folder)
        active = list(ONE_STEP_FEATURE_NAMES[:count])
        removed = list(ONE_STEP_FEATURE_NAMES[count:])
        rows.append(
            {
                "model": folder.name,
                "feature_count": count,
                "trainable_parameters": trainable_parameter_count(count),
                "active_features": "; ".join(active),
                "removed_features": "; ".join(removed) if removed else "None",
                "training_displacement_RMSE": float(training["RMSE"]),
                "training_displacement_Fit_percent": float(training["Fit_percent"]),
                "validation_displacement_RMSE": float(validation["RMSE"]),
                "validation_displacement_MAE": float(validation["MAE"]),
                "validation_displacement_R2": float(validation["R2"]),
                "validation_displacement_Fit_percent": float(validation["Fit_percent"]),
                "pure_displacement_RMSE": float(pure["RMSE"]),
                "pure_displacement_MAE": float(pure["MAE"]),
                "pure_displacement_R2": float(pure["R2"]),
                "pure_displacement_Fit_percent": float(pure["Fit_percent"]),
                "worst_pure_test": str(worst["Evaluation"]),
                "worst_pure_displacement_RMSE": float(worst["RMSE"]),
                "worst_pure_displacement_Fit_percent": float(worst["Fit_percent"]),
                "validation_force_RMSE": float(validation_force["RMSE"]),
                "validation_force_Fit_percent": float(validation_force["Fit_percent"]),
                "pure_force_RMSE": float(pure_force["RMSE"]),
                "pure_force_Fit_percent": float(pure_force["Fit_percent"]),
            }
        )
    table = pd.DataFrame(rows)
    table["validation_rank"] = (
        table["validation_displacement_RMSE"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return table


def make_figures(table, best_count):
    apply_publication_style()
    figures = HERE / "ComparisonFigures"
    figures.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("feature_count")

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
    axes[0, 0].plot(
        ordered["feature_count"], ordered["validation_displacement_Fit_percent"],
        "o-", color="#0072B2", label="Validation",
    )
    axes[0, 0].plot(
        ordered["feature_count"], ordered["pure_displacement_Fit_percent"],
        "s--", color="#D55E00", label="Pure test (confirmation only)",
    )
    axes[0, 0].set_ylabel("Displacement fit [%]")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        ordered["feature_count"], ordered["validation_displacement_RMSE"],
        "o-", color="#0072B2", label="Validation",
    )
    axes[0, 1].plot(
        ordered["feature_count"], ordered["pure_displacement_RMSE"],
        "s--", color="#D55E00", label="Pure test (confirmation only)",
    )
    axes[0, 1].set_ylabel("Displacement RMSE [mm]")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        ordered["feature_count"], ordered["worst_pure_displacement_Fit_percent"],
        "D-", color="#CC79A7",
    )
    axes[1, 0].set_ylabel("Worst pure-test displacement fit [%]")
    axes[1, 1].plot(
        ordered["feature_count"], ordered["trainable_parameters"],
        "o-", color="#6A1B9A",
    )
    axes[1, 1].set_ylabel("Trainable parameters")

    for axis in axes.ravel():
        axis.set_xlabel("Retained causal features")
        axis.set_xticks(range(1, 14))
        axis.grid(True, alpha=0.28)
        axis.axvline(best_count, color="#009E73", ls=":", lw=1.8)
    figure.suptitle(
        "Cumulative Feature-Ablation Comparison",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(figures / "01_all_13_models_accuracy.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    axis.plot(
        ordered["feature_count"], ordered["training_displacement_RMSE"],
        "o-", color="#56B4E9", label="Training RMSE",
    )
    axis.plot(
        ordered["feature_count"], ordered["validation_displacement_RMSE"],
        "s-", color="#D55E00", label="Validation RMSE",
    )
    best = ordered[ordered["feature_count"] == best_count].iloc[0]
    axis.scatter(
        [best_count], [best["validation_displacement_RMSE"]],
        s=140, marker="*", color="#009E73", zorder=5,
        label=f"Best validation result ({best_count} features)",
    )
    axis.set_xlabel("Model complexity: number of retained causal features")
    axis.set_ylabel("Displacement RMSE [mm]")
    axis.set_xticks(range(1, 14))
    axis.grid(True, alpha=0.28)
    axis.legend(frameon=False)
    axis.set_title(
        "Training and Validation Error versus Feature Count",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        figures / "02_training_validation_error_vs_features.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)

    matrix = np.zeros((13, 13), dtype=float)
    counts = list(range(13, 0, -1))
    for row, count in enumerate(counts):
        matrix[row, :count] = 1.0
    figure, axis = plt.subplots(figsize=(13.4, 7.0))
    axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(13))
    axis.set_xticklabels(ONE_STEP_FEATURE_NAMES, rotation=42, ha="right")
    axis.set_yticks(range(13))
    axis.set_yticklabels([f"{count} features" for count in counts])
    axis.set_title(
        "Cumulative Retained Features in the 13 Standalone Models",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel("Blue = retained; white = removed")
    figure.tight_layout()
    figure.savefig(figures / "03_feature_retention_matrix.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_report(table, best_count):
    results = HERE / "ComparisonResults"
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "all_13_models_comparison.csv", index=False)
    ranking = table.sort_values(
        ["validation_displacement_RMSE", "feature_count"],
        ascending=[True, True],
    )
    ranking.to_csv(results / "validation_model_ranking.csv", index=False)

    best = ranking.iloc[0]
    complete = table[table["feature_count"] == 13].iloc[0]
    report = (
        "V15 CUMULATIVE FEATURE-ABLATION COMPARISON\n"
        "===========================================\n\n"
        "Thirteen independent projects were trained: 13, 12, ..., 1 retained "
        "causal features. Each reduced case keeps the first N features in the "
        "documented V15 order.\n\n"
        f"Best validation displacement RMSE: {best['model']}\n"
        f"  RMSE = {best['validation_displacement_RMSE']:.8g} mm\n"
        f"  Fit  = {best['validation_displacement_Fit_percent']:.6g}%\n"
        f"  R2   = {best['validation_displacement_R2']:.8g}\n"
        f"  Pure-test confirmation fit = {best['pure_displacement_Fit_percent']:.6g}%\n"
        f"  Worst individual pure-test fit = "
        f"{best['worst_pure_displacement_Fit_percent']:.6g}% "
        f"({best['worst_pure_test']})\n\n"
        "Complete 13-feature model:\n"
        f"  Validation RMSE = {complete['validation_displacement_RMSE']:.8g} mm\n"
        f"  Validation fit  = {complete['validation_displacement_Fit_percent']:.6g}%\n"
        f"  Pure-test fit   = {complete['pure_displacement_Fit_percent']:.6g}%\n\n"
        "Scientific interpretation:\n"
        "- The best feature count is selected from validation displacement RMSE, "
        "not from pure-test results.\n"
        "- Pure tests are shown only to confirm the already selected model.\n"
        "- These are cumulative removals in one declared order; they do not prove "
        "that an individually removed feature is universally unimportant.\n"
        "- The LSTM feature ablation affects displacement prediction only. Lorentz "
        "force is produced by the unchanged separate causal force rule, so its "
        "metrics should remain essentially identical across cases.\n"
    )
    (results / "COMPARISON_REPORT.txt").write_text(report, encoding="utf-8")
    return report


def push_all_results():
    if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") == "1":
        print("GitHub push disabled by DLSTM_SKIP_GITHUB_PUSH=1.")
        return
    try:
        from github_push import GitPushError, push_simulation

        run_time = datetime.now().astimezone()
        (HERE / "last_github_update.txt").write_text(
            "Latest completed all-model ablation run: "
            f"{run_time.isoformat(timespec='microseconds')}\n",
            encoding="utf-8",
        )
        push_simulation(
            HERE,
            commit_message=(
                "Update all 13 V15 feature-ablation models - "
                f"{run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            ),
        )
    except (GitPushError, ImportError) as error:
        print("\nIMPORTANT: all models completed, but GitHub push failed:")
        print(error)
        print("Run 'python push_now.py' after correcting the Git configuration.")


def main():
    arguments = parse_arguments()
    for count in range(13, 0, -1):
        run_case(count, skip_completed=arguments.skip_completed)

    table = collect_comparison()
    best_count = int(
        table.sort_values(
            ["validation_displacement_RMSE", "feature_count"],
            ascending=[True, True],
        ).iloc[0]["feature_count"]
    )
    make_figures(table, best_count)
    report = write_report(table, best_count)
    print("\n" + report)
    print("Comparison tables:", HERE / "ComparisonResults")
    print("Comparison figures:", HERE / "ComparisonFigures")
    push_all_results()
    print("\nAll 13 models completed and compared.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAll-model run interrupted by the user.")
        raise SystemExit(130)
