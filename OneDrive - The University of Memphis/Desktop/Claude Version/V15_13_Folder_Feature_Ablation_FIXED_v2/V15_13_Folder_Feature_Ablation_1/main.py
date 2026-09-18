"""Run all 13 standalone cumulative feature-ablation projects and compare them."""

import argparse
from datetime import datetime
import json
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

DEVELOPMENT_WORKBOOK_NAME = "Total_Data.xlsx"
INDEPENDENT_WORKBOOK_NAME = "Test_idpd.xlsx"
DEVELOPMENT_WORKBOOK_ENV = "DLSTM_TOTAL_DATA"
INDEPENDENT_WORKBOOK_ENV = "DLSTM_INDEPENDENT_DATA"

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
    parser.add_argument(
        "--data",
        type=Path,
        help=(
            "path to the shared Total_Data.xlsx workbook; when omitted, the "
            "file must be beside this top-level main.py"
        ),
    )
    parser.add_argument(
        "--independent-data",
        type=Path,
        help="optional path to Test_idpd.xlsx",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="verify the workbook paths and exit without training",
    )
    return parser.parse_args()


def _resolve_workbook(argument, environment_name, default_path, required):
    """Resolve one workbook without silently selecting a similarly named file."""
    source = argument
    if source is None:
        environment_value = os.environ.get(environment_name, "").strip()
        source = Path(environment_value) if environment_value else default_path

    chosen = Path(source).expanduser().resolve()
    if chosen.is_file() and chosen.name == default_path.name:
        return chosen
    if not required and argument is None and not os.environ.get(environment_name, "").strip():
        return None

    kind = "required development" if required else "optional independent-test"
    message = [
        f"The exact {kind} workbook was not found:",
        f"  {chosen}",
        f"Expected filename: {default_path.name}",
    ]
    if required:
        message.extend(
            [
                "",
                f"Copy your unchanged {DEVELOPMENT_WORKBOOK_NAME} once beside:",
                f"  {HERE / 'main.py'}",
                "",
                "or provide its existing location:",
                '  python main.py --data "C:\\full\\path\\Total_Data.xlsx"',
                "",
                "Do not copy the workbook into all 13 model folders.",
            ]
        )
    raise FileNotFoundError("\n".join(message))


def resolve_workbooks(arguments):
    development = _resolve_workbook(
        arguments.data,
        DEVELOPMENT_WORKBOOK_ENV,
        HERE / DEVELOPMENT_WORKBOOK_NAME,
        required=True,
    )
    independent = _resolve_workbook(
        arguments.independent_data,
        INDEPENDENT_WORKBOOK_ENV,
        HERE / INDEPENDENT_WORKBOOK_NAME,
        required=False,
    )
    return development, independent


def run_case(
    feature_count,
    development_workbook,
    independent_workbook=None,
    skip_completed=False,
):
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
    environment[DEVELOPMENT_WORKBOOK_ENV] = str(development_workbook)
    if independent_workbook is not None:
        environment[INDEPENDENT_WORKBOOK_ENV] = str(independent_workbook)
    else:
        environment.pop(INDEPENDENT_WORKBOOK_ENV, None)
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
        with open(
            folder / "ResultsData" / "one_step_plot_metadata.json",
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)
        if int(metadata["feature_count"]) != count:
            raise RuntimeError(
                f"Ablation audit failed for {folder.name}: requested {count} "
                f"features but its saved model reports {metadata['feature_count']}."
            )
        active = list(ONE_STEP_FEATURE_NAMES[:count])
        removed = list(ONE_STEP_FEATURE_NAMES[count:])
        rows.append(
            {
                "model": folder.name,
                "feature_count": count,
                "trainable_parameters": trainable_parameter_count(count),
                "active_features": "; ".join(active),
                "removed_features": "; ".join(removed) if removed else "None",
                "actual_lstm_input_features": int(metadata["feature_count"]),
                "residual_trust_used": float(
                    metadata["displacement_residual_trust"]
                ),
                "validation_selected_residual_trust": float(
                    metadata["validation_selected_displacement_residual_trust"]
                ),
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
    full_prediction_folder = (
        case_folder(13) / "ResultsData" / "one_step_predictions"
    )
    difference_rmse = []
    difference_max = []
    for count in table["feature_count"]:
        current_prediction_folder = (
            case_folder(int(count)) / "ResultsData" / "one_step_predictions"
        )
        squared_differences = []
        absolute_differences = []
        for full_path in sorted(full_prediction_folder.glob("one_step_pure_test__*.csv")):
            current_path = current_prediction_folder / full_path.name
            if not current_path.exists():
                raise RuntimeError(f"Missing matching prediction file: {current_path}")
            full_values = pd.read_csv(full_path)["predicted_displacement_mm"].to_numpy()
            current_values = pd.read_csv(current_path)["predicted_displacement_mm"].to_numpy()
            if full_values.shape != current_values.shape:
                raise RuntimeError(f"Prediction length mismatch: {current_path}")
            delta = current_values - full_values
            squared_differences.extend(np.square(delta).tolist())
            absolute_differences.extend(np.abs(delta).tolist())
        difference_rmse.append(float(np.sqrt(np.mean(squared_differences))))
        difference_max.append(float(np.max(absolute_differences)))
    table["prediction_RMSE_difference_vs_13feature_mm"] = difference_rmse
    table["prediction_max_difference_vs_13feature_mm"] = difference_max
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

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    axis.semilogy(
        ordered["feature_count"],
        np.maximum(
            ordered["prediction_RMSE_difference_vs_13feature_mm"], 1e-16
        ),
        "o-",
        color="#6A1B9A",
    )
    axis.set_xlabel("Retained LSTM features")
    axis.set_ylabel("Prediction difference from 13-feature model [mm]")
    axis.set_xticks(range(1, 14))
    axis.grid(True, which="both", alpha=0.28)
    axis.set_title(
        "Does Feature Removal Actually Change the Prediction?",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        figures / "04_prediction_difference_audit.png",
        dpi=240,
        bbox_inches="tight",
    )
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
    reduced = table[table["feature_count"] < 13]
    exact_same = bool(
        np.all(reduced["prediction_max_difference_vs_13feature_mm"] <= 1e-12)
    )
    equality_note = (
        "- WARNING: all reduced displacement predictions are numerically "
        "identical to the 13-feature prediction. Check the saved audit columns.\n"
        if exact_same
        else "- The prediction-difference audit confirms that feature removal "
        "changed the displacement predictions.\n"
    )
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
        "metrics must remain identical across cases.\n"
        + equality_note
        + "- More features are not guaranteed to improve validation accuracy; "
        "redundant or noisy features can have little effect or reduce accuracy.\n"
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
    development_workbook, independent_workbook = resolve_workbooks(arguments)
    print(f"Shared development workbook: {development_workbook}")
    if independent_workbook is None:
        print("Optional independent-test workbook: not present")
    else:
        print(f"Optional independent-test workbook: {independent_workbook}")
    if arguments.check_data:
        print("Data-path check passed. No models were trained.")
        return

    for count in range(13, 0, -1):
        run_case(
            count,
            development_workbook,
            independent_workbook,
            skip_completed=arguments.skip_completed,
        )

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
    except FileNotFoundError as error:
        print("\nDATA SETUP ERROR")
        print("=" * 80)
        print(error)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        print("\nAll-model run interrupted by the user.")
        raise SystemExit(130)
