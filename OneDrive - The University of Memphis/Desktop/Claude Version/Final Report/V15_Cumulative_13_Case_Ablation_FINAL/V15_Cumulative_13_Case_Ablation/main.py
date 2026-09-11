"""Run all 13 cumulative-feature cases and compare their saved predictions."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FEATURE_NAMES = (
    "current",
    "current_change",
    "current_dc_estimate",
    "mass_ratio",
    "inverse_mass_ratio",
    "elapsed_time",
    "startup_indicator",
    "previous_displacement",
    "previous_velocity",
    "previous_acceleration",
    "previous_force",
    "force_change",
    "quadratic_displacement_baseline",
)
ROLES = {
    "one_step_training": "training",
    "one_step_validation": "validation",
    "one_step_internal_test": "internal_test",
    "one_step_pure_test": "pure_test",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Garamond", "DejaVu Serif"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
    }
)


def _metrics(measured, predicted):
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = measured - predicted
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    maximum = float(np.max(np.abs(error)))
    centered = measured - measured.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0.0:
        r2 = np.nan
        fit = np.nan
    else:
        r2 = 1.0 - float(np.sum(error**2)) / denominator
        fit = 100.0 * (
            1.0 - float(np.linalg.norm(error)) / float(np.linalg.norm(centered))
        )
    return {"RMSE": rmse, "MAE": mae, "MaxAbsError": maximum, "R2": r2, "Fit_percent": fit}


def _prediction_frames(case_folder, role):
    prediction_folder = case_folder / "ResultsData" / "one_step_predictions"
    files = sorted(prediction_folder.glob(f"{role}__*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No {role} predictions found in {prediction_folder}. Run this case first."
        )
    return [(path.stem.split("__", 1)[1], pd.read_csv(path)) for path in files]


def _pool(case_folder, role):
    frames = _prediction_frames(case_folder, role)
    combined = pd.concat([frame for _, frame in frames], ignore_index=True)
    return frames, combined


def _case_complete(case_folder, feature_count):
    audit = case_folder / "ResultsData" / "feature_audit.json"
    if not audit.is_file():
        return False
    try:
        values = json.loads(audit.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return int(values.get("actual_lstm_input_size", -1)) == feature_count


def _run_case(feature_count, skip_completed):
    case_folder = HERE / f"F{feature_count:02d}"
    if skip_completed and _case_complete(case_folder, feature_count):
        print(f"Skipping completed F{feature_count:02d}", flush=True)
        return
    print("\n" + "#" * 96, flush=True)
    print(f"RUNNING F{feature_count:02d}: {feature_count} FEATURE(S)", flush=True)
    print("#" * 96, flush=True)
    environment = os.environ.copy()
    environment["DLSTM_SKIP_GITHUB_PUSH"] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    completed = subprocess.run(
        [sys.executable, str(case_folder / "main.py")],
        cwd=case_folder,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"F{feature_count:02d} failed with exit code {completed.returncode}.")
    if not _case_complete(case_folder, feature_count):
        raise RuntimeError(f"F{feature_count:02d} ended without a valid feature audit.")


def _collect_results():
    summary_rows = []
    pure_rows = []
    correction_rows = []
    for feature_count in range(1, 14):
        case_folder = HERE / f"F{feature_count:02d}"
        audit = json.loads(
            (case_folder / "ResultsData" / "feature_audit.json").read_text(encoding="utf-8")
        )
        role_results = {}
        for role, label in ROLES.items():
            frames, combined = _pool(case_folder, role)
            displacement = _metrics(
                combined["measured_displacement_mm"],
                combined["predicted_displacement_mm"],
            )
            role_results[label] = displacement
            summary_rows.append(
                {
                    "feature_count": feature_count,
                    "new_feature_added": FEATURE_NAMES[feature_count - 1],
                    "retained_features": " | ".join(FEATURE_NAMES[:feature_count]),
                    "role": label,
                    "output": "Displacement",
                    "trainable_parameters": int(audit["trainable_parameters"]),
                    **displacement,
                }
            )
            force = _metrics(combined["measured_force_N"], combined["predicted_force_N"])
            summary_rows.append(
                {
                    "feature_count": feature_count,
                    "new_feature_added": FEATURE_NAMES[feature_count - 1],
                    "retained_features": " | ".join(FEATURE_NAMES[:feature_count]),
                    "role": label,
                    "output": "Lorentz force (separate causal rule; not LSTM-ablated)",
                    "trainable_parameters": int(audit["trainable_parameters"]),
                    **force,
                }
            )
            if role == "one_step_pure_test":
                baseline_metrics = _metrics(
                    combined["measured_displacement_mm"],
                    combined["displacement_baseline_mm"],
                )
                correction_rms = float(
                    np.sqrt(np.mean(np.asarray(combined["lstm_correction_mm"]) ** 2))
                )
                correction_rows.append(
                    {
                        "feature_count": feature_count,
                        "pure_test_baseline_RMSE_mm": baseline_metrics["RMSE"],
                        "pure_test_hybrid_RMSE_mm": displacement["RMSE"],
                        "LSTM_correction_RMS_mm": correction_rms,
                        "hybrid_improvement_over_baseline_RMSE_mm": (
                            baseline_metrics["RMSE"] - displacement["RMSE"]
                        ),
                    }
                )
                for record_name, frame in frames:
                    record_metrics = _metrics(
                        frame["measured_displacement_mm"],
                        frame["predicted_displacement_mm"],
                    )
                    pure_rows.append(
                        {
                            "feature_count": feature_count,
                            "record": record_name,
                            "output": "Displacement",
                            **record_metrics,
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(pure_rows), pd.DataFrame(correction_rows)


def _difference_audit():
    reference_folder = HERE / "F13"
    reference = dict(_prediction_frames(reference_folder, "one_step_pure_test"))
    rows = []
    for feature_count in range(1, 14):
        frames = dict(_prediction_frames(HERE / f"F{feature_count:02d}", "one_step_pure_test"))
        differences = []
        force_differences = []
        for name, reference_frame in reference.items():
            frame = frames[name]
            if len(frame) != len(reference_frame):
                raise RuntimeError(f"Prediction length differs for {name} in F{feature_count:02d}.")
            differences.append(
                np.asarray(frame["predicted_displacement_mm"])
                - np.asarray(reference_frame["predicted_displacement_mm"])
            )
            force_differences.append(
                np.asarray(frame["predicted_force_N"])
                - np.asarray(reference_frame["predicted_force_N"])
            )
        displacement_difference = np.concatenate(differences)
        force_difference = np.concatenate(force_differences)
        rows.append(
            {
                "feature_count": feature_count,
                "comparison_reference": "F13",
                "prediction_difference_RMSE_mm": float(
                    np.sqrt(np.mean(displacement_difference**2))
                ),
                "prediction_difference_max_abs_mm": float(
                    np.max(np.abs(displacement_difference))
                ),
                "force_prediction_difference_max_abs_N": float(
                    np.max(np.abs(force_difference))
                ),
            }
        )
    return pd.DataFrame(rows)


def _make_figures(summary, pure, correction, differences, folder):
    folder.mkdir(parents=True, exist_ok=True)
    displacement = summary[summary["output"] == "Displacement"]
    validation = displacement[displacement["role"] == "validation"].sort_values("feature_count")
    pure_pooled = displacement[displacement["role"] == "pure_test"].sort_values("feature_count")
    training = displacement[displacement["role"] == "training"].sort_values("feature_count")

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(validation["feature_count"], validation["RMSE"], "o-", label="validation")
    axes[0].plot(pure_pooled["feature_count"], pure_pooled["RMSE"], "s--", label="pure test (confirmation)")
    axes[0].set_ylabel("Pooled displacement RMSE [mm]")
    axes[0].grid(True)
    axes[0].legend(frameon=False)
    axes[1].plot(validation["feature_count"], validation["Fit_percent"], "o-", label="validation")
    axes[1].plot(pure_pooled["feature_count"], pure_pooled["Fit_percent"], "s--", label="pure test")
    axes[1].set_ylabel("Pooled displacement fit [%]")
    axes[1].set_xlabel("Number of retained LSTM features")
    axes[1].set_xticks(range(1, 14))
    axes[1].grid(True)
    axes[1].legend(frameon=False)
    figure.suptitle("Incremental feature study: displacement accuracy")
    figure.tight_layout()
    figure.savefig(folder / "01_accuracy_vs_feature_count.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(training["feature_count"], training["RMSE"], "o-", label="training RMSE")
    axis.plot(validation["feature_count"], validation["RMSE"], "s-", label="validation RMSE")
    axis.set_xlabel("Number of retained LSTM features")
    axis.set_ylabel("Pooled displacement RMSE [mm]")
    axis.set_xticks(range(1, 14))
    axis.grid(True)
    axis.legend(frameon=False)
    axis.set_title("Training–validation behavior as model inputs increase")
    figure.tight_layout()
    figure.savefig(folder / "02_training_validation_complexity.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
        correction["feature_count"], correction["LSTM_correction_RMS_mm"], "o-",
        color="#6a1b9a",
    )
    axes[0].set_ylabel("LSTM correction RMS [mm]")
    axes[0].grid(True)
    axes[1].semilogy(
        differences["feature_count"],
        np.maximum(differences["prediction_difference_RMSE_mm"], 1e-15),
        "s-",
        color="#c62828",
    )
    axes[1].set_ylabel("Prediction difference from F13 [mm]")
    axes[1].set_xlabel("Number of retained LSTM features")
    axes[1].set_xticks(range(1, 14))
    axes[1].grid(True, which="both")
    figure.suptitle("Feature-effect audit: LSTM contribution and output difference")
    figure.tight_layout()
    figure.savefig(folder / "03_feature_effect_audit.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    worst = (
        pure.sort_values(["feature_count", "Fit_percent"])
        .groupby("feature_count", as_index=False)
        .first()
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(worst["feature_count"], worst["Fit_percent"], "o-", color="#ef6c00")
    axis.set_xlabel("Number of retained LSTM features")
    axis.set_ylabel("Worst pure-test displacement fit [%]")
    axis.set_xticks(range(1, 14))
    axis.grid(True)
    axis.set_title("Worst experiment prevents a strong pooled result from hiding failure")
    figure.tight_layout()
    figure.savefig(folder / "04_worst_pure_test_vs_features.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_report(summary, pure, correction, differences, ranking, output_folder):
    best = ranking.iloc[0]
    maximum_difference = float(
        differences[differences["feature_count"] < 13]["prediction_difference_max_abs_mm"].max()
    )
    pure_displacement = summary[
        (summary["role"] == "pure_test") & (summary["output"] == "Displacement")
    ]
    report = f"""# V15 cumulative 1-to-13 feature comparison

## What was tested

Thirteen independently trained models used the same random seed, data split, 64-sample history, two 48-unit LSTM layers, and 48-to-32-to-1 displacement-residual head. F01 uses current only. Each following case adds exactly one causal feature until F13 uses all 13.

The ablation affects only the LSTM displacement correction. Lorentz force is produced by the unchanged separate causal force rule, so force accuracy is not evidence for or against an LSTM feature.

## Validation-based comparison

The lowest pooled validation displacement RMSE occurred for F{int(best['feature_count']):02d} ({int(best['feature_count'])} features): {best['RMSE']:.8g} mm, with fit {best['Fit_percent']:.4f}%. This is the defensible ranking because validation data—not pure-test data—must guide model selection.

## Pure-test confirmation

Across the 13 cases, pooled pure-test displacement RMSE ranged from {pure_displacement['RMSE'].min():.8g} to {pure_displacement['RMSE'].max():.8g} mm. The maximum pointwise prediction difference between a reduced model and F13 was {maximum_difference:.8g} mm.

If the curves still look visually similar, inspect `03_feature_effect_audit.png` and `prediction_difference_vs_F13.csv`. The original quadratic displacement-history baseline is already strong, so current-only LSTM corrections can still produce high one-step tracking. Results were not artificially degraded to force a preferred conclusion.

## Interpretation limits

- More features do not automatically improve generalization; redundant or noisy features can leave accuracy unchanged or make validation performance worse.
- This is cumulative, order-dependent ablation. It answers what happens as features are added in the stated order; it is not a unique importance ranking for every feature.
- These are one-step measured-feedback results, not autonomous/free-running simulation results.
- Do not select a feature count from pure-test performance. Use the validation ranking, then describe pure-test results as confirmation.
"""
    (output_folder / "comparison_report.md").write_text(report, encoding="utf-8")


def _compare():
    output_folder = HERE / "ComparisonResults"
    figure_folder = HERE / "ComparisonFigures"
    if output_folder.exists():
        shutil.rmtree(output_folder)
    if figure_folder.exists():
        shutil.rmtree(figure_folder)
    output_folder.mkdir(parents=True)
    figure_folder.mkdir(parents=True)

    summary, pure, correction = _collect_results()
    differences = _difference_audit()
    validation_ranking = summary[
        (summary["role"] == "validation") & (summary["output"] == "Displacement")
    ].sort_values(["RMSE", "feature_count"]).copy()
    validation_ranking.insert(0, "validation_rank", range(1, len(validation_ranking) + 1))

    summary.to_csv(output_folder / "all_case_metrics.csv", index=False)
    pure.to_csv(output_folder / "per_pure_test_comparison.csv", index=False)
    correction.to_csv(output_folder / "baseline_and_lstm_correction.csv", index=False)
    differences.to_csv(output_folder / "prediction_difference_vs_F13.csv", index=False)
    validation_ranking.to_csv(output_folder / "validation_ranking.csv", index=False)
    pd.DataFrame(
        {
            "case": [f"F{count:02d}" for count in range(1, 14)],
            "feature_count": range(1, 14),
            "new_feature_added": FEATURE_NAMES,
            "retained_features": [" | ".join(FEATURE_NAMES[:count]) for count in range(1, 14)],
        }
    ).to_csv(output_folder / "feature_schedule.csv", index=False)

    _make_figures(summary, pure, correction, differences, figure_folder)
    _write_report(summary, pure, correction, differences, validation_ranking, output_folder)

    # Create the three publication-ready figures used by the separate
    # Overleaf report. They are computed directly from the CSV files above.
    from generate_report_figures import generate_report_figures

    report_figure_folder = generate_report_figures()

    print("\n" + "=" * 96)
    print("VALIDATION-BASED RANKING (lower displacement RMSE is better)")
    print(
        validation_ranking[
            ["validation_rank", "feature_count", "new_feature_added", "RMSE", "Fit_percent"]
        ].to_string(index=False)
    )
    print("=" * 96)
    print("Comparison results:", output_folder)
    print("Comparison figures:", figure_folder)
    print("Overleaf/report figures:", report_figure_folder)


def _push_once():
    if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") == "1":
        return
    try:
        from github_push import GitPushError, push_simulation

        run_time = datetime.now().astimezone()
        (HERE / "last_github_update.txt").write_text(
            f"Latest completed 13-case study: {run_time.isoformat(timespec='microseconds')}\n",
            encoding="utf-8",
        )
        push_simulation(
            HERE,
            commit_message=(
                "Update V15 1-to-13 feature comparison - "
                f"{run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            ),
        )
    except (GitPushError, ImportError) as error:
        print("\nIMPORTANT: all simulations completed, but GitHub push failed:")
        print(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse only cases whose feature audit proves the expected model was run.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Do not train; rebuild comparison from 13 already completed cases.",
    )
    arguments = parser.parse_args()
    if not arguments.compare_only:
        for feature_count in range(1, 14):
            _run_case(feature_count, arguments.skip_completed)
    _compare()
    _push_once()
    print("\nAll 13 cumulative feature cases and the comparison are complete.")


if __name__ == "__main__":
    main()
