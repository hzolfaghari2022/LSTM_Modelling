"""Run one fixed cumulative feature-ablation case.

Each case keeps the first N causal one-step features in the documented V15
feature order.  The externally applied physical input remains coil current;
the additional channels are causal derived/history/configuration features.
"""

from datetime import datetime
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from architecture_figure import generate_architecture_figure
from ablation_spec import ONE_STEP_FEATURE_NAMES as ABLATION_FEATURE_NAMES
from config import CPU_THREADS, SEED, TARGET_SAMPLE_RATE_HZ
from data_utils import prepare_data
from one_step_lstm import (
    ONE_STEP_FEATURE_NAMES,
    OneStepDisplacementLSTM,
    run_one_step_study,
)

if tuple(ONE_STEP_FEATURE_NAMES) != tuple(ABLATION_FEATURE_NAMES):
    raise RuntimeError(
        "The ablation specification does not match the implemented V15 features."
    )


def _parameter_count(feature_count):
    model = OneStepDisplacementLSTM(input_features=feature_count)
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _summary_row(metrics_path, role, output):
    table = pd.read_csv(metrics_path)
    row = table[(table["Data role"] == role) & (table["Output"] == output)]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one pooled row for {role}/{output}; found {len(row)}."
        )
    return row.iloc[0]


def _write_case_summary(case_folder, active_indices, removed_names):
    pooled_path = case_folder / "ResultsData" / "one_step_pooled_role_metrics.csv"
    validation = _summary_row(pooled_path, "Validation", "Displacement")
    pure = _summary_row(pooled_path, "Pure test", "Displacement")
    validation_force = _summary_row(pooled_path, "Validation", "Lorentz force")
    pure_force = _summary_row(pooled_path, "Pure test", "Lorentz force")

    active_names = [ONE_STEP_FEATURE_NAMES[index] for index in active_indices]
    summary = {
        "case": case_folder.name,
        "feature_count": len(active_indices),
        "trainable_parameters": _parameter_count(len(active_indices)),
        "active_feature_indices": json.dumps(active_indices),
        "active_features": "; ".join(active_names),
        "removed_features": "; ".join(removed_names) if removed_names else "None",
        "validation_displacement_RMSE": float(validation["RMSE"]),
        "validation_displacement_MAE": float(validation["MAE"]),
        "validation_displacement_R2": float(validation["R2"]),
        "validation_displacement_Fit_percent": float(validation["Fit_percent"]),
        "pure_displacement_RMSE": float(pure["RMSE"]),
        "pure_displacement_MAE": float(pure["MAE"]),
        "pure_displacement_R2": float(pure["R2"]),
        "pure_displacement_Fit_percent": float(pure["Fit_percent"]),
        "validation_force_RMSE": float(validation_force["RMSE"]),
        "validation_force_Fit_percent": float(validation_force["Fit_percent"]),
        "pure_force_RMSE": float(pure_force["RMSE"]),
        "pure_force_Fit_percent": float(pure_force["Fit_percent"]),
        "selection_note": (
            "Feature-count models must be ranked using validation displacement "
            "metrics; pure-test metrics are confirmation only."
        ),
        "force_note": (
            "Force is calculated by the unchanged separate causal force rule; "
            "the feature ablation changes only the displacement-residual LSTM."
        ),
    }
    path = case_folder / "ResultsData" / "ablation_case_summary.csv"
    pd.DataFrame([summary]).to_csv(path, index=False)
    return summary


def _push_completed_case(project_root, feature_count):
    if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") == "1":
        print("GitHub push disabled by DLSTM_SKIP_GITHUB_PUSH=1.")
        return
    try:
        from github_push import GitPushError, push_simulation

        run_time = datetime.now().astimezone()
        (project_root / "last_github_update.txt").write_text(
            (
                f"Latest completed ablation case: {feature_count} features; "
                f"{run_time.isoformat(timespec='microseconds')}\n"
            ),
            encoding="utf-8",
        )
        push_simulation(
            project_root,
            commit_message=(
                f"Update V15 {feature_count}-feature ablation results - "
                f"{run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            ),
        )
    except (GitPushError, ImportError) as error:
        print("\nIMPORTANT: simulation completed, but GitHub push failed:")
        print(error)
        print("Run the top-level push_now.py after correcting Git configuration.")


def run_case(feature_count, case_folder):
    """Train and evaluate exactly one fixed N-feature model."""
    if not 1 <= int(feature_count) <= len(ONE_STEP_FEATURE_NAMES):
        raise ValueError("feature_count must be from 1 through 13.")

    case_folder = Path(case_folder).resolve()
    project_root = case_folder.parent
    results_folder = case_folder / "ResultsData"
    figures_folder = case_folder / "FiguresResults"
    results_folder.mkdir(parents=True, exist_ok=True)
    figures_folder.mkdir(parents=True, exist_ok=True)
    for old_figure in figures_folder.glob("*.png"):
        old_figure.unlink()

    active_indices = list(range(int(feature_count)))
    active_names = [ONE_STEP_FEATURE_NAMES[index] for index in active_indices]
    removed_names = list(ONE_STEP_FEATURE_NAMES[int(feature_count):])

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training device: {device}, {torch.cuda.get_device_name(0)}")
    else:
        print("Training device: cpu")

    print("\n" + "=" * 88)
    print(f"ABLATION CASE: {feature_count} OF 13 FEATURES")
    print("Retained:")
    for number, name in enumerate(active_names, start=1):
        print(f"  {number:2d}. {name}")
    print("Removed:")
    if removed_names:
        for name in removed_names:
            print(f"  - {name}")
    else:
        print("  None (complete model)")
    print("=" * 88)

    print("\nLoading every measured record ...", flush=True)
    data = prepare_data(project_root)
    records = data["records"]
    data["inventory_table"].to_csv(results_folder / "record_inventory.csv", index=False)
    data["split_table"].to_csv(results_folder / "data_split.csv", index=False)

    print(f"Development workbook: {data['workbook'].name}")
    if data["independent_test_workbook"] is not None:
        print(
            "Optional independent-test workbook: "
            f"{data['independent_test_workbook'].name}"
        )
    print(f"Unique measured records: {len(records)}")
    print(f"Common sample rate: {TARGET_SAMPLE_RATE_HZ:.0f} Hz")
    print(f"Training windows: {len(data['training_pairs']):,}")
    print(f"Validation windows: {len(data['validation_pairs']):,}")
    print(f"Internal-test windows: {len(data['internal_test_pairs']):,}")
    print(
        "\nPrediction mode: ONE STEP measured feedback. The LSTM predicts "
        "the displacement residual; force uses the unchanged causal force rule."
    )

    metrics = run_one_step_study(
        data,
        records,
        device,
        results_folder,
        figures_folder,
        feature_indices=active_indices,
    )
    # The regular plot pipeline uses the complete-model label by default;
    # overwrite Figure 19 with the correct retained-feature count for this case.
    generate_architecture_figure(figures_folder, feature_count=len(active_indices))
    summary = _write_case_summary(case_folder, active_indices, removed_names)

    pure = metrics[metrics["Kind"] == "one_step_pure_test"]
    failed = pure[~pure["Pass95"]]
    print("\n" + "=" * 88)
    print(
        f"Case completed: {feature_count} features, "
        f"{summary['trainable_parameters']:,} trainable parameters"
    )
    print(
        "Validation displacement: "
        f"RMSE={summary['validation_displacement_RMSE']:.8g} mm, "
        f"fit={summary['validation_displacement_Fit_percent']:.5g}%"
    )
    print(
        "Pure-test displacement: "
        f"RMSE={summary['pure_displacement_RMSE']:.8g} mm, "
        f"fit={summary['pure_displacement_Fit_percent']:.5g}%"
    )
    if failed.empty:
        print("Every untouched pure-test channel passed its defined criterion.")
    else:
        print("Some untouched pure-test channels did not pass; inspect one_step_metrics.csv.")
    print("Results:", results_folder)
    print("Figures:", figures_folder)
    print("=" * 88)

    _push_completed_case(project_root, int(feature_count))
    print("\nDone.")
    return summary
