"""Run one independently reproducible cumulative feature case."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil

# Required for deterministic CUDA matrix multiplication. This must be set
# before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from ablation_spec import (
    parameter_count,
    retained_feature_indices,
    retained_feature_names,
)
from config import CPU_THREADS, SEED, TARGET_SAMPLE_RATE_HZ
from data_utils import prepare_data
from one_step_lstm import ONE_STEP_FEATURE_NAMES, run_one_step_study


def _clean_generated_folder(folder):
    """Remove only this case's generated output directory."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def _summarise_case(metrics, feature_count, case_folder):
    names = retained_feature_names(feature_count)
    rows = []
    for kind in (
        "one_step_training",
        "one_step_validation",
        "one_step_internal_test",
        "one_step_pure_test",
    ):
        for output in ("Displacement", "Lorentz force"):
            selected = metrics[
                (metrics["Kind"] == kind) & (metrics["Output"] == output)
            ]
            if selected.empty:
                continue
            fits = pd.to_numeric(selected["Fit_percent"], errors="coerce")
            rows.append(
                {
                    "feature_count": feature_count,
                    "newest_feature": names[-1],
                    "retained_features": " | ".join(names),
                    "data_role": kind,
                    "output": output,
                    "mean_RMSE_across_records": float(selected["RMSE"].mean()),
                    "worst_RMSE_across_records": float(selected["RMSE"].max()),
                    "mean_fit_percent_across_records": float(fits.mean()),
                    "worst_fit_percent_across_records": float(fits.min()),
                    "trainable_parameters": parameter_count(feature_count),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(case_folder) / "ResultsData" / "case_summary.csv", index=False)
    return summary


def run_case(feature_count, case_folder):
    """Train/evaluate one case using exactly the first feature_count features."""
    feature_count = int(feature_count)
    case_folder = Path(case_folder).resolve()
    feature_indices = retained_feature_indices(feature_count)
    feature_names = retained_feature_names(feature_count)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(CPU_THREADS)

    results_folder = case_folder / "ResultsData"
    figures_folder = case_folder / "FiguresResults"
    _clean_generated_folder(results_folder)
    _clean_generated_folder(figures_folder)

    print("\n" + "=" * 88)
    print(f"CUMULATIVE FEATURE CASE: {feature_count} OF {len(ONE_STEP_FEATURE_NAMES)}")
    print("New feature added in this case:", feature_names[-1])
    print("Retained features:")
    for number, name in enumerate(feature_names, start=1):
        print(f"  {number:2d}. {name}")
    print("Expected trainable parameters:", f"{parameter_count(feature_count):,}")
    print("=" * 88)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training device: {device}, {torch.cuda.get_device_name(0)}")
    else:
        print("Training device: cpu")

    print("\nLoading every measured record ...", flush=True)
    data = prepare_data(case_folder)
    records = data["records"]
    data["inventory_table"].to_csv(results_folder / "record_inventory.csv", index=False)
    data["split_table"].to_csv(results_folder / "data_split.csv", index=False)

    print(f"Development workbook: {data['workbook'].name}")
    if data["independent_test_workbook"] is not None:
        print("Optional independent test:", data["independent_test_workbook"].name)
    print(f"Unique records: {len(records)}; sample rate: {TARGET_SAMPLE_RATE_HZ:.0f} Hz")

    # The complete LSTM correction is used in this controlled study. The
    # original validation-selected safety blend remains unchanged in the
    # ordinary V15 main.py; fixing alpha=1 here prevents alpha=0 from making
    # all feature cases collapse to the identical physical baseline.
    metrics = run_one_step_study(
        data,
        records,
        device,
        results_folder,
        figures_folder,
        feature_indices=feature_indices,
        residual_trust_override=1.0,
    )
    _summarise_case(metrics, feature_count, case_folder)

    metadata_path = results_folder / "one_step_plot_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_count = int(metadata["input_feature_count"])
    actual_parameters = int(metadata["trainable_parameters"])
    if actual_count != feature_count:
        raise RuntimeError(
            f"Feature audit failed: requested {feature_count}, model used {actual_count}."
        )
    if actual_parameters != parameter_count(feature_count):
        raise RuntimeError(
            "Parameter audit failed: expected "
            f"{parameter_count(feature_count)}, model used {actual_parameters}."
        )
    (results_folder / "feature_audit.json").write_text(
        json.dumps(
            {
                "requested_feature_count": feature_count,
                "actual_lstm_input_size": actual_count,
                "retained_features": list(feature_names),
                "trainable_parameters": actual_parameters,
                "history_samples": 64,
                "lstm_target": "displacement residual only",
                "force_note": "Force is computed by a separate causal rule.",
                "residual_trust_used": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") != "1":
        try:
            from github_push import GitPushError, push_simulation

            run_time = datetime.now().astimezone()
            (case_folder / "last_github_update.txt").write_text(
                f"Latest completed simulation: {run_time.isoformat(timespec='microseconds')}\n",
                encoding="utf-8",
            )
            push_simulation(
                case_folder,
                commit_message=(
                    f"Update V15 cumulative {feature_count:02d}-feature results - "
                    f"{run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                ),
            )
        except (GitPushError, ImportError) as error:
            print("\nIMPORTANT: simulation completed, but GitHub push failed:")
            print(error)

    print("\nCase completed:", case_folder.name)
    print("Results:", results_folder)
    print("Figures:", figures_folder)
    return metrics

