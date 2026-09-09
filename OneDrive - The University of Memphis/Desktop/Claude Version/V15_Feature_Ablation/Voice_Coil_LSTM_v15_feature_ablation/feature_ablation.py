"""Validation-selected backward feature ablation for the primary V15 model.

The study starts with all thirteen causal features and removes one feature at
a time.  At each step, every possible one-feature removal is trained from
scratch and ranked using development validation RMSE only.  Pure-test targets
are evaluated only after the choice has been made and never select features.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from architecture_figure import generate_architecture_figure
from config import BATCH_SIZE, CPU_THREADS, NEGLIGIBLE_SIGNAL_RANGE, SEED
from data_utils import prepare_data
from one_step_lstm import (
    FIT_TARGET_PERCENT,
    HIDDEN_SIZE,
    LAYERS,
    ONE_STEP_FEATURE_NAMES,
    OneStepDisplacementLSTM,
    _OneStepDataset,
    _fit_startup_force_gain,
    _force_prediction,
    _make_causal_arrays,
    _predict_displacement,
    _role_pairs,
    _training_statistics,
)
from plot_style import apply_publication_style
from train_utils import calculate_metrics


FULL_EPOCHS = int(os.environ.get("DLSTM_ONE_STEP_EPOCHS", "20"))
FULL_SAMPLES_PER_EPOCH = int(
    os.environ.get("DLSTM_ONE_STEP_SAMPLES_PER_EPOCH", "20000")
)
PATIENCE = 5
FIT_DROP_LIMIT_POINTS = float(
    os.environ.get("DLSTM_ABLATION_MAX_FIT_DROP", "0.5")
)
RMSE_INCREASE_LIMIT_PERCENT = float(
    os.environ.get("DLSTM_ABLATION_MAX_RMSE_INCREASE", "5.0")
)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sampler(training_pairs, record_count, sample_count, seed):
    counts = np.bincount(training_pairs[:, 0], minlength=record_count)
    weights = np.asarray(
        [1.0 / max(counts[record_index], 1) for record_index, _ in training_pairs],
        dtype=np.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=int(sample_count),
        replacement=True,
        generator=generator,
    )


def _limited_pairs(pairs, maximum):
    pairs = np.asarray(pairs, dtype=np.int64)
    if maximum is None or len(pairs) <= maximum:
        return pairs
    positions = np.linspace(0, len(pairs) - 1, maximum, dtype=np.int64)
    return pairs[positions]


def _predict_grouped(
    model,
    records,
    pairs,
    statistics,
    device,
    trust,
    feature_indices,
    startup_gain,
):
    measured_parts = []
    predicted_parts = []
    item_results = []
    pairs = np.asarray(pairs, dtype=np.int64)
    for record_index in sorted(set(pairs[:, 0].tolist())):
        subset = pairs[pairs[:, 0] == record_index]
        record = records[record_index]
        targets = subset[:, 1]
        displacement, _ = _predict_displacement(
            model,
            record,
            targets,
            statistics,
            device,
            residual_trust=trust,
            feature_indices=feature_indices,
        )
        force = _force_prediction(record, targets, startup_gain)
        measured = record["outputs"][targets]
        predicted = np.column_stack([displacement, force]).astype(np.float32)
        measured_parts.append(measured)
        predicted_parts.append(predicted)
        item_results.append(
            {
                "name": record["name"],
                "time": record["time"][targets],
                "current": record["current"][targets],
                "measured": measured,
                "predicted": predicted,
            }
        )
    return (
        np.concatenate(measured_parts, axis=0),
        np.concatenate(predicted_parts, axis=0),
        item_results,
    )


def _fit_validation_trust(
    model, records, validation_pairs, statistics, device, feature_indices
):
    actual = []
    correction = []
    for record_index in sorted(set(validation_pairs[:, 0].tolist())):
        subset = validation_pairs[validation_pairs[:, 0] == record_index]
        record = records[record_index]
        targets = subset[:, 1]
        full, baseline = _predict_displacement(
            model,
            record,
            targets,
            statistics,
            device,
            residual_trust=1.0,
            feature_indices=feature_indices,
        )
        actual.append(record["outputs"][targets, 0] - baseline)
        correction.append(full - baseline)
    actual = np.concatenate(actual).astype(np.float64)
    correction = np.concatenate(correction).astype(np.float64)
    denominator = float(np.dot(correction, correction))
    if denominator <= 1e-20:
        return 0.0
    return float(np.clip(np.dot(actual, correction) / denominator, 0.0, 1.0))


def _train_subset(
    records,
    training_pairs,
    validation_pairs,
    active_indices,
    device,
    seed,
    epochs,
    samples_per_epoch,
):
    _set_seed(seed)
    statistics = _training_statistics(records, training_pairs, active_indices)
    feature_mean, feature_std, target_mean, target_std = statistics
    training_dataset = _OneStepDataset(
        records,
        training_pairs,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        feature_indices=active_indices,
    )
    validation_dataset = _OneStepDataset(
        records,
        validation_pairs,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        feature_indices=active_indices,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        sampler=_sampler(training_pairs, len(records), samples_per_epoch, seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = OneStepDisplacementLSTM(input_features=len(active_indices)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_function = nn.MSELoss()
    best_loss = np.inf
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        training_losses = []
        for inputs, targets in training_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_losses.append(float(loss.item()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device, non_blocking=device.type == "cuda")
                targets = targets.to(device, non_blocking=device.type == "cuda")
                validation_losses.append(
                    float(loss_function(model(inputs), targets).item())
                )
        validation_loss = float(np.mean(validation_losses))
        training_loss = float(np.mean(training_losses))
        history.append((epoch, training_loss, validation_loss))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    trust = _fit_validation_trust(
        model,
        records,
        validation_pairs,
        statistics,
        device,
        active_indices,
    )
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {
        "model": model,
        "state": state,
        "statistics": statistics,
        "trust": trust,
        "best_loss": float(best_loss),
        "history": history,
    }


def _metric_row(feature_count, role, output, values, record="POOLED"):
    return {
        "feature_count": feature_count,
        "record": record,
        "role": role,
        "output": output,
        **values,
    }


def _evaluate_bundle(
    bundle,
    records,
    validation_pairs,
    pure_pairs_by_name,
    active_indices,
    device,
    startup_gain,
    maximum_evaluation_samples=None,
):
    model = bundle["model"]
    statistics = bundle["statistics"]
    trust = bundle["trust"]
    feature_count = len(active_indices)
    rows = []

    validation_subset = _limited_pairs(validation_pairs, maximum_evaluation_samples)
    validation_measured, validation_predicted, _ = _predict_grouped(
        model,
        records,
        validation_subset,
        statistics,
        device,
        trust,
        active_indices,
        startup_gain,
    )
    for column, output in ((0, "Displacement"), (1, "Lorentz force")):
        values = calculate_metrics(
            validation_measured[:, column],
            validation_predicted[:, column],
            negligible_range=NEGLIGIBLE_SIGNAL_RANGE[output],
        )
        rows.append(_metric_row(feature_count, "validation", output, values))

    pure_measured_parts = []
    pure_predicted_parts = []
    pure_predictions = {}
    for name, pairs in pure_pairs_by_name.items():
        subset = _limited_pairs(pairs, maximum_evaluation_samples)
        measured, predicted, items = _predict_grouped(
            model,
            records,
            subset,
            statistics,
            device,
            trust,
            active_indices,
            startup_gain,
        )
        item = items[0]
        pure_predictions[name] = item
        pure_measured_parts.append(measured)
        pure_predicted_parts.append(predicted)
        for column, output in ((0, "Displacement"), (1, "Lorentz force")):
            values = calculate_metrics(
                measured[:, column],
                predicted[:, column],
                negligible_range=NEGLIGIBLE_SIGNAL_RANGE[output],
            )
            rows.append(
                _metric_row(feature_count, "pure_test", output, values, record=name)
            )

    if pure_measured_parts:
        pure_measured = np.concatenate(pure_measured_parts, axis=0)
        pure_predicted = np.concatenate(pure_predicted_parts, axis=0)
        for column, output in ((0, "Displacement"), (1, "Lorentz force")):
            values = calculate_metrics(
                pure_measured[:, column],
                pure_predicted[:, column],
                negligible_range=NEGLIGIBLE_SIGNAL_RANGE[output],
            )
            rows.append(_metric_row(feature_count, "pure_test", output, values))

    return pd.DataFrame(rows), pure_predictions


def _validation_displacement(metrics):
    row = metrics[
        (metrics["record"] == "POOLED")
        & (metrics["role"] == "validation")
        & (metrics["output"] == "Displacement")
    ].iloc[0]
    return float(row["RMSE"]), float(row["Fit_percent"])


def _save_selected_case(
    output_folder,
    active_indices,
    removed_feature,
    bundle,
    metrics,
    predictions,
    mode,
):
    feature_count = len(active_indices)
    folder = Path(output_folder) / f"F{feature_count:02d}"
    prediction_folder = folder / "pure_test_predictions"
    prediction_folder.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(
        bundle["history"],
        columns=["epoch", "training_loss", "validation_loss"],
    ).to_csv(folder / "training_history.csv", index=False)
    metadata = {
        "mode": mode,
        "reportable": mode == "full",
        "feature_count": feature_count,
        "active_feature_indices": list(active_indices),
        "active_features": [ONE_STEP_FEATURE_NAMES[index] for index in active_indices],
        "removed_at_this_step": removed_feature,
        "validation_selected_residual_trust": bundle["trust"],
        "best_validation_loss": bundle["best_loss"],
        "lstm_hidden_size": HIDDEN_SIZE,
        "lstm_layers": LAYERS,
        "lstm_output": "displacement residual only",
        "force_prediction": "separate causal force rule; unaffected by feature ablation",
    }
    with open(folder / "case_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    torch.save(
        {
            "model_state_dict": bundle["state"],
            "feature_indices": list(active_indices),
            "feature_names": metadata["active_features"],
            "feature_mean": bundle["statistics"][0],
            "feature_std": bundle["statistics"][1],
            "target_mean": bundle["statistics"][2],
            "target_std": bundle["statistics"][3],
            "displacement_residual_trust": bundle["trust"],
        },
        folder / "model.pt",
    )
    for name, item in predictions.items():
        pd.DataFrame(
            {
                "time_s": item["time"],
                "current_A": item["current"],
                "measured_displacement_mm": item["measured"][:, 0],
                "predicted_displacement_mm": item["predicted"][:, 0],
                "displacement_error_mm": item["measured"][:, 0] - item["predicted"][:, 0],
                "measured_force_N": item["measured"][:, 1],
                "predicted_force_N": item["predicted"][:, 1],
                "force_error_N": item["measured"][:, 1] - item["predicted"][:, 1],
            }
        ).to_csv(prediction_folder / f"{_safe_name(name)}.csv", index=False)


def _parameter_count(feature_count):
    model = OneStepDisplacementLSTM(input_features=feature_count)
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _recommend_feature_count(path_table):
    ordered = path_table.sort_values("feature_count", ascending=False)
    full = ordered.iloc[0]
    fit_minimum = full["validation_fit_percent"] - FIT_DROP_LIMIT_POINTS
    rmse_maximum = full["validation_rmse"] * (
        1.0 + RMSE_INCREASE_LIMIT_PERCENT / 100.0
    )
    acceptable = ordered[
        (ordered["validation_fit_percent"] >= fit_minimum)
        & (ordered["validation_rmse"] <= rmse_maximum)
    ]
    if acceptable.empty:
        return int(full["feature_count"]), fit_minimum, rmse_maximum
    return int(acceptable["feature_count"].min()), fit_minimum, rmse_maximum


def _plot_accuracy(path_table, figures_folder, reportable):
    apply_publication_style()
    table = path_table.sort_values("feature_count")
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    axes[0, 0].plot(table["feature_count"], table["validation_fit_percent"], "o-", label="Validation")
    axes[0, 0].plot(table["feature_count"], table["pure_fit_percent"], "s--", label="Pure test")
    axes[0, 0].set_ylabel("Displacement fit [%]")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(table["feature_count"], table["validation_rmse"], "o-", label="Validation")
    axes[0, 1].plot(table["feature_count"], table["pure_rmse"], "s--", label="Pure test")
    axes[0, 1].set_ylabel("Displacement RMSE [mm]")
    axes[1, 0].plot(table["feature_count"], table["validation_r2"], "o-", color="#2e7d32")
    axes[1, 0].set_ylabel(r"Validation $R^2$")
    axes[1, 1].plot(table["feature_count"], table["trainable_parameters"], "o-", color="#6a1b9a")
    axes[1, 1].set_ylabel("Trainable parameters")
    for axis in axes.ravel():
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("Number of retained features")
        axis.set_xticks(table["feature_count"])
    label = "FULL REPORTABLE STUDY" if reportable else "QUICK SMOKE TEST — NOT REPORTABLE"
    figure.suptitle(f"Feature-ablation accuracy ({label})", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(Path(figures_folder) / "01_feature_ablation_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_feature_matrix(path_table, figures_folder):
    apply_publication_style()
    ordered = path_table.sort_values("feature_count", ascending=False)
    matrix = np.zeros((len(ordered), len(ONE_STEP_FEATURE_NAMES)))
    for row_number, row in enumerate(ordered.itertuples(index=False)):
        active = set(json.loads(row.active_feature_indices))
        for index in active:
            matrix[row_number, index] = 1.0
    figure, axis = plt.subplots(figsize=(13.5, 6.8))
    axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(ONE_STEP_FEATURE_NAMES)))
    axis.set_xticklabels(ONE_STEP_FEATURE_NAMES, rotation=42, ha="right")
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([f"{count} features" for count in ordered["feature_count"]])
    axis.set_title("Validation-selected backward-elimination path", fontsize=14, fontweight="bold")
    axis.set_xlabel("Causal model feature (blue = retained; white = removed)")
    figure.tight_layout()
    figure.savefig(Path(figures_folder) / "02_selected_features_by_count.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_response_comparison(all_predictions, path_table, recommended_count, figures_folder):
    full_count = int(path_table["feature_count"].max())
    minimum_count = int(path_table["feature_count"].min())
    full_metrics = all_predictions[full_count]["metrics"]
    candidates = full_metrics[
        (full_metrics["role"] == "pure_test")
        & (full_metrics["output"] == "Displacement")
        & (full_metrics["record"] != "POOLED")
    ]
    worst_name = candidates.sort_values("Fit_percent", na_position="first").iloc[0]["record"]
    counts = []
    for value in (full_count, recommended_count, minimum_count):
        if value not in counts:
            counts.append(value)
    colors = ["#d81b60", "#0072b2", "#e67e22"]
    item = all_predictions[full_count]["predictions"][worst_name]
    time = item["time"]
    measured = item["measured"]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 7.8), sharex="col")
    axes[0, 0].plot(time, measured[:, 0], color="black", lw=2.0, label="Measured")
    for color, count in zip(colors, counts):
        prediction = all_predictions[count]["predictions"][worst_name]["predicted"][:, 0]
        axes[0, 0].plot(time, prediction, color=color, lw=1.35, ls="--", label=f"{count} features")
        axes[1, 0].plot(time, measured[:, 0] - prediction, color=color, lw=1.0, label=f"{count} features")
    axes[0, 0].set_ylabel("Displacement [mm]")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_ylabel("Displacement error [mm]")
    axes[1, 0].set_xlabel("Time [s]")
    force_prediction = item["predicted"][:, 1]
    axes[0, 1].plot(time, measured[:, 1], color="black", lw=2.0, label="Measured")
    axes[0, 1].plot(time, force_prediction, color="#0072b2", lw=1.4, ls="--", label="Causal force rule")
    axes[0, 1].set_ylabel("Lorentz force [N]")
    axes[0, 1].legend(frameon=False)
    axes[1, 1].plot(time, measured[:, 1] - force_prediction, color="#c62828", lw=1.0)
    axes[1, 1].set_ylabel("Force error [N]")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].text(
        0.5, 0.08, "Force is computed by a separate causal rule;\nfeature ablation changes only the displacement LSTM.",
        transform=axes[1, 1].transAxes, ha="center", va="bottom", fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#b0bec5"},
    )
    for axis in axes.ravel():
        axis.grid(True, alpha=0.3)
    figure.suptitle(
        f"Response effect of feature ablation — representative worst pure test: {worst_name}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(Path(figures_folder) / "03_response_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_pure_test_heatmap(metrics_table, figures_folder):
    rows = metrics_table[
        (metrics_table["role"] == "pure_test")
        & (metrics_table["output"] == "Displacement")
        & (metrics_table["record"] != "POOLED")
    ].copy()
    pivot = rows.pivot_table(
        index="feature_count", columns="record", values="Fit_percent", aggfunc="first"
    ).sort_index(ascending=False)
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(
        figsize=(max(10.5, 2.0 * len(pivot.columns)), 0.48 * len(pivot.index) + 2.6)
    )
    image = axis.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=90, vmax=100)
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels([f"{count} features" for count in pivot.index])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_value = "n/a" if not np.isfinite(value) else f"{value:.2f}"
            axis.text(column, row, text_value, ha="center", va="center", fontsize=8.5)
    axis.set_title(
        "Displacement fit for every untouched pure-test experiment",
        fontsize=14,
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Fit [%]")
    figure.tight_layout()
    figure.savefig(Path(figures_folder) / "04_pure_test_fit_by_feature_count.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _summary_text(path_table, recommended_count, mode):
    full = path_table.loc[path_table["feature_count"].idxmax()]
    reduced = path_table[path_table["feature_count"] == recommended_count].iloc[0]
    return (
        f"Feature-ablation mode: {mode}\n"
        f"Reportable: {'yes' if mode == 'full' else 'no; smoke test only'}\n"
        "Selection rule: greedy backward elimination using validation displacement RMSE only.\n"
        "Pure-test results were reported after each validation decision and were not used to remove features.\n"
        f"Full model: 13 features, validation RMSE={full['validation_rmse']:.8g} mm, "
        f"fit={full['validation_fit_percent']:.5g}%.\n"
        f"Smallest validation-acceptable model: {recommended_count} features, "
        f"validation RMSE={reduced['validation_rmse']:.8g} mm, "
        f"fit={reduced['validation_fit_percent']:.5g}%.\n"
        f"Acceptance limits: fit decrease <= {FIT_DROP_LIMIT_POINTS:g} percentage points and "
        f"RMSE increase <= {RMSE_INCREASE_LIMIT_PERCENT:g}% relative to the 13-feature model.\n"
        "Important: the one-step LSTM predicts displacement residual only. Lorentz force is calculated by a separate causal rule.\n"
    )


def _push_project(project_folder, mode):
    if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") == "1":
        print("GitHub push disabled by DLSTM_SKIP_GITHUB_PUSH=1.")
        return
    try:
        from github_push import GitPushError, push_simulation

        now = datetime.now().astimezone()
        (Path(project_folder) / "last_github_update.txt").write_text(
            f"Latest completed feature-ablation study: {now.isoformat(timespec='microseconds')}\n",
            encoding="utf-8",
        )
        push_simulation(
            project_folder,
            commit_message=f"Update V15 {mode} feature-ablation results - {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        )
    except (GitPushError, ImportError) as error:
        print("\nIMPORTANT: ablation completed, but GitHub push failed:")
        print(error)
        print("Run 'python push_now.py' after correcting the Git configuration.")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick is a non-reportable smoke test; full uses the main training settings",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help="stop at this retained-feature count (default: 1)",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="quick-mode diagnostic only; limits removal candidates per step",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if not 1 <= arguments.min_features <= len(ONE_STEP_FEATURE_NAMES):
        raise ValueError("--min-features must be from 1 through 13.")
    if arguments.mode == "full" and arguments.candidate_limit is not None:
        raise ValueError("--candidate-limit is allowed only in quick smoke-test mode.")

    project_folder = Path(__file__).resolve().parent
    if arguments.mode == "full":
        results_folder = project_folder / "FeatureAblationResults"
        figures_folder = project_folder / "FeatureAblationFigures"
        epochs = FULL_EPOCHS
        samples_per_epoch = FULL_SAMPLES_PER_EPOCH
        maximum_evaluation_samples = None
        candidate_limit = None
    else:
        results_folder = project_folder / "FeatureAblationSmokeResults"
        figures_folder = project_folder / "FeatureAblationSmokeFigures"
        epochs = 1
        samples_per_epoch = 512
        maximum_evaluation_samples = 600
        candidate_limit = arguments.candidate_limit or 3
    results_folder.mkdir(parents=True, exist_ok=True)
    figures_folder.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Ablation device: {device}, {torch.cuda.get_device_name(0)}")
    else:
        print("Ablation device: cpu")
    print(f"Mode: {arguments.mode} ({'REPORTABLE' if arguments.mode == 'full' else 'NOT REPORTABLE'})")
    print("Loading Total_Data.xlsx and optional Test_idpd.xlsx ...", flush=True)
    data = prepare_data(project_folder)
    records = data["records"]
    for record in records:
        record["one_step"] = _make_causal_arrays(record)
    training_pairs = _role_pairs(records, "training")
    validation_pairs = _role_pairs(records, "validation")
    if arguments.mode == "quick":
        training_pairs = _limited_pairs(training_pairs, 1200)
        validation_pairs = _limited_pairs(validation_pairs, 600)
    pure_pairs_by_name = {}
    for name in data["pure_test_names"]:
        record_index = data["index_of"][name]
        record = records[record_index]
        targets = np.arange(record["pad"], record["samples"], dtype=np.int64)
        pure_pairs_by_name[name] = np.column_stack(
            [np.full(len(targets), record_index, dtype=np.int64), targets]
        )
    startup_gain = _fit_startup_force_gain(records)

    active_indices = list(range(len(ONE_STEP_FEATURE_NAMES)))
    selected_rows = []
    candidate_rows = []
    all_metric_tables = []
    all_predictions = {}
    removed_feature = "none (complete model)"

    while len(active_indices) >= arguments.min_features:
        feature_count = len(active_indices)
        print(f"\nSelecting and evaluating {feature_count} retained features ...", flush=True)
        if feature_count == len(ONE_STEP_FEATURE_NAMES):
            bundle = _train_subset(
                records,
                training_pairs,
                validation_pairs,
                active_indices,
                device,
                arguments.seed,
                epochs,
                samples_per_epoch,
            )
        else:
            raise RuntimeError("Internal ablation state error.")

        while True:
            metrics, predictions = _evaluate_bundle(
                bundle,
                records,
                validation_pairs,
                pure_pairs_by_name,
                active_indices,
                device,
                startup_gain,
                maximum_evaluation_samples,
            )
            validation_rmse, validation_fit = _validation_displacement(metrics)
            validation_row = metrics[
                (metrics["record"] == "POOLED")
                & (metrics["role"] == "validation")
                & (metrics["output"] == "Displacement")
            ].iloc[0]
            pure_row = metrics[
                (metrics["record"] == "POOLED")
                & (metrics["role"] == "pure_test")
                & (metrics["output"] == "Displacement")
            ].iloc[0]
            selected_rows.append(
                {
                    "feature_count": len(active_indices),
                    "active_feature_indices": json.dumps(active_indices),
                    "active_features": "; ".join(ONE_STEP_FEATURE_NAMES[index] for index in active_indices),
                    "removed_at_this_step": removed_feature,
                    "trainable_parameters": _parameter_count(len(active_indices)),
                    "validation_rmse": validation_rmse,
                    "validation_mae": float(validation_row["MAE"]),
                    "validation_r2": float(validation_row["R2"]),
                    "validation_fit_percent": validation_fit,
                    "pure_rmse": float(pure_row["RMSE"]),
                    "pure_mae": float(pure_row["MAE"]),
                    "pure_r2": float(pure_row["R2"]),
                    "pure_fit_percent": float(pure_row["Fit_percent"]),
                }
            )
            all_metric_tables.append(metrics)
            all_predictions[len(active_indices)] = {
                "metrics": metrics,
                "predictions": predictions,
            }
            _save_selected_case(
                results_folder,
                active_indices,
                removed_feature,
                bundle,
                metrics,
                predictions,
                arguments.mode,
            )

            if len(active_indices) == arguments.min_features:
                break

            removal_candidates = list(active_indices)
            if candidate_limit is not None:
                removal_candidates = removal_candidates[:candidate_limit]
            best_candidate = None
            for position, removed_index in enumerate(removal_candidates, start=1):
                candidate_indices = [index for index in active_indices if index != removed_index]
                removed_name = ONE_STEP_FEATURE_NAMES[removed_index]
                print(
                    f"  candidate {position}/{len(removal_candidates)}: remove {removed_name}",
                    flush=True,
                )
                candidate_bundle = _train_subset(
                    records,
                    training_pairs,
                    validation_pairs,
                    candidate_indices,
                    device,
                    arguments.seed,
                    epochs,
                    samples_per_epoch,
                )
                candidate_metrics, _ = _evaluate_bundle(
                    candidate_bundle,
                    records,
                    validation_pairs,
                    pure_pairs_by_name={},
                    active_indices=candidate_indices,
                    device=device,
                    startup_gain=startup_gain,
                    maximum_evaluation_samples=maximum_evaluation_samples,
                )
                candidate_rmse, candidate_fit = _validation_displacement(candidate_metrics)
                candidate_rows.append(
                    {
                        "from_feature_count": len(active_indices),
                        "to_feature_count": len(candidate_indices),
                        "candidate_removed": removed_name,
                        "candidate_active_features": "; ".join(
                            ONE_STEP_FEATURE_NAMES[index] for index in candidate_indices
                        ),
                        "validation_rmse": candidate_rmse,
                        "validation_fit_percent": candidate_fit,
                    }
                )
                candidate = (
                    candidate_rmse,
                    -candidate_fit,
                    removed_index,
                    candidate_indices,
                    candidate_bundle,
                )
                if best_candidate is None or candidate[:2] < best_candidate[:2]:
                    if best_candidate is not None:
                        del best_candidate[4]["model"]
                    best_candidate = candidate
                else:
                    del candidate_bundle["model"]
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            _, _, removed_index, active_indices, bundle = best_candidate
            removed_feature = ONE_STEP_FEATURE_NAMES[removed_index]
            print(f"  selected removal: {removed_feature}", flush=True)

        break

    path_table = pd.DataFrame(selected_rows).sort_values("feature_count", ascending=False)
    candidate_table = pd.DataFrame(candidate_rows)
    metrics_table = pd.concat(all_metric_tables, ignore_index=True)
    recommended_count, fit_minimum, rmse_maximum = _recommend_feature_count(path_table)
    path_table["validation_acceptable"] = (
        (path_table["validation_fit_percent"] >= fit_minimum)
        & (path_table["validation_rmse"] <= rmse_maximum)
    )
    path_table.to_csv(results_folder / "ablation_summary.csv", index=False)
    candidate_table.to_csv(results_folder / "candidate_removal_trials.csv", index=False)
    metrics_table.to_csv(results_folder / "all_selected_case_metrics.csv", index=False)
    pure_table = metrics_table[
        (metrics_table["role"] == "pure_test")
        & (metrics_table["record"] != "POOLED")
    ]
    pure_table.to_csv(results_folder / "per_pure_test_metrics.csv", index=False)
    summary = _summary_text(path_table, recommended_count, arguments.mode)
    (results_folder / "README_RESULTS.txt").write_text(summary, encoding="utf-8")

    _plot_accuracy(path_table, figures_folder, arguments.mode == "full")
    _plot_feature_matrix(path_table, figures_folder)
    _plot_response_comparison(all_predictions, path_table, recommended_count, figures_folder)
    _plot_pure_test_heatmap(metrics_table, figures_folder)
    generate_architecture_figure(figures_folder)

    print("\n" + summary)
    print("Ablation results:", results_folder)
    print("Ablation figures:", figures_folder)
    _push_project(project_folder, arguments.mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nFeature ablation interrupted by the user.")
        sys.exit(130)
