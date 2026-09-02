"""Causal one-step measured-feedback LSTM and its pure-test report.

This module is intentionally separate from the autonomous model in main.py.
At sample k it may use measured displacement and force through k-1, plus the
known current at k.  It never uses the measured output at k.  That makes it a
valid one-step predictor, but not a free-running simulator.

The LSTM learns only the displacement correction to a local constant-
acceleration predictor.  Lorentz force uses a causal actuator identity whose
gain is updated from the previous measured force/current pair.  This guarded
hybrid is much more reliable on an unseen load than asking an LSTM to
extrapolate an absolute force level.
"""

from copy import deepcopy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import (
    BATCH_SIZE,
    NEGLIGIBLE_SIGNAL_RANGE,
    SEED,
    TARGET_TIME_STEP,
    ZERO_CURRENT_TOLERANCE_A,
)
from data_utils import causal_dc_estimate
from plot_results import generate_complete_figures
from train_utils import calculate_metrics
from ablation_settings import (
    SELECTED_FEATURE_COUNT,
    SELECTED_FEATURE_INDICES,
)


HISTORY_SAMPLES = 64
HIDDEN_SIZE = 48
LAYERS = 2
EPOCHS = int(os.environ.get("DLSTM_ONE_STEP_EPOCHS", "20"))
PATIENCE = 5
SAMPLES_PER_EPOCH = int(os.environ.get("DLSTM_ONE_STEP_SAMPLES_PER_EPOCH", "20000"))
MIN_PREVIOUS_CURRENT_A = 0.002
FIT_TARGET_PERCENT = 95.0

ALL_ONE_STEP_FEATURE_NAMES = (
    "current",
    "current_change",
    "current_dc_estimate",
    "mass_ratio",
    "inverse_mass_ratio",
    "elapsed_time",
    "startup_indicator",
    "previous_displacement",
    "estimated_velocity",
    "estimated_acceleration",
    "previous_force",
    "previous_force_change",
    "constant_acceleration_displacement",
)

SELECTED_FEATURE_NAMES = tuple(
    ALL_ONE_STEP_FEATURE_NAMES[index] for index in SELECTED_FEATURE_INDICES
)
REMOVED_FEATURE_NAMES = tuple(
    name
    for index, name in enumerate(ALL_ONE_STEP_FEATURE_NAMES)
    if index not in SELECTED_FEATURE_INDICES
)


class OneStepDisplacementLSTM(nn.Module):
    """Predict the next displacement residual from causal measured history."""

    def __init__(self, input_features=SELECTED_FEATURE_COUNT):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_features,
            HIDDEN_SIZE,
            num_layers=LAYERS,
            batch_first=True,
            dropout=0.10,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, sequence):
        hidden, _ = self.recurrent(sequence)
        return self.head(hidden[:, -1]).squeeze(-1)


def _shift(values, amount):
    return np.concatenate(
        [np.repeat(values[:1], amount, axis=0), values[:-amount]], axis=0
    )


def _make_causal_arrays(record):
    current = record["current"].astype(np.float32)
    outputs = record["outputs"].astype(np.float32)
    displacement = outputs[:, 0]
    force = outputs[:, 1]

    x1, x2, x3 = (_shift(displacement, lag) for lag in (1, 2, 3))
    f1, f2 = (_shift(force, lag) for lag in (1, 2))
    i1 = _shift(current, 1)

    velocity = (x1 - x2) / TARGET_TIME_STEP
    acceleration = (x1 - 2.0 * x2 + x3) / TARGET_TIME_STEP**2
    quadratic_displacement = 3.0 * x1 - 3.0 * x2 + x3

    start = float(record["time"][record["pad"]])
    elapsed = np.maximum(record["time"] - start, 0.0).astype(np.float32)
    mass = np.full_like(current, record["mass_ratio"])

    all_features = np.column_stack(
        [
            current,
            current - i1,
            causal_dc_estimate(current),
            mass,
            1.0 / mass,
            elapsed,
            np.exp(-elapsed / 0.30),
            x1,
            velocity,
            acceleration,
            f1,
            f1 - f2,
            quadratic_displacement,
        ]
    ).astype(np.float32)
    features = all_features[:, SELECTED_FEATURE_INDICES]

    return {
        "features": features,
        "displacement_baseline": quadratic_displacement.astype(np.float32),
        "displacement_residual": (
            displacement - quadratic_displacement
        ).astype(np.float32),
    }


def _role_pairs(records, role):
    pairs = []
    for record_index, record in enumerate(records):
        if record["is_pure_test"]:
            continue
        for _, start, stop, block_role in record["blocks"]:
            if block_role != role:
                continue
            first = max(start, record["pad"])
            pairs.extend((record_index, target) for target in range(first, stop))
    return np.asarray(pairs, dtype=np.int64)


class _OneStepDataset(Dataset):
    def __init__(self, records, pairs, feature_mean, feature_std,
                 target_mean, target_std):
        self.records = records
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        record_index, target = self.pairs[item]
        record = self.records[record_index]
        window = record["one_step"]["features"][
            target - HISTORY_SAMPLES + 1:target + 1
        ]
        normalised_window = (
            (window - self.feature_mean) / self.feature_std
        ).astype(np.float32)
        residual = record["one_step"]["displacement_residual"][target]
        normalised_target = np.float32(
            (residual - self.target_mean) / self.target_std
        )
        return torch.from_numpy(normalised_window), torch.tensor(normalised_target)


def _training_statistics(records, pairs):
    # Every statistic is based only on samples labelled training.
    feature_rows = []
    residuals = []
    for record_index, target in pairs:
        record = records[record_index]
        feature_rows.append(record["one_step"]["features"][target])
        residuals.append(record["one_step"]["displacement_residual"][target])
    features = np.asarray(feature_rows, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    target_mean = float(residuals.mean())
    target_std = max(float(residuals.std()), 1e-8)
    return feature_mean, feature_std, target_mean, target_std


def _fit_startup_force_gain(records):
    """Fit F = gain*I on training samples only for zero-to-current startup."""
    numerator = 0.0
    denominator = 0.0
    for record in records:
        if record["is_pure_test"]:
            continue
        for _, start, stop, role in record["blocks"]:
            if role != "training":
                continue
            current = record["current"][start:stop].astype(np.float64)
            force = record["outputs"][start:stop, 1].astype(np.float64)
            numerator += float(np.dot(current, force))
            denominator += float(np.dot(current, current))
    return numerator / max(denominator, 1e-12)


def _force_prediction(record, targets, startup_gain):
    """Causal force estimate from I[k] and the latest reliable past F/I."""
    current = record["current"]
    force = record["outputs"][:, 1]
    predicted = np.empty(len(targets), dtype=np.float32)
    for position, target in enumerate(targets):
        present_current = float(current[target])
        if abs(present_current) <= ZERO_CURRENT_TOLERANCE_A:
            predicted[position] = 0.0
        else:
            # At a sine-wave zero crossing I[k-1] is too small to divide by.
            # Retain the most recent reliable causal gain instead of reverting
            # to a global value and creating a spike at every crossing.
            reliable = None
            for lag in range(1, 25):
                candidate = target - lag
                if candidate < 0:
                    break
                if abs(float(current[candidate])) >= MIN_PREVIOUS_CURRENT_A:
                    reliable = candidate
                    break
            if reliable is None:
                gain = startup_gain
            else:
                gain = float(force[reliable]) / float(current[reliable])
            predicted[position] = gain * present_current
    return predicted


@torch.no_grad()
def _predict_displacement(
    model, record, targets, statistics, device, residual_trust=1.0
):
    feature_mean, feature_std, target_mean, target_std = statistics
    windows = np.asarray(
        [
            (
                record["one_step"]["features"][
                    target - HISTORY_SAMPLES + 1:target + 1
                ] - feature_mean
            ) / feature_std
            for target in targets
        ],
        dtype=np.float32,
    )
    corrections = []
    for start in range(0, len(windows), 512):
        batch = torch.from_numpy(windows[start:start + 512]).to(device)
        corrections.append(model(batch).cpu().numpy())
    residual = np.concatenate(corrections) * target_std + target_mean
    baseline = record["one_step"]["displacement_baseline"][targets]
    return baseline + residual_trust * residual, baseline


def _metrics(name, kind, measured, predicted):
    rows = []
    for column, output, unit in (
        (0, "Displacement", "mm"),
        (1, "Lorentz force", "N"),
    ):
        values = calculate_metrics(
            measured[:, column],
            predicted[:, column],
            negligible_range=NEGLIGIBLE_SIGNAL_RANGE[output],
        )
        fit = values["Fit_percent"]
        if np.isfinite(fit):
            pass_95 = bool(fit >= FIT_TARGET_PERCENT)
            criterion = "Fit_percent >= 95"
        else:
            # For a flat reference, a variance-normalised fit is undefined.
            pass_95 = bool(values["MaxAbsError"] <= 1e-3)
            criterion = "flat reference: max abs error <= 0.001"
        rows.append(
            {
                "Evaluation": name,
                "Kind": kind,
                "Output": output,
                "Unit": unit,
                "Samples": int(len(measured)),
                "Pass95": pass_95,
                "PassCriterion": criterion,
                **values,
            }
        )
    return rows


def _plot_pure_test(name, time, measured, predicted, metrics, folder, number):
    rows = metrics[metrics["Evaluation"] == name]
    x_fit = rows[rows["Output"] == "Displacement"]["Fit_percent"].iloc[0]
    f_fit = rows[rows["Output"] == "Lorentz force"]["Fit_percent"].iloc[0]
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    for column, (output, unit, fit_value) in enumerate(
        [("Displacement", "mm", x_fit), ("Lorentz force", "N", f_fit)]
    ):
        axes[0, column].plot(time, measured[:, column], label="measured", lw=1.5)
        axes[0, column].plot(
            time, predicted[:, column], "--", label="one-step prediction", lw=1.2
        )
        fit_label = "n/a (flat)" if not np.isfinite(fit_value) else f"{fit_value:.2f}%"
        axes[0, column].set_title(f"{output}: fit {fit_label}")
        axes[0, column].set_ylabel(unit)
        axes[0, column].grid(True, alpha=0.35)
        axes[0, column].legend(frameon=False)
        error = measured[:, column] - predicted[:, column]
        axes[1, column].plot(time, error, color="#c62828", lw=0.9)
        axes[1, column].axhline(0.0, color="black", lw=0.7)
        axes[1, column].set_ylabel(f"error [{unit}]")
        axes[1, column].set_xlabel("time [s]")
        axes[1, column].grid(True, alpha=0.35)
    figure.suptitle(
        f"One-step measured-feedback prediction — {name}\n"
        "Pure-test targets were not used for fitting; this is not autonomous simulation"
    )
    figure.tight_layout()
    figure.savefig(folder / f"03_{number:02d}_one_step_pure_{name}.png", dpi=170)
    plt.close(figure)


def _plot_metric_table(metrics, folder):
    pure = metrics[metrics["Kind"] == "one_step_pure_test"].copy()
    displayed = []
    for _, row in pure.iterrows():
        fit = row["Fit_percent"]
        displayed.append(
            [
                row["Evaluation"],
                row["Output"],
                f"{row['RMSE']:.6f}",
                "n/a (flat)" if not np.isfinite(fit) else f"{fit:.2f}",
                "PASS" if row["Pass95"] else "FAIL",
            ]
        )
    figure, axis = plt.subplots(figsize=(13, 0.48 * len(displayed) + 2.0))
    axis.axis("off")
    table = axis.table(
        cellText=displayed,
        colLabels=["pure test record", "output", "RMSE", "fit %", "95% criterion"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for column in range(5):
        table[(0, column)].set_facecolor("#455a64")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row_number, source in enumerate(displayed, start=1):
        color = "#e8f5e9" if source[-1] == "PASS" else "#ffebee"
        for column in range(5):
            table[(row_number, column)].set_facecolor(color)
    axis.set_title(
        "One-step measured-feedback pure-test metrics — targets untouched\n"
        "Flat references use max absolute error <= 0.001 instead of fit %"
    )
    figure.tight_layout()
    figure.savefig(folder / "04_one_step_pure_metric_table.png", dpi=170)
    plt.close(figure)


def _plot_split_map(data, folder):
    table = data["split_table"].copy()
    order = data["development_names"] + data["pure_test_names"]
    colors = {
        "training": "#81c784",
        "validation": "#fbc02d",
        "test": "#ef6c00",
        "pure_test": "#8e24aa",
    }
    figure, axis = plt.subplots(figsize=(14, 0.42 * len(order) + 2.2))
    for row_number, name in enumerate(order):
        blocks = table[table["record"] == name]
        for _, block in blocks.iterrows():
            axis.barh(
                row_number,
                block["end_time_s"] - block["start_time_s"],
                left=block["start_time_s"],
                height=0.68,
                color=colors.get(block["role"], "#cfd8dc"),
                edgecolor="white",
                linewidth=0.3,
            )
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(
        [
            f"{name}  ({'PURE TEST' if name in data['pure_test_names'] else 'development'})"
            for name in order
        ]
    )
    axis.invert_yaxis()
    axis.set_xlabel("time within record [s]")
    axis.set_title(
        "Data split: whole pure-test records are untouched; development records "
        "contain training, validation, and internal-test blocks"
    )
    axis.grid(True, axis="x", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[role])
        for role in ("training", "validation", "test", "pure_test")
    ]
    axis.legend(
        handles,
        ["training", "validation", "internal test", "pure test"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(folder / "01_data_split_map.png", dpi=170)
    plt.close(figure)


def _plot_training_history(history, best_loss, residual_trust, folder):
    frame = pd.DataFrame(
        history, columns=["Epoch", "TrainingLoss", "ValidationLoss"]
    )
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(frame["Epoch"], frame["TrainingLoss"], marker="o", label="training")
    axis.plot(
        frame["Epoch"], frame["ValidationLoss"], marker="s", label="validation"
    )
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalised displacement-residual MSE")
    axis.set_title(
        f"One-step LSTM learning history; best validation loss {best_loss:.5g}; "
        f"validation residual trust {residual_trust:.3f}"
    )
    axis.grid(True, which="both", alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(folder / "02_one_step_learning_history.png", dpi=170)
    plt.close(figure)


def run_one_step_study(data, records, device, results_folder, figures_folder):
    """Train on development blocks, freeze, then evaluate whole pure records."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    results_folder = Path(results_folder)
    figures_folder = Path(figures_folder)
    figures_folder.mkdir(parents=True, exist_ok=True)
    prediction_folder = results_folder / "one_step_predictions"
    prediction_folder.mkdir(parents=True, exist_ok=True)

    for record in records:
        record["one_step"] = _make_causal_arrays(record)

    training_pairs = _role_pairs(records, "training")
    validation_pairs = _role_pairs(records, "validation")
    internal_test_pairs = _role_pairs(records, "test")
    statistics = _training_statistics(records, training_pairs)
    feature_mean, feature_std, target_mean, target_std = statistics

    training_dataset = _OneStepDataset(
        records, training_pairs, feature_mean, feature_std, target_mean, target_std
    )
    validation_dataset = _OneStepDataset(
        records, validation_pairs, feature_mean, feature_std, target_mean, target_std
    )
    counts = np.bincount(training_pairs[:, 0], minlength=len(records))
    weights = np.asarray(
        [1.0 / counts[record_index] for record_index, _ in training_pairs]
    )
    sampling_generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(
        weights,
        SAMPLES_PER_EPOCH,
        replacement=True,
        generator=sampling_generator,
    )
    training_loader = DataLoader(
        training_dataset, batch_size=BATCH_SIZE, sampler=sampler
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=512, shuffle=False
    )

    model = OneStepDisplacementLSTM().to(device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"\nFeature-ablation variant: {SELECTED_FEATURE_COUNT} retained "
        "feature(s)"
    )
    print("Retained features: " + ", ".join(SELECTED_FEATURE_NAMES))
    print(
        "Removed features: "
        + (", ".join(REMOVED_FEATURE_NAMES) if REMOVED_FEATURE_NAMES else "none")
    )
    print(f"Trainable LSTM parameters: {trainable_parameters:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_function = nn.MSELoss()
    best_loss = np.inf
    best_state = None
    stale_epochs = 0
    history = []

    print("\nTraining causal one-step displacement residual LSTM ...", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for inputs, targets in training_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                validation_losses.append(
                    float(loss_function(model(inputs), targets).item())
                )
        validation_loss = float(np.mean(validation_losses))
        history.append((epoch, float(np.mean(train_losses)), validation_loss))
        print(
            f"  epoch {epoch:02d}: train {history[-1][1]:.6f}, "
            f"validation {validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    startup_gain = _fit_startup_force_gain(records)

    # Fit one scalar trust factor on development validation blocks only.
    # If a platform-specific LSTM run learns an unhelpful correction, alpha
    # falls toward zero and the predictor safely returns to the already strong
    # causal constant-acceleration baseline. Pure-test targets are not read.
    validation_actual = []
    validation_correction = []
    for record_index in sorted(set(validation_pairs[:, 0].tolist())):
        subset = validation_pairs[validation_pairs[:, 0] == record_index]
        record = records[record_index]
        targets = subset[:, 1]
        full, baseline = _predict_displacement(
            model, record, targets, statistics, device, residual_trust=1.0
        )
        validation_actual.append(record["outputs"][targets, 0] - baseline)
        validation_correction.append(full - baseline)
    validation_actual = np.concatenate(validation_actual).astype(np.float64)
    validation_correction = np.concatenate(validation_correction).astype(np.float64)
    denominator = float(np.dot(validation_correction, validation_correction))
    if denominator <= 1e-20:
        residual_trust = 0.0
    else:
        residual_trust = float(
            np.clip(
                np.dot(validation_actual, validation_correction) / denominator,
                0.0,
                1.0,
            )
        )
    print(f"Validation-selected displacement residual trust: {residual_trust:.3f}")

    rows = []
    evaluations = []

    def evaluate(name, kind, pairs):
        pairs = np.asarray(pairs, dtype=np.int64)
        record_index = int(pairs[0, 0])
        record = records[record_index]
        targets = pairs[:, 1]
        displacement, displacement_baseline = _predict_displacement(
            model,
            record,
            targets,
            statistics,
            device,
            residual_trust=residual_trust,
        )
        force = _force_prediction(record, targets, startup_gain)
        measured = record["outputs"][targets]
        predicted = np.column_stack([displacement, force]).astype(np.float32)
        time = record["time"][targets]
        rows.extend(_metrics(name, kind, measured, predicted))
        evaluations.append(
            {
                "name": name,
                "kind": kind,
                "time": time,
                "current": record["current"][targets],
                "measured": measured,
                "predicted": predicted,
            }
        )
        pd.DataFrame(
            {
                "time_s": time,
                "current_A": record["current"][targets],
                "data_role": kind,
                "measured_displacement_mm": measured[:, 0],
                "predicted_displacement_mm": predicted[:, 0],
                "displacement_error_mm": measured[:, 0] - predicted[:, 0],
                "measured_force_N": measured[:, 1],
                "predicted_force_N": predicted[:, 1],
                "force_error_N": measured[:, 1] - predicted[:, 1],
            }
        ).to_csv(prediction_folder / f"{kind}__{name}.csv", index=False)

    # The checkpoint and every selected quantity are frozen before these
    # reporting evaluations. Training and validation predictions are exported
    # only so the complete figure report can display every development block;
    # they do not feed back into fitting or model selection.
    for role, pairs in (
        ("one_step_training", training_pairs),
        ("one_step_validation", validation_pairs),
        ("one_step_internal_test", internal_test_pairs),
    ):
        for record_index in sorted(set(pairs[:, 0].tolist())):
            subset = pairs[pairs[:, 0] == record_index]
            evaluate(records[record_index]["name"], role, subset)

    # The model and every statistic are frozen before this loop begins.
    for name in data["pure_test_names"]:
        record_index = data["index_of"][name]
        record = records[record_index]
        targets = np.arange(record["pad"], record["samples"], dtype=np.int64)
        evaluate(
            name,
            "one_step_pure_test",
            np.column_stack(
                [np.full(len(targets), record_index, dtype=np.int64), targets]
            ),
        )

    metrics = pd.DataFrame(rows)
    metrics["Test_origin"] = np.where(
        metrics["Evaluation"].isin(data.get("independent_test_names", [])),
        "Test_idpd.xlsx",
        np.where(
            metrics["Kind"] == "one_step_pure_test",
            "Total_Data.xlsx pure test",
            "Total_Data.xlsx development",
        ),
    )
    metrics.to_csv(results_folder / "one_step_metrics.csv", index=False)
    independent_metrics_path = (
        results_folder / "one_step_independent_test_metrics.csv"
    )
    if independent_metrics_path.exists():
        independent_metrics_path.unlink()
    independent_metrics = metrics[
        metrics["Evaluation"].isin(data.get("independent_test_names", []))
    ]
    if not independent_metrics.empty:
        independent_metrics.to_csv(independent_metrics_path, index=False)
    pd.DataFrame(history, columns=["Epoch", "TrainingLoss", "ValidationLoss"]).to_csv(
        results_folder / "one_step_training_history.csv", index=False
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "startup_force_gain_N_per_A": startup_gain,
            "displacement_residual_trust": residual_trust,
            "history_samples": HISTORY_SAMPLES,
            "selected_feature_count": SELECTED_FEATURE_COUNT,
            "selected_feature_indices_1_based": [
                index + 1 for index in SELECTED_FEATURE_INDICES
            ],
            "selected_feature_names": list(SELECTED_FEATURE_NAMES),
            "removed_feature_names": list(REMOVED_FEATURE_NAMES),
            "trainable_parameters": trainable_parameters,
            "prediction_mode": "one-step measured feedback, not autonomous",
        },
        results_folder / "one_step_lstm.pt",
    )
    with open(
        results_folder / "one_step_plot_metadata.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "best_validation_loss": float(best_loss),
                "displacement_residual_trust": float(residual_trust),
                "selected_feature_count": SELECTED_FEATURE_COUNT,
                "selected_feature_indices_1_based": [
                    index + 1 for index in SELECTED_FEATURE_INDICES
                ],
                "selected_feature_names": list(SELECTED_FEATURE_NAMES),
                "removed_feature_names": list(REMOVED_FEATURE_NAMES),
                "trainable_parameters": trainable_parameters,
                "prediction_mode": "one-step measured feedback, not autonomous",
            },
            handle,
            indent=2,
        )

    pure_metrics = metrics[metrics["Kind"] == "one_step_pure_test"]
    print(
        "\nONE-STEP PURE/INDEPENDENT TEST "
        "(whole records, untouched during model selection)"
    )
    print(
        pure_metrics[
            ["Evaluation", "Output", "RMSE", "Fit_percent", "Pass95"]
        ].to_string(index=False, float_format=lambda value: f"{value:.5f}")
    )

    generate_complete_figures(
        data=data,
        records=records,
        history=history,
        best_loss=best_loss,
        residual_trust=residual_trust,
        evaluations=evaluations,
        metrics=metrics,
        results_folder=results_folder,
        figures_folder=figures_folder,
    )
    return metrics
