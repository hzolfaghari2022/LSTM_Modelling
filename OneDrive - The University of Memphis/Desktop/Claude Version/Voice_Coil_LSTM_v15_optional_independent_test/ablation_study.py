"""Validation-only architecture and input-feature ablation for the V15 model.

This file is intentionally separate from main.py.  It does not change the
working one-step model, data preparation, figures, autonomous simulation, or
GitHub helper.  The study concerns the LSTM displacement correction only;
the project's Lorentz-force estimate is a separate causal rule and is not
affected by the LSTM width, depth, or selected input features.

Selection rule
--------------
Training blocks fit model weights and normalization.  Validation blocks choose
network size and removable features.  Whole pure-test experiments are read
only once, after the choice is frozen, for final confirmation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import random

# Required by PyTorch for repeatable CuBLAS operations on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import BATCH_SIZE, NEGLIGIBLE_SIGNAL_RANGE
from data_utils import prepare_data
from github_push import GitPushError, push_simulation
from one_step_lstm import HISTORY_SAMPLES, _make_causal_arrays, _role_pairs
from train_utils import calculate_metrics


HERE = Path(__file__).resolve().parent
BASE_WIDTH = 48
BASE_DEPTH = 2
BASE_HEAD_WIDTH = 32
FEATURE_NAMES = (
    "current i[k]",
    "current change i[k]-i[k-1]",
    "causal DC-current estimate",
    "mass ratio",
    "inverse mass ratio",
    "elapsed time",
    "startup indicator exp(-t/0.30)",
    "previous displacement x[k-1]",
    "velocity estimate from past displacement",
    "acceleration estimate from past displacement",
    "previous force F[k-1]",
    "previous force change F[k-1]-F[k-2]",
    "quadratic displacement extrapolation",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V15 network-capacity and/or feature-ablation studies."
    )
    parser.add_argument(
        "--study", choices=("capacity", "features", "all"), default="all",
        help="Choose network width/depth, input features, or both.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Fast workflow check; its numerical results are NOT reportable.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--widths", nargs="+", type=int, default=None)
    parser.add_argument("--depths", nargs="+", type=int, default=None)
    parser.add_argument("--feature-seed", type=int, default=123)
    parser.add_argument("--feature-max-removals", type=int, default=12)
    parser.add_argument("--validation-fit-drop", type=float, default=0.5)
    parser.add_argument("--validation-rmse-increase", type=float, default=5.0)
    return parser.parse_args()


def settings_from(arguments: argparse.Namespace) -> dict:
    seeds = arguments.seeds or ([123] if arguments.smoke_test else [123, 456, 789])
    seeds = list(dict.fromkeys(seeds))
    widths = arguments.widths or (
        [16, BASE_WIDTH] if arguments.smoke_test else [8, 16, 32, 48, 64, 96]
    )
    depths = arguments.depths or (
        [1, BASE_DEPTH] if arguments.smoke_test else [1, 2, 3, 4]
    )
    epochs = arguments.epochs or (1 if arguments.smoke_test else 20)
    samples = arguments.samples_per_epoch or (512 if arguments.smoke_test else 20000)
    maximum_removals = min(
        arguments.feature_max_removals,
        1 if arguments.smoke_test else len(FEATURE_NAMES) - 1,
    )
    if not seeds or any(value < 1 for value in widths + depths):
        raise ValueError("Seeds must be nonempty and widths/depths must be positive.")
    if epochs < 1 or samples < 1 or arguments.patience < 1:
        raise ValueError("Epochs, samples, and patience must be positive.")
    if maximum_removals < 0:
        raise ValueError("--feature-max-removals cannot be negative.")
    if arguments.validation_fit_drop < 0 or arguments.validation_rmse_increase < 0:
        raise ValueError("Validation tolerances cannot be negative.")
    root = HERE / (
        "AblationStudy_SMOKE_NOT_REPORTABLE"
        if arguments.smoke_test else "AblationStudy"
    )
    return {
        "seeds": seeds,
        "widths": sorted(set(widths)),
        "depths": sorted(set(depths)),
        "epochs": epochs,
        "samples": samples,
        "patience": arguments.patience,
        "max_removals": maximum_removals,
        "candidate_limit": 3 if arguments.smoke_test else None,
        "root": root,
        "label": "SMOKE TEST — NOT REPORTABLE" if arguments.smoke_test else "FULL STUDY",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class ConfigurableDisplacementLSTM(nn.Module):
    """The V15 one-step architecture with configurable recurrent width/depth."""

    def __init__(self, input_features: int, width: int, depth: int):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_features,
            width,
            num_layers=depth,
            batch_first=True,
            dropout=0.10 if depth > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(width, BASE_HEAD_WIDTH),
            nn.Tanh(),
            nn.Linear(BASE_HEAD_WIDTH, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.recurrent(sequence)
        return self.head(hidden[:, -1]).squeeze(-1)


class SelectedFeatureDataset(Dataset):
    def __init__(self, records, pairs, feature_indices, statistics):
        self.records = records
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.indices = np.asarray(feature_indices, dtype=np.int64)
        self.feature_mean, self.feature_std, self.target_mean, self.target_std = statistics

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int):
        record_index, target = self.pairs[item]
        record = self.records[record_index]
        window = record["one_step"]["features"][
            target - HISTORY_SAMPLES + 1:target + 1, self.indices
        ]
        window = ((window - self.feature_mean) / self.feature_std).astype(np.float32)
        residual = record["one_step"]["displacement_residual"][target]
        residual = np.float32((residual - self.target_mean) / self.target_std)
        return torch.from_numpy(np.ascontiguousarray(window)), torch.tensor(residual)


def prepare_study_data():
    data = prepare_data(HERE)
    records = data["records"]
    for record in records:
        if not record["is_pure_test"]:
            record["one_step"] = _make_causal_arrays(record)
    training_pairs = _role_pairs(records, "training")
    validation_pairs = _role_pairs(records, "validation")
    if len(training_pairs) == 0 or len(validation_pairs) == 0:
        raise RuntimeError("Training or validation samples are missing.")
    sample_features = next(
        record["one_step"]["features"] for record in records if not record["is_pure_test"]
    )
    if sample_features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Expected {len(FEATURE_NAMES)} one-step features but found "
            f"{sample_features.shape[1]}."
        )
    return data, training_pairs, validation_pairs


def training_statistics(records, pairs, feature_indices):
    indices = np.asarray(feature_indices, dtype=np.int64)
    features = np.stack([
        records[record_index]["one_step"]["features"][target, indices]
        for record_index, target in pairs
    ]).astype(np.float64)
    residuals = np.asarray([
        records[record_index]["one_step"]["displacement_residual"][target]
        for record_index, target in pairs
    ], dtype=np.float64)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    target_mean = float(residuals.mean())
    target_std = max(float(residuals.std()), 1e-8)
    return feature_mean, feature_std, target_mean, target_std


def predict_residual(bundle, record, targets, device):
    feature_mean, feature_std, target_mean, target_std = bundle["statistics"]
    indices = bundle["feature_indices"]
    windows = np.asarray([
        (
            record["one_step"]["features"][
                target - HISTORY_SAMPLES + 1:target + 1, indices
            ] - feature_mean
        ) / feature_std
        for target in targets
    ], dtype=np.float32)
    predictions = []
    bundle["model"].eval()
    with torch.no_grad():
        for start in range(0, len(windows), 512):
            batch = torch.from_numpy(windows[start:start + 512]).to(device)
            predictions.append(bundle["model"](batch).cpu().numpy())
    return np.concatenate(predictions) * target_std + target_mean


def fit_validation_trust(bundle, records, validation_pairs, device) -> float:
    actual_parts, predicted_parts = [], []
    for record_index in sorted(set(validation_pairs[:, 0].tolist())):
        subset = validation_pairs[validation_pairs[:, 0] == record_index]
        record = records[record_index]
        targets = subset[:, 1]
        actual_parts.append(record["one_step"]["displacement_residual"][targets])
        predicted_parts.append(predict_residual(bundle, record, targets, device))
    actual = np.concatenate(actual_parts).astype(np.float64)
    predicted = np.concatenate(predicted_parts).astype(np.float64)
    denominator = float(np.dot(predicted, predicted))
    if denominator <= 1e-20:
        return 0.0
    return float(np.clip(np.dot(actual, predicted) / denominator, 0.0, 1.0))


def train_case(
    records, training_pairs, validation_pairs, feature_indices, width, depth,
    seed, device, settings, verbose=True,
):
    seed_everything(seed)
    indices = tuple(int(value) for value in feature_indices)
    statistics = training_statistics(records, training_pairs, indices)
    training_set = SelectedFeatureDataset(records, training_pairs, indices, statistics)
    validation_set = SelectedFeatureDataset(records, validation_pairs, indices, statistics)
    counts = np.bincount(training_pairs[:, 0], minlength=len(records))
    weights = np.asarray([1.0 / counts[record_index] for record_index, _ in training_pairs])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=settings["samples"],
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    training_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )
    validation_loader = DataLoader(
        validation_set, batch_size=512, shuffle=False, num_workers=0
    )
    model = ConfigurableDisplacementLSTM(len(indices), width, depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_function = nn.MSELoss()
    best_loss = np.inf
    best_state = deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        training_losses = []
        for inputs, targets in training_loader:
            inputs, targets = inputs.to(device), targets.to(device)
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
                inputs, targets = inputs.to(device), targets.to(device)
                validation_losses.append(float(loss_function(model(inputs), targets).item()))
        train_loss = float(np.mean(training_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append((epoch, train_loss, validation_loss))
        if verbose:
            print(
                f"  seed={seed}, features={len(indices)}, width={width}, depth={depth}, "
                f"epoch={epoch:02d}: train={train_loss:.6f}, val={validation_loss:.6f}",
                flush=True,
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    bundle = {
        "model": model,
        "feature_indices": indices,
        "statistics": statistics,
        "seed": seed,
        "width": width,
        "depth": depth,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_normalized_validation_loss": float(best_loss),
        "history": history,
    }
    bundle["trust"] = fit_validation_trust(bundle, records, validation_pairs, device)
    return bundle


def displacement_metrics(bundle, records, pairs, device):
    measured_parts, predicted_parts = [], []
    for record_index in sorted(set(pairs[:, 0].tolist())):
        subset = pairs[pairs[:, 0] == record_index]
        record = records[record_index]
        targets = subset[:, 1]
        residual = predict_residual(bundle, record, targets, device)
        baseline = record["one_step"]["displacement_baseline"][targets]
        predicted = baseline + bundle["trust"] * residual
        measured_parts.append(record["outputs"][targets, 0])
        predicted_parts.append(predicted)
    measured = np.concatenate(measured_parts)
    predicted = np.concatenate(predicted_parts)
    return calculate_metrics(
        measured,
        predicted,
        negligible_range=NEGLIGIBLE_SIGNAL_RANGE["Displacement"],
    )


def pure_test_metrics(bundle, data, device) -> pd.DataFrame:
    # This function is called only after architecture/features are frozen.
    rows = []
    pooled_measured, pooled_predicted = [], []
    for name in data["pure_test_names"]:
        record = data["records"][data["index_of"][name]]
        if "one_step" not in record:
            record["one_step"] = _make_causal_arrays(record)
        targets = np.arange(record["pad"], record["samples"], dtype=np.int64)
        residual = predict_residual(bundle, record, targets, device)
        predicted = (
            record["one_step"]["displacement_baseline"][targets]
            + bundle["trust"] * residual
        )
        measured = record["outputs"][targets, 0]
        values = calculate_metrics(
            measured, predicted,
            negligible_range=NEGLIGIBLE_SIGNAL_RANGE["Displacement"],
        )
        error = measured.astype(np.float64) - predicted.astype(np.float64)
        rows.append({
            "Experiment": name,
            "Samples": len(targets),
            **values,
            "ErrorMean": float(np.mean(error)),
            "ErrorStd": float(np.std(error, ddof=1)) if len(error) > 1 else 0.0,
        })
        pooled_measured.append(measured)
        pooled_predicted.append(predicted)
    measured = np.concatenate(pooled_measured)
    predicted = np.concatenate(pooled_predicted)
    values = calculate_metrics(
        measured, predicted,
        negligible_range=NEGLIGIBLE_SIGNAL_RANGE["Displacement"],
    )
    error = measured.astype(np.float64) - predicted.astype(np.float64)
    rows.append({
        "Experiment": "ALL_PURE_TESTS_POOLED",
        "Samples": len(measured),
        **values,
        "ErrorMean": float(np.mean(error)),
        "ErrorStd": float(np.std(error, ddof=1)),
    })
    return pd.DataFrame(rows)


def metric_columns(prefix: str, values: dict) -> dict:
    return {
        f"{prefix}_RMSE_mm": values["RMSE"],
        f"{prefix}_MAE_mm": values["MAE"],
        f"{prefix}_R2": values["R2"],
        f"{prefix}_Fit_percent": values["Fit_percent"],
    }


def summarise_capacity(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Train_RMSE_mm", "Validation_RMSE_mm", "Train_MAE_mm",
        "Validation_MAE_mm", "Train_R2", "Validation_R2",
        "Train_Fit_percent", "Validation_Fit_percent",
    ]
    rows = []
    for (width, depth, parameters), group in raw.groupby(
        ["Width", "Depth", "Parameter_count"], sort=True
    ):
        row = {
            "Width": width,
            "Depth": depth,
            "Parameter_count": parameters,
            "Seeds": int(group["Seed"].nunique()),
        }
        for column in columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_SD"] = (
                float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_capacity(summary: pd.DataFrame, figures: Path, label: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sweeps = (
        (axes[0], summary[summary["Depth"] == BASE_DEPTH].sort_values("Width"),
         "Width study (depth fixed at 2)", "Width (LSTM units)"),
        (axes[1], summary[summary["Width"] == BASE_WIDTH].sort_values("Depth"),
         "Depth study (width fixed at 48)", "Depth (LSTM layers)"),
    )
    for axis, subset, title, xlabel in sweeps:
        x_column = "Width" if "Width" in xlabel else "Depth"
        x = subset[x_column].to_numpy()
        train = subset["Train_RMSE_mm_mean"].to_numpy()
        validation = subset["Validation_RMSE_mm_mean"].to_numpy()
        train_sd = subset["Train_RMSE_mm_SD"].to_numpy()
        validation_sd = subset["Validation_RMSE_mm_SD"].to_numpy()
        axis.errorbar(x, train, yerr=train_sd, marker="o", color="#0072B2",
                      capsize=3, label="training RMSE")
        axis.errorbar(x, validation, yerr=validation_sd, marker="s", color="#7B3294",
                      capsize=3, label="validation RMSE")
        if len(subset):
            best_position = int(np.nanargmin(validation))
            best_x = x[best_position]
            if len(x) > 1:
                spacing = float(np.min(np.diff(np.sort(np.unique(x)))))
            else:
                spacing = max(abs(float(best_x)) * 0.15, 0.5)
            lower, upper = axis.get_xlim()
            axis.axvspan(lower, best_x - 0.45 * spacing, color="#F0E442", alpha=0.07)
            axis.axvspan(best_x - 0.45 * spacing, best_x + 0.45 * spacing,
                         color="#009E73", alpha=0.10)
            axis.axvspan(best_x + 0.45 * spacing, upper, color="#D55E00", alpha=0.06)
            axis.axvline(best_x, color="#009E73", ls="--", lw=1.5,
                         label=f"validation-selected: {best_x:g}")
            base_x = BASE_WIDTH if x_column == "Width" else BASE_DEPTH
            axis.axvline(base_x, color="#E69F00", ls=":", lw=1.5,
                         label=f"current V15: {base_x}")
        for row in subset.itertuples():
            axis.annotate(
                f"{int(row.Parameter_count):,} p",
                (getattr(row, x_column), row.Validation_RMSE_mm_mean),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Displacement RMSE (mm)")
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Measured network capacity study — lower validation error is better\n"
        "Selection uses development validation data, never pure-test data"
        + (f"\n{label}" if "SMOKE" in label else "")
    )
    figure.tight_layout()
    figure.savefig(figures / "network_width_depth_capacity.png", dpi=220)
    plt.close(figure)


def run_capacity(arguments, settings, data, training_pairs, validation_pairs, device):
    root = settings["root"] / "NetworkCapacity"
    results, figures = root / "Results", root / "Figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    all_features = tuple(range(len(FEATURE_NAMES)))
    cases = {(width, BASE_DEPTH) for width in settings["widths"]}
    cases.update((BASE_WIDTH, depth) for depth in settings["depths"])
    rows = []
    print("\nNETWORK WIDTH/DEPTH STUDY")
    for width, depth in sorted(cases, key=lambda item: (item[1], item[0])):
        for seed in settings["seeds"]:
            print(f"\nCase: width={width}, depth={depth}, seed={seed}")
            bundle = train_case(
                data["records"], training_pairs, validation_pairs, all_features,
                width, depth, seed, device, settings,
            )
            train_metric = displacement_metrics(
                bundle, data["records"], training_pairs, device
            )
            validation_metric = displacement_metrics(
                bundle, data["records"], validation_pairs, device
            )
            rows.append({
                "Width": width,
                "Depth": depth,
                "Seed": seed,
                "Parameter_count": bundle["parameter_count"],
                "Residual_trust": bundle["trust"],
                "Best_normalized_validation_loss": bundle["best_normalized_validation_loss"],
                **metric_columns("Train", train_metric),
                **metric_columns("Validation", validation_metric),
            })
            del bundle
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raw = pd.DataFrame(rows)
    summary = summarise_capacity(raw)
    raw.to_csv(results / "network_capacity_runs.csv", index=False)
    summary.to_csv(results / "network_capacity_summary.csv", index=False)
    plot_capacity(summary, figures, settings["label"])

    selected = summary.loc[summary["Validation_RMSE_mm_mean"].idxmin()]
    selected_width, selected_depth = int(selected["Width"]), int(selected["Depth"])
    print(
        f"\nValidation-selected architecture: width={selected_width}, "
        f"depth={selected_depth}, parameters={int(selected['Parameter_count']):,}"
    )
    # Confirm only the frozen choice on the pure tests.
    final_bundle = train_case(
        data["records"], training_pairs, validation_pairs, all_features,
        selected_width, selected_depth, settings["seeds"][0], device, settings,
    )
    pure = pure_test_metrics(final_bundle, data, device)
    pure.to_csv(results / "selected_architecture_pure_test_confirmation.csv", index=False)
    base = summary[(summary["Width"] == BASE_WIDTH) & (summary["Depth"] == BASE_DEPTH)].iloc[0]
    selected_change = 100.0 * (
        selected["Validation_RMSE_mm_mean"] - base["Validation_RMSE_mm_mean"]
    ) / base["Validation_RMSE_mm_mean"]
    (results / "selected_architecture.txt").write_text(
        "\n".join([
            f"Study label: {settings['label']}",
            "Selection source: development validation data only",
            f"Selected width: {selected_width}",
            f"Selected depth: {selected_depth}",
            f"Trainable parameters: {int(selected['Parameter_count'])}",
            f"Current V15 width/depth: {BASE_WIDTH}/{BASE_DEPTH}",
            f"Validation RMSE change versus current V15: {selected_change:.6g}%",
            "Yellow/green/red shading means lower/selected/higher capacity; it does not by itself prove underfitting or overfitting.",
            "Underfitting is supported only if lower-capacity models have worse training and validation errors.",
            "Overfitting is supported only if higher-capacity models reduce training error while validation error worsens.",
            "Pure tests were used only after selection was frozen.",
            "Lorentz force is unchanged because it is produced by a separate causal rule.",
        ]) + "\n",
        encoding="utf-8",
    )
    return selected_width, selected_depth


def plot_feature_ablation(table: pd.DataFrame, path: Path, arguments, label: str):
    if table.empty:
        return
    selected = table[table["Selected_this_step"]].copy()
    figure, axes = plt.subplots(2, 1, figsize=(13, 9))
    labels = [
        f"S{row.Step}: -{row.Candidate_removed}" for row in table.itertuples()
    ]
    colors = ["#009E73" if value else "#9E9E9E" for value in table["Selected_this_step"]]
    axes[0].bar(range(len(table)), table["RMSE_change_from_full_percent"], color=colors)
    axes[0].axhline(arguments.validation_rmse_increase, color="#D55E00", ls="--",
                    label="maximum accepted increase")
    axes[0].set_ylabel("Validation RMSE change from full model (%)")
    axes[0].set_xticks(range(len(table)), labels, rotation=75, ha="right", fontsize=7)
    axes[0].legend(frameon=False)
    if not selected.empty:
        axes[1].plot(
            [len(FEATURE_NAMES)] + selected["Retained_feature_count"].tolist(),
            [0.0] + selected["RMSE_change_from_full_percent"].tolist(),
            marker="o", color="#7B3294", label="selected elimination path",
        )
        axes[1].invert_xaxis()
    axes[1].axhline(arguments.validation_rmse_increase, color="#D55E00", ls="--")
    axes[1].set_xlabel("Retained LSTM input features")
    axes[1].set_ylabel("Validation RMSE change from full model (%)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        "Validation-only backward feature ablation (green bars were removed)"
        + (f"\n{label}" if "SMOKE" in label else "")
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_feature_ablation(arguments, settings, data, training_pairs, validation_pairs, device):
    root = settings["root"] / "FeatureAblation"
    results, figures = root / "Results", root / "Figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    full_indices = tuple(range(len(FEATURE_NAMES)))
    print("\nFEATURE ABLATION: validation-only backward elimination")
    full_bundle = train_case(
        data["records"], training_pairs, validation_pairs, full_indices,
        BASE_WIDTH, BASE_DEPTH, arguments.feature_seed, device, settings,
    )
    full_metric = displacement_metrics(
        full_bundle, data["records"], validation_pairs, device
    )
    current_indices = full_indices
    current_bundle = full_bundle
    rows, removed_names = [], []
    for step in range(1, settings["max_removals"] + 1):
        candidates = list(current_indices)
        if settings["candidate_limit"] is not None:
            candidates = candidates[:settings["candidate_limit"]]
        best = None
        for removed in candidates:
            retained = tuple(index for index in current_indices if index != removed)
            print(f"  step {step}: testing removal of {FEATURE_NAMES[removed]}")
            bundle = train_case(
                data["records"], training_pairs, validation_pairs, retained,
                BASE_WIDTH, BASE_DEPTH, arguments.feature_seed, device, settings,
                verbose=False,
            )
            metric = displacement_metrics(bundle, data["records"], validation_pairs, device)
            rmse_change = 100.0 * (
                metric["RMSE"] - full_metric["RMSE"]
            ) / full_metric["RMSE"]
            fit_drop = full_metric["Fit_percent"] - metric["Fit_percent"]
            accepted = bool(
                rmse_change <= arguments.validation_rmse_increase
                and fit_drop <= arguments.validation_fit_drop
            )
            row = {
                "Step": step,
                "Candidate_removed": FEATURE_NAMES[removed],
                "Candidate_index_1_based": removed + 1,
                "Retained_feature_count": len(retained),
                "Retained_features": "; ".join(FEATURE_NAMES[i] for i in retained),
                "Validation_RMSE_mm": metric["RMSE"],
                "Validation_MAE_mm": metric["MAE"],
                "Validation_R2": metric["R2"],
                "Validation_fit_percent": metric["Fit_percent"],
                "RMSE_change_from_full_percent": rmse_change,
                "Fit_drop_from_full_points": fit_drop,
                "Within_validation_limits": accepted,
                "Selected_this_step": False,
            }
            rows.append(row)
            row_index = len(rows) - 1
            if accepted and (best is None or metric["RMSE"] < best["metric"]["RMSE"]):
                best = {
                    "removed": removed,
                    "indices": retained,
                    "bundle": bundle,
                    "metric": metric,
                    "row_index": row_index,
                }
        if best is None:
            print("  No additional feature satisfies the validation limits.")
            break
        rows[best["row_index"]]["Selected_this_step"] = True
        removed_names.append(FEATURE_NAMES[best["removed"]])
        current_indices = best["indices"]
        current_bundle = best["bundle"]
        print(f"  selected removal: {FEATURE_NAMES[best['removed']]}")

    table = pd.DataFrame(rows)
    table.to_csv(results / "feature_ablation_table.csv", index=False)
    pd.DataFrame({
        "Feature_index_1_based": range(1, len(FEATURE_NAMES) + 1),
        "Feature_name": FEATURE_NAMES,
        "Retained_in_final_model": [index in current_indices for index in range(len(FEATURE_NAMES))],
    }).to_csv(results / "feature_inventory_and_selection.csv", index=False)
    plot_feature_ablation(
        table, figures / "feature_ablation_results.png", arguments, settings["label"]
    )

    # Confirm only the frozen reduced feature set on pure tests.
    pure = pure_test_metrics(current_bundle, data, device)
    pure.to_csv(results / "selected_features_pure_test_confirmation.csv", index=False)
    individual = pure[pure["Experiment"] != "ALL_PURE_TESTS_POOLED"]
    passed = bool(
        individual["Fit_percent"].notna().all()
        and (individual["Fit_percent"] >= 95.0).all()
    )
    (results / "selected_features.txt").write_text(
        "\n".join([
            f"Study label: {settings['label']}",
            "Selection source: development validation data only",
            f"Removed feature count: {len(removed_names)}",
            "Removed: " + (", ".join(removed_names) if removed_names else "none"),
            "Retained: " + ", ".join(FEATURE_NAMES[i] for i in current_indices),
            f"Pure-test confirmation (all displacement fit >= 95%): {passed}",
            "Pure tests were not used to choose removals.",
            "Lorentz force is unchanged because it is produced by a separate causal rule.",
        ]) + "\n",
        encoding="utf-8",
    )
    return current_indices


def write_readme(settings: dict, studies: str) -> None:
    results = settings["root"] / "README_RESULTS.txt"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        "\n".join([
            f"Ablation evaluation: {settings['label']}",
            f"Studies requested: {studies}",
            "The LSTM predicts displacement correction only.",
            "Lorentz force is calculated by the unchanged causal force rule.",
            "Lower validation RMSE is better.",
            "Architecture and feature choices use validation data only.",
            "Pure-test data are used only after the choice is frozen.",
            "Training and normalization use training data only.",
            "Do not report smoke-test numerical results.",
        ]) + "\n",
        encoding="utf-8",
    )


def maybe_push() -> None:
    if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") == "1":
        print("\nGitHub push skipped because DLSTM_SKIP_GITHUB_PUSH=1.")
        return
    try:
        push_simulation(
            HERE,
            commit_message="Add V15 network-capacity and feature-ablation results",
        )
    except (GitPushError, ImportError) as error:
        print("\nIMPORTANT: ablation finished, but GitHub push failed:")
        print(error)
        print("Run 'python push_now.py' after correcting the Git configuration.")


def main() -> None:
    arguments = parse_arguments()
    settings = settings_from(arguments)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training device: {device}, {torch.cuda.get_device_name(0)}")
    else:
        print("Training device: cpu")
    print(f"Evaluation label: {settings['label']}")
    print("The existing main.py and original model settings are not modified.")
    print("The ablation target is the LSTM displacement correction, not force.")
    data, training_pairs, validation_pairs = prepare_study_data()
    print(f"Training windows: {len(training_pairs):,}")
    print(f"Validation windows: {len(validation_pairs):,}")
    print(f"One-step input features: {len(FEATURE_NAMES)}")

    if arguments.study in ("capacity", "all"):
        run_capacity(
            arguments, settings, data, training_pairs, validation_pairs, device
        )
    if arguments.study in ("features", "all"):
        run_feature_ablation(
            arguments, settings, data, training_pairs, validation_pairs, device
        )
    write_readme(settings, arguments.study)
    maybe_push()
    print("\nDone.")


if __name__ == "__main__":
    main()
