"""
HYBRID HAMMERSTEIN-LSTM SYSTEM IDENTIFICATION
=============================================

Run this single file directly in VS Code.

Data use:
    Development: 67, 87, 107, and 127 mA
    Independent test: 147 mA

Paper ideas:
    Ogunmolu/Gans:
        static nonlinear neural block -> recurrent dynamic block

    Wang:
        series-parallel training with measured output history,
        followed by scheduled recursive rollout training

Place this file in the same folder as:
    COMSOL_07_13_2026.xlsx

Then open this file in VS Code and click "Run Python File".
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# 0. AUTOMATIC PACKAGE CHECK
# ---------------------------------------------------------------------

import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
    "torch": "torch",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *missing_packages]
    )


# ---------------------------------------------------------------------
# 1. IMPORTS AND USER SETTINGS
# ---------------------------------------------------------------------

import csv
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Main settings. These are the values you may change.
WINDOW = 120
HIDDEN_SIZE = 64
STATIC_SIZE = 24

ONE_STEP_EPOCHS = 30
ROLLOUT_EPOCHS = 12
ROLLOUT_HORIZON = 12

BATCH_SIZE = 256
ROLLOUT_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ROLLOUT_LEARNING_RATE = 2e-4

BLOCK_SIZE = 1000
VALIDATION_FRACTION = 0.20
TRAIN_STRIDE = 5
VALIDATION_STRIDE = 3
ROLLOUT_STRIDE = 100

RANDOM_SEED = 123

DEVELOPMENT_SHEETS = [
    "DC_Offset_67mA",
    "DC_Offset_87mA",
    "DC_Offset_107mA",
    "DC_Offset_127mA",
]
TEST_SHEET = "DC_Offset_147mA"

# Set QUICK_MODE = True only to check that the program works.
# Keep it False for the final simulation.
QUICK_MODE = os.environ.get("QUICK_MODE", "0") == "1"

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# ---------------------------------------------------------------------
# 2. SMALL DATA CLASSES
# ---------------------------------------------------------------------

@dataclass
class Block:
    sheet: str
    start: int
    end: int
    role: str


@dataclass
class Normalizer:
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray

    def normalize_input(self, values: np.ndarray) -> np.ndarray:
        return (values - self.input_mean) / self.input_std

    def normalize_output(self, values: np.ndarray) -> np.ndarray:
        return (values - self.output_mean) / self.output_std

    def restore_output(self, values: np.ndarray) -> np.ndarray:
        return values * self.output_std + self.output_mean


# ---------------------------------------------------------------------
# 3. DATA READING AND PREPARATION
# ---------------------------------------------------------------------

def set_random_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def find_workbook() -> Path:
    preferred = ROOT / "COMSOL_07_13_2026.xlsx"

    if preferred.exists():
        return preferred

    workbooks = [
        path
        for path in ROOT.glob("*.xlsx")
        if not path.name.startswith("~$")
    ]

    if len(workbooks) == 1:
        return workbooks[0]

    raise FileNotFoundError(
        "Put COMSOL_07_13_2026.xlsx in the same folder as this Python file."
    )


def load_sheet(
    workbook: openpyxl.Workbook,
    sheet_name: str,
    maximum_rows: int | None = None,
) -> np.ndarray:
    """
    Read the first four numeric columns:
        time, displacement, coil current, Lorentz force.
    """
    worksheet = workbook[sheet_name]
    rows = []

    for row in worksheet.iter_rows(
        min_col=1,
        max_col=4,
        values_only=True,
    ):
        try:
            values = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        except (TypeError, ValueError):
            continue

        if np.all(np.isfinite(values)):
            rows.append(values)

            if maximum_rows is not None and len(rows) >= maximum_rows:
                break

    data = np.asarray(rows, dtype=np.float32)

    if len(data) <= WINDOW + ROLLOUT_HORIZON:
        raise ValueError(f"Not enough numeric rows in sheet {sheet_name}")

    return data


def dc_offset_from_sheet(sheet_name: str) -> float:
    match = re.search(r"_(\d+)mA", sheet_name)

    if match is None:
        raise ValueError(f"Cannot read the DC offset from {sheet_name}")

    return float(match.group(1)) / 1000.0


def known_input_features(raw: np.ndarray, sheet_name: str) -> np.ndarray:
    """
    Known input features:
        coil current,
        change in current,
        DC operating offset.
    """
    current = raw[:, 2]
    change_in_current = np.r_[0.0, np.diff(current)].astype(np.float32)
    dc_offset = np.full_like(current, dc_offset_from_sheet(sheet_name))

    return np.column_stack(
        [current, change_in_current, dc_offset]
    ).astype(np.float32)


def measured_outputs(raw: np.ndarray) -> np.ndarray:
    """Outputs: displacement and Lorentz force."""
    return raw[:, [1, 3]].astype(np.float32)


def create_random_blocks(
    development_data: dict[str, np.ndarray],
    block_size: int,
) -> list[Block]:
    """
    Randomly assign complete non-overlapping time blocks.

    We do not randomly split individual overlapping windows because that
    would place nearly identical samples in training and validation.
    """
    random_generator = np.random.default_rng(RANDOM_SEED)
    blocks = []

    for sheet_name in DEVELOPMENT_SHEETS:
        starts = list(
            range(0, len(development_data[sheet_name]), block_size)
        )

        number_of_validation_blocks = max(
            1,
            round(VALIDATION_FRACTION * len(starts)),
        )

        validation_ids = set(
            random_generator.choice(
                len(starts),
                size=number_of_validation_blocks,
                replace=False,
            ).tolist()
        )

        for block_number, start in enumerate(starts):
            end = min(
                start + block_size,
                len(development_data[sheet_name]),
            )

            if end - start <= WINDOW + ROLLOUT_HORIZON:
                continue

            role = (
                "validation"
                if block_number in validation_ids
                else "training"
            )

            blocks.append(Block(sheet_name, start, end, role))

    return blocks


def create_normalizer(
    development_data: dict[str, np.ndarray],
    blocks: list[Block],
) -> Normalizer:
    """Calculate statistics from training blocks only."""
    input_rows = []
    output_rows = []

    for block in blocks:
        if block.role != "training":
            continue

        input_rows.append(
            known_input_features(
                development_data[block.sheet],
                block.sheet,
            )[block.start:block.end]
        )

        output_rows.append(
            measured_outputs(
                development_data[block.sheet]
            )[block.start:block.end]
        )

    all_inputs = np.concatenate(input_rows)
    all_outputs = np.concatenate(output_rows)

    input_std = all_inputs.std(axis=0)
    output_std = all_outputs.std(axis=0)

    input_std[input_std < 1e-12] = 1.0
    output_std[output_std < 1e-12] = 1.0

    return Normalizer(
        input_mean=all_inputs.mean(axis=0).astype(np.float32),
        input_std=input_std.astype(np.float32),
        output_mean=all_outputs.mean(axis=0).astype(np.float32),
        output_std=output_std.astype(np.float32),
    )


def normalize_all_records(
    raw_data: dict[str, np.ndarray],
    normalizer: Normalizer,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    input_data = {}
    output_data = {}

    for sheet_name, raw in raw_data.items():
        input_data[sheet_name] = normalizer.normalize_input(
            known_input_features(raw, sheet_name)
        ).astype(np.float32)

        output_data[sheet_name] = normalizer.normalize_output(
            measured_outputs(raw)
        ).astype(np.float32)

    return input_data, output_data


# ---------------------------------------------------------------------
# 4. PYTORCH DATASETS
# ---------------------------------------------------------------------

class OneStepDataset(Dataset):
    """Measured output history -> next measured output."""

    def __init__(
        self,
        input_data: dict[str, np.ndarray],
        output_data: dict[str, np.ndarray],
        blocks: list[Block],
        role: str,
        stride: int,
    ) -> None:
        self.input_data = input_data
        self.output_data = output_data
        self.indices = []

        for block in blocks:
            if block.role != role:
                continue

            for target_index in range(
                block.start + WINDOW,
                block.end,
                stride,
            ):
                self.indices.append(
                    (block.sheet, target_index)
                )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sheet_name, target_index = self.indices[index]
        start_index = target_index - WINDOW

        return (
            torch.from_numpy(
                self.input_data[sheet_name][start_index:target_index]
            ),
            torch.from_numpy(
                self.output_data[sheet_name][start_index:target_index]
            ),
            torch.from_numpy(
                self.output_data[sheet_name][target_index]
            ),
        )


class RolloutDataset(Dataset):
    """Warm-up history followed by a short recursive prediction horizon."""

    def __init__(
        self,
        input_data: dict[str, np.ndarray],
        output_data: dict[str, np.ndarray],
        blocks: list[Block],
        role: str,
        stride: int,
        horizon: int,
    ) -> None:
        self.input_data = input_data
        self.output_data = output_data
        self.horizon = horizon
        self.indices = []

        for block in blocks:
            if block.role != role:
                continue

            last_start = block.end - WINDOW - horizon

            for start_index in range(
                block.start,
                last_start,
                stride,
            ):
                self.indices.append(
                    (block.sheet, start_index)
                )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sheet_name, start_index = self.indices[index]
        end_index = start_index + WINDOW + self.horizon

        return (
            torch.from_numpy(
                self.input_data[sheet_name][start_index:end_index]
            ),
            torch.from_numpy(
                self.output_data[sheet_name][start_index:end_index]
            ),
        )


# ---------------------------------------------------------------------
# 5. HYBRID HAMMERSTEIN-LSTM MODEL
# ---------------------------------------------------------------------

class HammersteinLSTM(nn.Module):
    """
    Ogunmolu/Gans idea:
        static nonlinear block -> recurrent dynamic block

    Wang idea:
        measured or predicted output history enters the dynamic model
    """

    def __init__(self) -> None:
        super().__init__()

        # Static nonlinear transformation of the known input.
        self.static_nonlinearity = nn.Sequential(
            nn.Linear(3, STATIC_SIZE),
            nn.Tanh(),
            nn.Linear(STATIC_SIZE, STATIC_SIZE),
            nn.Tanh(),
        )

        # Dynamic recurrent block.
        self.lstm = nn.LSTM(
            input_size=STATIC_SIZE + 2,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

        # Predict the next change in displacement and force.
        self.delta_output = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        input_history: torch.Tensor,
        output_history: torch.Tensor,
    ) -> torch.Tensor:
        nonlinear_input = self.static_nonlinearity(
            input_history
        )

        recurrent_input = torch.cat(
            [nonlinear_input, output_history],
            dim=-1,
        )

        recurrent_output, _ = self.lstm(
            recurrent_input
        )

        predicted_change = self.delta_output(
            recurrent_output[:, -1]
        )

        # Residual prediction improves numerical stability.
        return output_history[:, -1] + predicted_change


# ---------------------------------------------------------------------
# 6. TRAINING STAGE 1: SERIES-PARALLEL
# ---------------------------------------------------------------------

def evaluate_one_step_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for input_history, output_history, target in data_loader:
            input_history = input_history.to(device)
            output_history = output_history.to(device)
            target = target.to(device)

            prediction = model(
                input_history,
                output_history,
            )

            loss = nn.functional.mse_loss(
                prediction,
                target,
            )

            total_loss += float(loss) * len(target)
            total_samples += len(target)

    return total_loss / total_samples


def train_series_parallel(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> list[dict]:
    """
    Wang series-parallel training:
    measured displacement and force histories are always supplied.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-6,
    )

    history = []
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for input_history, output_history, target in training_loader:
            input_history = input_history.to(device)
            output_history = output_history.to(device)
            target = target.to(device)

            # Small noise makes feedback more robust.
            noisy_output_history = (
                output_history
                + 0.01 * torch.randn_like(output_history)
            )

            prediction = model(
                input_history,
                noisy_output_history,
            )

            loss = nn.functional.mse_loss(
                prediction,
                target,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += float(loss) * len(target)
            total_samples += len(target)

        training_loss = total_loss / total_samples

        validation_loss = evaluate_one_step_loss(
            model,
            validation_loader,
            device,
        )

        history.append(
            {
                "stage": "series_parallel",
                "epoch": epoch,
                "teacher_forcing_ratio": 1.0,
                "training_normalized_mse": training_loss,
                "validation_normalized_mse": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        print(
            f"Series-parallel {epoch:02d}/{epochs} | "
            f"train={training_loss:.7f} | "
            f"validation={validation_loss:.7f}"
        )

    if best_state is None:
        raise RuntimeError("Series-parallel training failed.")

    model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------
# 7. TRAINING STAGE 2: SCHEDULED MULTI-STEP ROLLOUT
# ---------------------------------------------------------------------

def recursive_batch_prediction(
    model: nn.Module,
    input_sequence: torch.Tensor,
    output_sequence: torch.Tensor,
    horizon: int,
    teacher_forcing_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_history = input_sequence[:, :WINDOW].clone()
    output_history = output_sequence[:, :WINDOW].clone()

    predictions = []
    targets = []

    for step in range(horizon):
        prediction = model(
            input_history,
            output_history,
        )

        target = output_sequence[:, WINDOW + step]

        predictions.append(prediction)
        targets.append(target)

        if teacher_forcing_ratio <= 0.0:
            feedback = prediction
        else:
            use_measured = (
                torch.rand(
                    len(prediction),
                    1,
                    device=prediction.device,
                )
                < teacher_forcing_ratio
            )

            feedback = torch.where(
                use_measured,
                target,
                prediction,
            )

        next_input = input_sequence[
            :,
            WINDOW + step:WINDOW + step + 1,
        ]

        input_history = torch.cat(
            [input_history[:, 1:], next_input],
            dim=1,
        )

        output_history = torch.cat(
            [output_history[:, 1:], feedback[:, None]],
            dim=1,
        )

    return (
        torch.stack(predictions, dim=1),
        torch.stack(targets, dim=1),
    )


def evaluate_rollout_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    horizon: int,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for input_sequence, output_sequence in data_loader:
            input_sequence = input_sequence.to(device)
            output_sequence = output_sequence.to(device)

            prediction, target = recursive_batch_prediction(
                model,
                input_sequence,
                output_sequence,
                horizon,
                teacher_forcing_ratio=0.0,
            )

            losses.append(
                float(
                    nn.functional.mse_loss(
                        prediction,
                        target,
                    )
                )
            )

    return float(np.mean(losses))


def train_scheduled_rollout(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    horizon: int,
) -> list[dict]:
    """
    Gradually replace measured feedback with predicted feedback.
    The loss includes all steps in the rollout horizon.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ROLLOUT_LEARNING_RATE,
        weight_decay=1e-6,
    )

    history = []
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        teacher_forcing_ratio = (
            0.1
            if epochs == 1
            else 0.9 - 0.8 * (epoch - 1) / (epochs - 1)
        )

        model.train()
        training_losses = []

        for input_sequence, output_sequence in training_loader:
            input_sequence = input_sequence.to(device)
            output_sequence = output_sequence.to(device)

            prediction, target = recursive_batch_prediction(
                model,
                input_sequence,
                output_sequence,
                horizon,
                teacher_forcing_ratio,
            )

            loss = nn.functional.mse_loss(
                prediction,
                target,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            training_losses.append(float(loss))

        training_loss = float(
            np.mean(training_losses)
        )

        validation_loss = evaluate_rollout_loss(
            model,
            validation_loader,
            device,
            horizon,
        )

        history.append(
            {
                "stage": "scheduled_rollout",
                "epoch": epoch,
                "teacher_forcing_ratio": teacher_forcing_ratio,
                "training_normalized_mse": training_loss,
                "validation_normalized_mse": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        print(
            f"Rollout {epoch:02d}/{epochs} | "
            f"measured feedback={teacher_forcing_ratio:.2f} | "
            f"train={training_loss:.7f} | "
            f"validation={validation_loss:.7f}"
        )

    if best_state is None:
        raise RuntimeError("Rollout training failed.")

    model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------
# 8. COMPLETE TEST PREDICTIONS
# ---------------------------------------------------------------------

def predict_one_step(
    model: nn.Module,
    normalized_input: np.ndarray,
    normalized_output: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Measured output history is supplied at every test step."""
    predicted = np.zeros_like(normalized_output)
    predicted[:WINDOW] = normalized_output[:WINDOW]

    model.eval()

    with torch.no_grad():
        for target_index in range(WINDOW, len(normalized_input)):
            start_index = target_index - WINDOW

            input_history = torch.from_numpy(
                normalized_input[
                    start_index:target_index
                ][None]
            ).to(device)

            output_history = torch.from_numpy(
                normalized_output[
                    start_index:target_index
                ][None]
            ).to(device)

            predicted[target_index] = (
                model(
                    input_history,
                    output_history,
                )
                .cpu()
                .numpy()[0]
            )

    return predicted


def predict_parallel(
    model: nn.Module,
    normalized_input: np.ndarray,
    normalized_output: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Only the first WINDOW measured outputs initialize the simulation.
    All later displacement and force histories are predictions.
    """
    predicted = np.zeros_like(normalized_output)
    predicted[:WINDOW] = normalized_output[:WINDOW]

    model.eval()

    with torch.no_grad():
        for target_index in range(WINDOW, len(normalized_input)):
            start_index = target_index - WINDOW

            input_history = torch.from_numpy(
                normalized_input[
                    start_index:target_index
                ][None]
            ).to(device)

            output_history = torch.from_numpy(
                predicted[
                    start_index:target_index
                ][None]
            ).to(device)

            predicted[target_index] = (
                model(
                    input_history,
                    output_history,
                )
                .cpu()
                .numpy()[0]
            )

    return predicted


# ---------------------------------------------------------------------
# 9. METRICS AND FILE HELPERS
# ---------------------------------------------------------------------

def calculate_metrics(
    measured: np.ndarray,
    predicted: np.ndarray,
    evaluation_name: str,
) -> list[dict]:
    rows = []

    for column, output_name, unit in [
        (0, "Displacement", "mm"),
        (1, "Lorentz force", "N"),
    ]:
        actual = measured[:, column]
        estimate = predicted[:, column]
        error = actual - estimate

        mse = float(np.mean(error**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(error)))

        total_variation = float(
            np.sum(
                (actual - actual.mean()) ** 2
            )
        )

        r_squared = (
            1.0
            - float(np.sum(error**2))
            / total_variation
            if total_variation > 0
            else float("nan")
        )

        norm_denominator = float(
            np.linalg.norm(
                actual - actual.mean()
            )
        )

        fit_percent = (
            100.0
            * (
                1.0
                - float(np.linalg.norm(error))
                / norm_denominator
            )
            if norm_denominator > 0
            else float("nan")
        )

        rows.append(
            {
                "dataset": "147mA",
                "evaluation": evaluation_name,
                "output": output_name,
                "unit": unit,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r_squared,
                "fit_percent": fit_percent,
            }
        )

    return rows


def write_dictionary_rows(
    path: Path,
    rows: list[dict],
) -> None:
    columns = []

    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


def create_output_folders() -> tuple[Path, Path, Path]:
    """
    Create one new timestamped run.
    This avoids Windows/OneDrive permission errors.
    """
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_folder = (
        ROOT
        / "01_Report_Figures"
        / timestamp
    )

    presentation_folder = (
        ROOT
        / "02_Presentation_Figures"
        / timestamp
    )

    complete_folder = (
        ROOT
        / "03_Complete_Results"
        / timestamp
    )

    for folder in [
        report_folder,
        presentation_folder,
        complete_folder,
    ]:
        folder.mkdir(parents=True)

    return (
        report_folder,
        presentation_folder,
        complete_folder,
    )


def save_figure(
    figure: plt.Figure,
    filename: str,
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
    for_presentation: bool = False,
) -> None:
    complete_path = complete_folder / filename

    figure.tight_layout()

    figure.savefig(
        complete_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    shutil.copy2(
        complete_path,
        report_folder / filename,
    )

    if for_presentation:
        shutil.copy2(
            complete_path,
            presentation_folder / filename,
        )


# ---------------------------------------------------------------------
# 10. FIGURES
# ---------------------------------------------------------------------

def make_cost_figure(
    history: list[dict],
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    offset = 0

    for stage in [
        "series_parallel",
        "scheduled_rollout",
    ]:
        rows = [
            row
            for row in history
            if row["stage"] == stage
        ]

        if not rows:
            continue

        epoch_axis = (
            np.arange(1, len(rows) + 1)
            + offset
        )

        training_loss = [
            row["training_normalized_mse"]
            for row in rows
        ]

        validation_loss = [
            row["validation_normalized_mse"]
            for row in rows
        ]

        axis.plot(
            epoch_axis,
            training_loss,
            label=f"{stage}: training",
        )

        axis.plot(
            epoch_axis,
            validation_loss,
            linestyle="--",
            label=f"{stage}: validation",
        )

        offset += len(rows)

    axis.set_yscale("log")
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Normalized MSE")
    axis.set_title(
        "Cost function during both training stages"
    )
    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )
    axis.legend()

    save_figure(
        figure,
        "01_cost_function.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )


def make_block_figure(
    blocks: list[Block],
    raw_data: dict[str, np.ndarray],
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    positions = {
        sheet_name: position
        for position, sheet_name
        in enumerate(DEVELOPMENT_SHEETS)
    }

    for role in [
        "training",
        "validation",
    ]:
        first_label = True

        for block in blocks:
            if block.role != role:
                continue

            time_values = raw_data[
                block.sheet
            ][:, 0]

            start_time = float(
                time_values[block.start]
            )

            end_time = float(
                time_values[block.end - 1]
            )

            axis.barh(
                positions[block.sheet],
                end_time - start_time,
                left=start_time,
                height=0.55,
                label=(
                    role.capitalize()
                    if first_label
                    else None
                ),
            )

            first_label = False

    axis.set_yticks(
        list(positions.values())
    )

    axis.set_yticklabels(
        [
            sheet.replace(
                "DC_Offset_",
                "",
            )
            for sheet in DEVELOPMENT_SHEETS
        ]
    )

    axis.set_xlabel(
        "Time within each chirp record (s)"
    )
    axis.set_ylabel(
        "Development dataset"
    )
    axis.set_title(
        "Random non-overlapping training and validation blocks"
    )
    axis.grid(
        True,
        axis="x",
        alpha=0.3,
    )
    axis.legend()

    save_figure(
        figure,
        "02_random_data_blocks.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )


def make_prediction_figures(
    time_values: np.ndarray,
    measured: np.ndarray,
    one_step: np.ndarray,
    parallel: np.ndarray,
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    output_definitions = [
        (
            0,
            "Displacement",
            "mm",
            "displacement",
        ),
        (
            1,
            "Lorentz force",
            "N",
            "force",
        ),
    ]

    for column, name, unit, short_name in output_definitions:
        for mode, prediction, title in [
            (
                "one_step",
                one_step,
                f"147 mA unseen test: {name} one-step prediction",
            ),
            (
                "parallel",
                parallel,
                f"147 mA unseen test: {name} parallel prediction",
            ),
        ]:
            figure, axis = plt.subplots(
                figsize=(12, 5)
            )

            axis.plot(
                time_values,
                measured[:, column],
                label="Measured",
            )

            axis.plot(
                time_values,
                prediction[:, column],
                label="LSTM prediction",
            )

            axis.set_xlabel("Time (s)")
            axis.set_ylabel(
                f"{name} ({unit})"
            )
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.legend()

            save_figure(
                figure,
                f"147mA_{short_name}_{mode}.png",
                report_folder,
                presentation_folder,
                complete_folder,
                for_presentation=True,
            )

        figure, axis = plt.subplots(
            figsize=(12, 4)
        )

        axis.plot(
            time_values,
            measured[:, column]
            - one_step[:, column],
            label="One-step error",
        )

        axis.plot(
            time_values,
            measured[:, column]
            - parallel[:, column],
            label="Parallel error",
        )

        axis.set_xlabel("Time (s)")
        axis.set_ylabel(
            f"{name} error ({unit})"
        )
        axis.set_title(
            f"147 mA unseen test: {name} errors"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()

        save_figure(
            figure,
            f"147mA_{short_name}_errors.png",
            report_folder,
            presentation_folder,
            complete_folder,
        )

        measured_points = measured[
            WINDOW::10,
            column,
        ]

        predicted_points = one_step[
            WINDOW::10,
            column,
        ]

        lower_limit = float(
            min(
                measured_points.min(),
                predicted_points.min(),
            )
        )

        upper_limit = float(
            max(
                measured_points.max(),
                predicted_points.max(),
            )
        )

        figure, axis = plt.subplots(
            figsize=(6, 6)
        )

        axis.scatter(
            measured_points,
            predicted_points,
            s=9,
            alpha=0.45,
        )

        axis.plot(
            [lower_limit, upper_limit],
            [lower_limit, upper_limit],
            linestyle="--",
            label="Perfect prediction",
        )

        axis.set_xlabel(
            f"Measured {name} ({unit})"
        )

        axis.set_ylabel(
            f"Predicted {name} ({unit})"
        )

        axis.set_title(
            f"147 mA one-step {name} accuracy"
        )

        axis.grid(True, alpha=0.3)
        axis.legend()

        save_figure(
            figure,
            f"147mA_{short_name}_parity.png",
            report_folder,
            presentation_folder,
            complete_folder,
        )


def make_accuracy_table(
    metric_rows: list[dict],
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    table_values = []

    for row in metric_rows:
        mode = (
            "One-step"
            if row["evaluation"]
            == "series_parallel_one_step"
            else "Parallel"
        )

        table_values.append(
            [
                mode,
                row["output"],
                f"{row['RMSE']:.6g}",
                f"{row['MAE']:.6g}",
                f"{row['R2']:.5f}",
                f"{row['fit_percent']:.3f}",
            ]
        )

    figure, axis = plt.subplots(
        figsize=(11, 4)
    )

    axis.axis("off")

    table = axis.table(
        cellText=table_values,
        colLabels=[
            "Mode",
            "Output",
            "RMSE",
            "MAE",
            "R²",
            "Fit (%)",
        ],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    axis.set_title(
        "147 mA unseen-test accuracy"
    )

    save_figure(
        figure,
        "03_accuracy_table.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )


# ---------------------------------------------------------------------
# 11. MAIN PROGRAM
# ---------------------------------------------------------------------

def main() -> None:
    set_random_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    maximum_rows = 3000 if QUICK_MODE else None
    block_size = 750 if QUICK_MODE else BLOCK_SIZE
    one_step_epochs = 2 if QUICK_MODE else ONE_STEP_EPOCHS
    rollout_epochs = 1 if QUICK_MODE else ROLLOUT_EPOCHS
    rollout_horizon = 4 if QUICK_MODE else ROLLOUT_HORIZON

    workbook_path = find_workbook()

    print("=" * 72)
    print("Hybrid Hammerstein-LSTM system identification")
    print("=" * 72)
    print(f"Workbook: {workbook_path}")
    print(f"Device: {device}")
    print(
        "Development datasets: "
        "67, 87, 107, and 127 mA"
    )
    print(
        "Independent test dataset: 147 mA"
    )
    print()

    workbook = openpyxl.load_workbook(
        workbook_path,
        read_only=True,
        data_only=True,
    )

    sheet_names = (
        DEVELOPMENT_SHEETS
        + [TEST_SHEET]
    )

    raw_data = {
        sheet_name: load_sheet(
            workbook,
            sheet_name,
            maximum_rows=maximum_rows,
        )
        for sheet_name in sheet_names
    }

    workbook.close()

    development_data = {
        sheet_name: raw_data[sheet_name]
        for sheet_name in DEVELOPMENT_SHEETS
    }

    blocks = create_random_blocks(
        development_data,
        block_size,
    )

    normalizer = create_normalizer(
        development_data,
        blocks,
    )

    input_data, output_data = (
        normalize_all_records(
            raw_data,
            normalizer,
        )
    )

    one_step_training_data = OneStepDataset(
        input_data,
        output_data,
        blocks,
        role="training",
        stride=TRAIN_STRIDE,
    )

    one_step_validation_data = OneStepDataset(
        input_data,
        output_data,
        blocks,
        role="validation",
        stride=VALIDATION_STRIDE,
    )

    rollout_training_data = RolloutDataset(
        input_data,
        output_data,
        blocks,
        role="training",
        stride=ROLLOUT_STRIDE,
        horizon=rollout_horizon,
    )

    rollout_validation_data = RolloutDataset(
        input_data,
        output_data,
        blocks,
        role="validation",
        stride=ROLLOUT_STRIDE,
        horizon=rollout_horizon,
    )

    one_step_training_loader = DataLoader(
        one_step_training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    one_step_validation_loader = DataLoader(
        one_step_validation_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    rollout_training_loader = DataLoader(
        rollout_training_data,
        batch_size=ROLLOUT_BATCH_SIZE,
        shuffle=True,
    )

    rollout_validation_loader = DataLoader(
        rollout_validation_data,
        batch_size=ROLLOUT_BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"One-step training windows: "
        f"{len(one_step_training_data)}"
    )

    print(
        f"One-step validation windows: "
        f"{len(one_step_validation_data)}"
    )

    print(
        f"Rollout training sequences: "
        f"{len(rollout_training_data)}"
    )

    print(
        f"Rollout validation sequences: "
        f"{len(rollout_validation_data)}"
    )

    print()

    model = HammersteinLSTM().to(device)

    training_start = time.perf_counter()

    history = train_series_parallel(
        model,
        one_step_training_loader,
        one_step_validation_loader,
        device,
        one_step_epochs,
    )

    history += train_scheduled_rollout(
        model,
        rollout_training_loader,
        rollout_validation_loader,
        device,
        rollout_epochs,
        rollout_horizon,
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    test_input = input_data[TEST_SHEET]
    test_output = output_data[TEST_SHEET]

    one_step_normalized = predict_one_step(
        model,
        test_input,
        test_output,
        device,
    )

    parallel_normalized = predict_parallel(
        model,
        test_input,
        test_output,
        device,
    )

    measured = measured_outputs(
        raw_data[TEST_SHEET]
    )

    one_step = normalizer.restore_output(
        one_step_normalized
    )

    parallel = normalizer.restore_output(
        parallel_normalized
    )

    metric_rows = calculate_metrics(
        measured[WINDOW:],
        one_step[WINDOW:],
        "series_parallel_one_step",
    )

    metric_rows += calculate_metrics(
        measured[WINDOW:],
        parallel[WINDOW:],
        "parallel_free_running",
    )

    (
        report_folder,
        presentation_folder,
        complete_folder,
    ) = create_output_folders()

    write_dictionary_rows(
        complete_folder
        / "training_history.csv",
        history,
    )

    write_dictionary_rows(
        complete_folder
        / "test_metrics.csv",
        metric_rows,
    )

    block_rows = []

    for block in blocks:
        time_values = raw_data[
            block.sheet
        ][:, 0]

        block_rows.append(
            {
                "dataset": block.sheet,
                "start_sample": block.start,
                "end_sample_exclusive": block.end,
                "start_time_s": float(
                    time_values[block.start]
                ),
                "end_time_s": float(
                    time_values[block.end - 1]
                ),
                "role": block.role,
            }
        )

    write_dictionary_rows(
        complete_folder
        / "random_block_assignment.csv",
        block_rows,
    )

    prediction_table = np.column_stack(
        [
            raw_data[TEST_SHEET][:, 0],
            measured[:, 0],
            one_step[:, 0],
            parallel[:, 0],
            measured[:, 1],
            one_step[:, 1],
            parallel[:, 1],
        ]
    )

    np.savetxt(
        complete_folder
        / "147mA_full_predictions.csv",
        prediction_table,
        delimiter=",",
        comments="",
        header=(
            "time_s,"
            "measured_displacement_mm,"
            "one_step_displacement_mm,"
            "parallel_displacement_mm,"
            "measured_force_N,"
            "one_step_force_N,"
            "parallel_force_N"
        ),
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "window": WINDOW,
            "hidden_size": HIDDEN_SIZE,
            "static_size": STATIC_SIZE,
            "input_mean": normalizer.input_mean,
            "input_std": normalizer.input_std,
            "output_mean": normalizer.output_mean,
            "output_std": normalizer.output_std,
        },
        complete_folder
        / "trained_hammerstein_lstm.pt",
    )

    run_information = {
        "device": str(device),
        "development_sheets": DEVELOPMENT_SHEETS,
        "test_sheet": TEST_SHEET,
        "window": WINDOW,
        "hidden_size": HIDDEN_SIZE,
        "static_size": STATIC_SIZE,
        "one_step_epochs": one_step_epochs,
        "rollout_epochs": rollout_epochs,
        "rollout_horizon": rollout_horizon,
        "training_seconds": training_seconds,
        "quick_mode": QUICK_MODE,
    }

    (
        complete_folder
        / "run_information.json"
    ).write_text(
        json.dumps(
            run_information,
            indent=2,
        ),
        encoding="utf-8",
    )

    make_cost_figure(
        history,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    make_block_figure(
        blocks,
        raw_data,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    make_prediction_figures(
        raw_data[TEST_SHEET][:, 0],
        measured,
        one_step,
        parallel,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    make_accuracy_table(
        metric_rows,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    print()
    print("=" * 72)
    print("Finished successfully.")
    print(
        f"Training time: "
        f"{training_seconds:.2f} s"
    )
    print(f"Report figures: {report_folder}")
    print(
        "Presentation figures: "
        f"{presentation_folder}"
    )
    print(
        f"Complete results: "
        f"{complete_folder}"
    )
    print("=" * 72)

    if os.name == "nt":
        try:
            os.startfile(str(report_folder))
            os.startfile(str(presentation_folder))
            os.startfile(str(complete_folder))
        except OSError:
            pass


if __name__ == "__main__":
    main()
