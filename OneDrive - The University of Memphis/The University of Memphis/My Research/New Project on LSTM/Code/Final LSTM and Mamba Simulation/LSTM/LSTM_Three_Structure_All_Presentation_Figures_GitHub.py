"""
THREE-STRUCTURE LSTM SYSTEM IDENTIFICATION
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

This updated version generates and verifies every simulation figure referenced
by the current Overleaf presentation.
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

import copy
import csv
import json
import os
import random
import re
import shutil
import tempfile
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

SERIES_EPOCHS = 15
SERIES_PARALLEL_EPOCHS = 15
PARALLEL_EPOCHS = 8
# Longer horizon improves the match between training and free-running evaluation.
# Keep this modest if CPU training is slow, but do not interpret long rollouts
# unless the model remains physically stable.
PARALLEL_HORIZON = 200

BATCH_SIZE = 256
ROLLOUT_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ROLLOUT_LEARNING_RATE = 1e-4

TRAIN_COLOR = "#1f77b4"
VALIDATION_COLOR = "#ff7f0e"
DEVELOPMENT_TEST_COLOR = "#9467bd"
TEST_COLOR = "#2ca02c"
GAP_COLOR = "#b0b0b0"

# Cyclic blocked split for the four development chirps.
# This follows the idea: train block -> validation block -> test block -> repeat.
# Guard gaps are unused samples between roles to reduce leakage from overlapping LSTM windows.
CYCLIC_TRAIN_SAMPLES = 1000
CYCLIC_VALIDATION_SAMPLES = 500
CYCLIC_DEVELOPMENT_TEST_SAMPLES = 500
GUARD_GAP_SAMPLES = WINDOW

BLOCK_SIZE = 1000
VALIDATION_FRACTION = 0.20
TRAIN_STRIDE = 10
VALIDATION_STRIDE = 5
ROLLOUT_STRIDE = 100

RANDOM_SEED = 123

# Automatic GitHub push after a completely successful simulation run.
# Set this to False whenever you want to run the code without pushing.
AUTO_GIT_PUSH = True  # automatically push code and results after a successful run
GIT_REPOSITORY_SSH = "git@github.com:hzolfaghari2022/LSTM_Modelling.git"
GIT_USER_NAME = "Hussein Zolfaghari"
GIT_USER_EMAIL = "h.zolfaghari2015@gmail.com"

DEVELOPMENT_SHEETS = [
    "DC_Offset_67mA",
    "DC_Offset_87mA",
    "DC_Offset_107mA",
    "DC_Offset_127mA",
]
TEST_SHEET = "DC_Offset_147mA"

# Exact filenames referenced by the current Overleaf presentation.
# The program verifies these files at the end of every successful run.
PRESENTATION_REQUIRED_FIGURES = [
    "00_cyclic_split_schematic.png",
    "01_all_structures_cost.png",
    "02_data_usage_summary.png",
    "04_displacement_errors_valid_models.png",
    "04_force_errors_valid_models.png",
    "05_revised_accuracy_table.png",
    "06_series_measured_vs_predicted.png",
    "06_series_parallel_measured_vs_predicted.png",
    "08_regression_valid_models_with_baseline.png",
    "09_parallel_rollout_instability_diagnostic.png",
]

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



def open_workbook_safely(workbook_path: Path) -> openpyxl.Workbook:
    """
    Open the Excel workbook safely on Windows and OneDrive.

    First, the code tries to make a local temporary copy. Reading the local
    copy prevents OneDrive synchronization from interrupting openpyxl.

    If Windows has locked the original workbook, close it in Excel and close
    the File Explorer Preview pane. The code then waits for Enter and retries.
    """
    temporary_folder = Path(tempfile.gettempdir()) / "LSTM_COMSOL_Data"
    temporary_folder.mkdir(parents=True, exist_ok=True)

    temporary_copy = (
        temporary_folder
        / f"COMSOL_copy_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    )

    maximum_attempts = 3

    for attempt in range(1, maximum_attempts + 1):
        try:
            shutil.copyfile(workbook_path, temporary_copy)

            print(
                "Using a local temporary workbook copy to avoid "
                "OneDrive/Excel file locking:"
            )
            print(temporary_copy)
            print()

            return openpyxl.load_workbook(
                temporary_copy,
                read_only=True,
                data_only=True,
            )

        except PermissionError:
            print()
            print("=" * 76)
            print("Windows cannot read the Excel workbook because it is locked.")
            print()
            print("Please do these steps:")
            print("1. Close COMSOL_07_13_2026.xlsx in Excel.")
            print("2. Close any Excel preview of the file.")
            print("3. In File Explorer, turn off the Preview pane if it is open.")
            print("4. Wait a few seconds for OneDrive to finish syncing.")
            print()
            print(
                f"Attempt {attempt} of {maximum_attempts} could not access:"
            )
            print(workbook_path)
            print("=" * 76)

            if attempt < maximum_attempts:
                input(
                    "After closing the workbook, press Enter here to retry..."
                )
            else:
                raise PermissionError(
                    "The Excel workbook is still locked. "
                    "Close Excel or copy the workbook and this Python file "
                    "to a normal local folder outside OneDrive, then run again."
                )

        except OSError as error:
            raise OSError(
                f"Could not copy or open the workbook: {error}"
            ) from error

    raise RuntimeError("The workbook could not be opened.")


def load_sheet(
    workbook: openpyxl.Workbook,
    sheet_name: str,
    maximum_rows: int | None = None,
) -> np.ndarray:
    """
    Read the first four numeric columns:
        time, displacement, coil current, Lorentz force.

    Important corrections in the revised version:
        1. duplicate time rows are removed;
        2. the record is sorted by time if needed;
        3. the time step is checked for consistency.
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

    if len(data) <= WINDOW + PARALLEL_HORIZON:
        raise ValueError(f"Not enough numeric rows in sheet {sheet_name}")

    # Remove duplicate time samples. COMSOL/Excel exports can contain repeated
    # rows, which make the sequence nonuniform and create artificial zero ΔI.
    original_length = len(data)
    _, unique_indices = np.unique(data[:, 0], return_index=True)
    unique_indices = np.sort(unique_indices)
    data = data[unique_indices]

    if len(data) < original_length:
        print(
            f"{sheet_name}: removed {original_length - len(data)} "
            "duplicate time rows."
        )

    # Ensure increasing time order.
    order = np.argsort(data[:, 0])
    if not np.all(order == np.arange(len(data))):
        data = data[order]
        print(f"{sheet_name}: sorted rows by time.")

    dt = np.diff(data[:, 0])
    median_dt = float(np.median(dt))
    if not np.allclose(dt, median_dt, rtol=1e-3, atol=1e-9):
        print(
            f"Warning: {sheet_name} has a nonuniform time step. "
            f"median dt = {median_dt:.6g} s."
        )

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
    maximum_horizon: int,
) -> list[Block]:
    """
    Create a cyclic blocked train/validation/development-test split.

    The chirp frequency changes continuously with time. A single contiguous
    split, for example first 70% train and last 30% test, can make one split
    see mostly low-frequency behavior and another split see mostly
    high-frequency behavior. This cyclic split repeats short contiguous
    blocks across the full chirp so that train, validation, and internal
    development-test blocks all sample different frequency regions.

    Guard gaps are inserted between roles. These samples are not used for
    training or validation, which reduces leakage from overlapping LSTM
    windows crossing a role boundary.

    The complete 147 mA record is still kept as the final independent test.
    """
    blocks: list[Block] = []

    cycle = [
        ("training", CYCLIC_TRAIN_SAMPLES),
        ("gap", GUARD_GAP_SAMPLES),
        ("validation", CYCLIC_VALIDATION_SAMPLES),
        ("gap", GUARD_GAP_SAMPLES),
        ("development_test", CYCLIC_DEVELOPMENT_TEST_SAMPLES),
        ("gap", GUARD_GAP_SAMPLES),
    ]

    minimum_useful_length = WINDOW + maximum_horizon + 1

    for sheet_name in DEVELOPMENT_SHEETS:
        number_of_samples = len(development_data[sheet_name])
        start = 0

        while start < number_of_samples:
            for role, length in cycle:
                end = min(start + length, number_of_samples)

                if end <= start:
                    break

                if role == "gap":
                    blocks.append(Block(sheet_name, start, end, role))
                elif end - start >= minimum_useful_length:
                    blocks.append(Block(sheet_name, start, end, role))

                start = end

                if number_of_samples - start < minimum_useful_length:
                    if start < number_of_samples:
                        blocks.append(Block(sheet_name, start, number_of_samples, "gap"))
                    start = number_of_samples
                    break

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


class SeriesLSTM(nn.Module):
    """
    Series / input-only model:
        [current, current change, DC offset]_past
        -> [displacement, force]_next
    """

    def __init__(self) -> None:
        super().__init__()

        self.static_nonlinearity = nn.Sequential(
            nn.Linear(3, STATIC_SIZE),
            nn.Tanh(),
            nn.Linear(STATIC_SIZE, STATIC_SIZE),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=STATIC_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, input_history: torch.Tensor) -> torch.Tensor:
        nonlinear_input = self.static_nonlinearity(input_history)
        recurrent_output, _ = self.lstm(nonlinear_input)
        return self.output_layer(recurrent_output[:, -1])


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


def evaluate_series_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for input_history, _, target in data_loader:
            input_history = input_history.to(device)
            target = target.to(device)

            prediction = model(input_history)
            loss = nn.functional.mse_loss(prediction, target)

            total_loss += float(loss) * len(target)
            total_samples += len(target)

    return total_loss / total_samples


def train_series(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> list[dict]:
    """Train the input-only series structure."""
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

        for input_history, _, target in training_loader:
            input_history = input_history.to(device)
            target = target.to(device)

            prediction = model(input_history)
            loss = nn.functional.mse_loss(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss) * len(target)
            total_samples += len(target)

        training_loss = total_loss / total_samples
        validation_loss = evaluate_series_loss(
            model,
            validation_loader,
            device,
        )

        history.append(
            {
                "structure": "series",
                "stage": "series",
                "epoch": epoch,
                "teacher_forcing_ratio": 0.0,
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
            f"Series {epoch:02d}/{epochs} | "
            f"train={training_loss:.7f} | "
            f"validation={validation_loss:.7f}"
        )

    if best_state is None:
        raise RuntimeError("Series training failed.")

    model.load_state_dict(best_state)
    return history


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
                "structure": "series_parallel",
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
            0.0
            if epochs == 1
            else max(0.0, 1.0 - epoch / epochs)
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
                "structure": "parallel",
                "stage": "parallel",
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


def predict_series(
    model: nn.Module,
    normalized_input: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Input-only prediction over the complete test record."""
    predicted = np.zeros((len(normalized_input), 2), dtype=np.float32)

    windows = np.asarray(
        [
            normalized_input[index - WINDOW:index]
            for index in range(WINDOW, len(normalized_input))
        ],
        dtype=np.float32,
    )

    model.eval()

    with torch.no_grad():
        for start in range(0, len(windows), 1024):
            end = min(start + 1024, len(windows))
            batch = torch.from_numpy(windows[start:end]).to(device)
            predicted[WINDOW + start:WINDOW + end] = (
                model(batch).cpu().numpy()
            )

    return predicted


def predict_one_step(
    model: nn.Module,
    normalized_input: np.ndarray,
    normalized_output: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Measured output history is supplied at every test step."""
    predicted = np.zeros_like(normalized_output)
    predicted[:WINDOW] = normalized_output[:WINDOW]

    input_windows = np.asarray(
        [
            normalized_input[index - WINDOW:index]
            for index in range(WINDOW, len(normalized_input))
        ],
        dtype=np.float32,
    )
    output_windows = np.asarray(
        [
            normalized_output[index - WINDOW:index]
            for index in range(WINDOW, len(normalized_output))
        ],
        dtype=np.float32,
    )

    model.eval()

    with torch.no_grad():
        for start in range(0, len(input_windows), 1024):
            end = min(start + 1024, len(input_windows))
            input_batch = torch.from_numpy(input_windows[start:end]).to(device)
            output_batch = torch.from_numpy(output_windows[start:end]).to(device)
            predicted[WINDOW + start:WINDOW + end] = (
                model(input_batch, output_batch).cpu().numpy()
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



def make_persistence_baseline(measured: np.ndarray) -> np.ndarray:
    """Simple baseline: predict the next output as the previous measured output."""
    baseline = measured.copy()
    baseline[WINDOW:] = measured[WINDOW - 1:-1]
    return baseline


def prediction_status(measured: np.ndarray, predicted: np.ndarray) -> str:
    """
    Flag physically meaningless predictions before plotting/reporting.
    The bound is deliberately generous: 10 times the measured range or 30 mm
    for displacement-like outputs. This prevents one diverged recursive model
    from destroying figure axes and table interpretation.
    """
    measured_eval = measured[WINDOW:]
    predicted_eval = predicted[WINDOW:]
    measured_range = float(np.nanmax(measured_eval) - np.nanmin(measured_eval))
    limit = max(30.0, 10.0 * measured_range)
    max_abs_prediction = float(np.nanmax(np.abs(predicted_eval)))
    if not np.isfinite(max_abs_prediction) or max_abs_prediction > limit:
        return "invalid_diverged"
    return "valid"


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

def make_cyclic_split_explanation(
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    """Create a simple schematic of the cyclic blocked split."""
    figure, axis = plt.subplots(figsize=(12, 2.2))

    schematic = [
        ("Train", 10, TRAIN_COLOR),
        ("Gap", 1.2, GAP_COLOR),
        ("Validation", 5, VALIDATION_COLOR),
        ("Gap", 1.2, GAP_COLOR),
        ("Dev. test", 5, DEVELOPMENT_TEST_COLOR),
        ("Gap", 1.2, GAP_COLOR),
        ("Train", 10, TRAIN_COLOR),
        ("Gap", 1.2, GAP_COLOR),
        ("Validation", 5, VALIDATION_COLOR),
        ("Gap", 1.2, GAP_COLOR),
        ("Dev. test", 5, DEVELOPMENT_TEST_COLOR),
    ]

    left = 0.0
    for label, width, color in schematic:
        alpha = 0.45 if label == "Gap" else 1.0
        axis.barh(0, width, left=left, height=0.6, color=color, alpha=alpha)
        axis.text(
            left + width / 2,
            0,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )
        left += width

    axis.set_xlim(0, left)
    axis.set_ylim(-0.6, 0.6)
    axis.set_yticks([])
    axis.set_xlabel("Time / increasing chirp frequency")
    axis.set_title("Cyclic blocked split repeated across each chirp record")
    axis.grid(True, axis="x", alpha=0.25)

    save_figure(
        figure,
        "00_cyclic_split_schematic.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )


def make_cost_figure(
    history: list[dict],
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))

    for structure in ["series", "series_parallel", "parallel"]:
        rows = [
            row for row in history
            if row.get("structure") == structure
        ]

        if not rows:
            continue

        epochs = np.arange(1, len(rows) + 1)

        axis.plot(
            epochs,
            [row["training_normalized_mse"] for row in rows],
            label=f"{structure}: training",
        )

        axis.plot(
            epochs,
            [row["validation_normalized_mse"] for row in rows],
            linestyle="--",
            label=f"{structure}: validation",
        )

    axis.set_yscale("log")
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Normalized MSE")
    axis.set_title("Cost comparison of series, series-parallel, and parallel models")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    save_figure(
        figure,
        "01_all_structures_cost.png",
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
    figure, axis = plt.subplots(figsize=(12, 5))

    all_sheets = DEVELOPMENT_SHEETS + [TEST_SHEET]
    positions = {
        sheet_name: position
        for position, sheet_name in enumerate(all_sheets)
    }

    role_styles = {
        "training": (TRAIN_COLOR, "Training"),
        "validation": (VALIDATION_COLOR, "Validation"),
        "development_test": (DEVELOPMENT_TEST_COLOR, "Development test"),
        "gap": (GAP_COLOR, "Guard gap"),
    }
    shown_labels: set[str] = set()

    for block in blocks:
        time_values = raw_data[block.sheet][:, 0]
        start_time = float(time_values[block.start])
        end_time = float(time_values[block.end - 1])

        color, role_label = role_styles.get(block.role, (GAP_COLOR, block.role))
        label = None if role_label in shown_labels else role_label
        shown_labels.add(role_label)

        axis.barh(
            positions[block.sheet],
            end_time - start_time,
            left=start_time,
            height=0.55,
            color=color,
            alpha=0.45 if block.role == "gap" else 1.0,
            label=label,
        )

    test_time = raw_data[TEST_SHEET][:, 0]
    axis.barh(
        positions[TEST_SHEET],
        float(test_time[-1] - test_time[0]),
        left=float(test_time[0]),
        height=0.55,
        color=TEST_COLOR,
        label="Independent test",
    )

    axis.set_yticks(list(positions.values()))
    axis.set_yticklabels(
        [sheet.replace("DC_Offset_", "") for sheet in all_sheets]
    )
    axis.set_xlabel("Time within each chirp record (s)")
    axis.set_ylabel("Dataset")
    axis.set_title("Cyclic blocked train, validation, development-test, and final-test regions")
    axis.grid(True, axis="x", alpha=0.3)
    axis.legend()

    save_figure(
        figure,
        "02_data_usage_summary.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )


def make_colored_data_figures(
    blocks: list[Block],
    raw_data: dict[str, np.ndarray],
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    """Color the actual current records by training/validation/test role."""
    for sheet_name in DEVELOPMENT_SHEETS + [TEST_SHEET]:
        raw = raw_data[sheet_name]
        time_values = raw[:, 0]
        current = raw[:, 2]

        figure, axis = plt.subplots(figsize=(12, 4.5))

        if sheet_name == TEST_SHEET:
            axis.plot(
                time_values,
                current,
                color=TEST_COLOR,
                label="Independent test",
            )
        else:
            role_styles = {
                "training": (TRAIN_COLOR, "Training", 1.0),
                "validation": (VALIDATION_COLOR, "Validation", 1.0),
                "development_test": (DEVELOPMENT_TEST_COLOR, "Development test", 1.0),
                "gap": (GAP_COLOR, "Guard gap", 0.45),
            }
            shown_labels: set[str] = set()

            for block in blocks:
                if block.sheet != sheet_name:
                    continue

                color, role_label, alpha = role_styles.get(
                    block.role, (GAP_COLOR, block.role, 0.45)
                )
                label = None if role_label in shown_labels else role_label
                shown_labels.add(role_label)

                axis.plot(
                    time_values[block.start:block.end],
                    current[block.start:block.end],
                    color=color,
                    alpha=alpha,
                    label=label,
                )

        label_name = sheet_name.replace("DC_Offset_", "")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Coil current (A)")
        axis.set_title(
            f"{label_name}: cyclic split across the changing-frequency chirp"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()

        save_figure(
            figure,
            f"data_usage_{label_name}_current.png",
            report_folder,
            presentation_folder,
            complete_folder,
            for_presentation=True,
        )

def make_prediction_figures(
    time_values: np.ndarray,
    measured: np.ndarray,
    series: np.ndarray,
    series_parallel: np.ndarray,
    parallel: np.ndarray,
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    outputs = [
        (0, "Displacement", "mm", "displacement"),
        (1, "Lorentz force", "N", "force"),
    ]

    comparison_data = {
        "Measured": measured,
        "Series": series,
        "Series-parallel": series_parallel,
        "Parallel": parallel,
    }

    zoom_start = max(WINDOW, int(0.65 * len(time_values)))
    t_full = time_values[WINDOW:]
    t_zoom = time_values[zoom_start:]

    for column, name, unit, short_name in outputs:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))

        for label, values in comparison_data.items():
            axes[0].plot(
                t_full,
                values[WINDOW:, column],
                label=label,
                linewidth=2 if label == "Measured" else 1.5,
            )
            axes[1].plot(
                t_zoom,
                values[zoom_start:, column],
                label=label,
                linewidth=2 if label == "Measured" else 1.5,
            )

        axes[0].set_title("Full unseen-test record")
        axes[1].set_title("Zoom on the high-frequency region")

        for axis in axes:
            axis.set_xlabel("Time (s)")
            axis.set_ylabel(f"{name} ({unit})")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8)

        figure.suptitle(
            f"147 mA unseen test: {name.lower()} comparison across all structures",
            fontsize=13,
        )

        save_figure(
            figure,
            f"03_{short_name}_all_structures.png",
            report_folder,
            presentation_folder,
            complete_folder,
            for_presentation=True,
        )

        figure, axis = plt.subplots(figsize=(12, 4.5))

        axis.plot(
            t_full,
            measured[WINDOW:, column] - series[WINDOW:, column],
            label="Series error",
        )
        axis.plot(
            t_full,
            measured[WINDOW:, column] - series_parallel[WINDOW:, column],
            label="Series-parallel error",
        )
        axis.plot(
            t_full,
            measured[WINDOW:, column] - parallel[WINDOW:, column],
            label="Parallel error",
        )

        axis.set_xlabel("Time (s)")
        axis.set_ylabel(f"{name} error ({unit})")
        axis.set_title(f"147 mA unseen test: {name.lower()} error comparison")
        axis.grid(True, alpha=0.3)
        axis.legend()

        save_figure(
            figure,
            f"04_{short_name}_errors_all_structures.png",
            report_folder,
            presentation_folder,
            complete_folder,
            for_presentation=True,
        )


def _regression_statistics(
    measured_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[float, float, float]:
    """Return slope, intercept, and regression R-squared."""
    mask = (
        np.isfinite(measured_values)
        & np.isfinite(predicted_values)
    )

    x = measured_values[mask]
    y = predicted_values[mask]

    if len(x) < 2 or float(np.std(x)) < 1e-14:
        return float("nan"), float("nan"), float("nan")

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept

    denominator = float(
        np.sum((y - np.mean(y)) ** 2)
    )

    r_squared = (
        1.0
        - float(np.sum((y - fitted) ** 2))
        / denominator
        if denominator > 0
        else float("nan")
    )

    return float(slope), float(intercept), float(r_squared)


def make_individual_measured_prediction_figures(
    time_values: np.ndarray,
    measured: np.ndarray,
    predictions: dict[str, np.ndarray],
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    """
    Create one clear measured-versus-predicted figure for each structure.

    Each figure contains:
        1. complete displacement tracking;
        2. complete force tracking;
        3. high-frequency displacement zoom;
        4. high-frequency force zoom.
    """
    evaluation_start = WINDOW
    t = time_values[evaluation_start:]
    actual = measured[evaluation_start:]

    # The final 25 percent of the chirp contains the highest frequencies.
    zoom_start = max(
        evaluation_start,
        int(0.75 * len(time_values)),
    )

    t_zoom = time_values[zoom_start:]
    actual_zoom = measured[zoom_start:]

    for structure_name, prediction in predictions.items():
        estimate = prediction[evaluation_start:]
        estimate_zoom = prediction[zoom_start:]

        figure, axes = plt.subplots(
            2,
            2,
            figsize=(15, 8),
        )

        axes[0, 0].plot(
            t,
            actual[:, 0],
            label="Measured displacement",
            linewidth=1.8,
        )
        axes[0, 0].plot(
            t,
            estimate[:, 0],
            "--",
            label=f"{structure_name} prediction",
            linewidth=1.3,
        )
        axes[0, 0].set_title("Complete displacement record")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Displacement (mm)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[1, 0].plot(
            t,
            actual[:, 1],
            label="Measured Lorentz force",
            linewidth=1.8,
        )
        axes[1, 0].plot(
            t,
            estimate[:, 1],
            "--",
            label=f"{structure_name} prediction",
            linewidth=1.3,
        )
        axes[1, 0].set_title("Complete force record")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Lorentz force (N)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        axes[0, 1].plot(
            t_zoom,
            actual_zoom[:, 0],
            label="Measured displacement",
            linewidth=1.8,
        )
        axes[0, 1].plot(
            t_zoom,
            estimate_zoom[:, 0],
            "--",
            label=f"{structure_name} prediction",
            linewidth=1.3,
        )
        axes[0, 1].set_title(
            "High-frequency displacement zoom"
        )
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Displacement (mm)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[1, 1].plot(
            t_zoom,
            actual_zoom[:, 1],
            label="Measured Lorentz force",
            linewidth=1.8,
        )
        axes[1, 1].plot(
            t_zoom,
            estimate_zoom[:, 1],
            "--",
            label=f"{structure_name} prediction",
            linewidth=1.3,
        )
        axes[1, 1].set_title(
            "High-frequency force zoom"
        )
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Lorentz force (N)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        figure.suptitle(
            "147 mA unseen test: "
            f"measured data versus {structure_name} prediction",
            fontsize=14,
        )

        safe_name = (
            structure_name.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        save_figure(
            figure,
            f"06_{safe_name}_measured_vs_predicted.png",
            report_folder,
            presentation_folder,
            complete_folder,
            for_presentation=True,
        )


def make_regression_figures(
    measured: np.ndarray,
    predictions: dict[str, np.ndarray],
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    """
    Robust regression/parity plots.

    Correction relative to the earlier version:
        * the number of columns adapts to the number of structures;
        * a diverged parallel rollout is flagged instead of letting extreme
          values collapse the whole figure;
        * axes for invalid predictions stay on the measured physical scale
          and report the true out-of-range prediction interval in the panel.
    """
    output_definitions = [
        (0, "Displacement", "mm"),
        (1, "Lorentz force", "N"),
    ]

    evaluation_start = WINDOW
    actual = measured[evaluation_start:]
    number_of_points = len(actual)
    scatter_stride = max(1, number_of_points // 5000)

    def plot_panel(axis, actual_values, predicted_values, title, unit):
        mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
        actual_values = actual_values[mask]
        predicted_values = predicted_values[mask]

        measured_lower = float(np.min(actual_values))
        measured_upper = float(np.max(actual_values))
        measured_range = measured_upper - measured_lower
        pad = 0.08 * measured_range if measured_range > 0 else 1.0

        prediction_min = float(np.min(predicted_values))
        prediction_max = float(np.max(predicted_values))
        physical_limit = max(30.0, 10.0 * measured_range)
        invalid = (
            not np.isfinite(prediction_min)
            or not np.isfinite(prediction_max)
            or max(abs(prediction_min), abs(prediction_max)) > physical_limit
        )

        axis.scatter(
            actual_values[::scatter_stride],
            predicted_values[::scatter_stride],
            s=8,
            alpha=0.30,
        )
        axis.plot(
            [measured_lower - pad, measured_upper + pad],
            [measured_lower - pad, measured_upper + pad],
            "--",
            linewidth=1.2,
            label="Perfect prediction",
        )

        if invalid:
            axis.set_xlim(measured_lower - pad, measured_upper + pad)
            axis.set_ylim(measured_lower - pad, measured_upper + pad)
            axis.set_title(f"{title}\ninvalid long rollout")
            axis.text(
                0.03,
                0.97,
                f"Prediction range:\n[{prediction_min:.3g}, {prediction_max:.3g}] {unit}",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.90),
            )
        else:
            lower = float(min(np.min(actual_values), np.min(predicted_values)))
            upper = float(max(np.max(actual_values), np.max(predicted_values)))
            pad2 = 0.05 * (upper - lower) if upper > lower else 1.0
            line_values = np.linspace(lower - pad2, upper + pad2, 200)

            slope, intercept, regression_r2 = _regression_statistics(
                actual_values,
                predicted_values,
            )
            if np.isfinite(slope):
                axis.plot(
                    line_values,
                    slope * line_values + intercept,
                    ":",
                    linewidth=1.5,
                    label="Regression",
                )
            axis.set_xlim(lower - pad2, upper + pad2)
            axis.set_ylim(lower - pad2, upper + pad2)
            axis.set_title(f"{title}\nR² = {regression_r2:.5f}")

        axis.set_xlabel(f"Measured ({unit})")
        axis.set_ylabel(f"Predicted ({unit})")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)

    # Combined comparison figure.
    structure_items = list(predictions.items())
    number_of_structures = len(structure_items)
    figure, axes = plt.subplots(
        2,
        number_of_structures,
        figsize=(5.1 * number_of_structures, 8.4),
        squeeze=False,
    )

    for column_index, (structure_name, prediction) in enumerate(structure_items):
        estimate = prediction[evaluation_start:]
        for row_index, (output_column, output_name, unit) in enumerate(output_definitions):
            plot_panel(
                axes[row_index, column_index],
                actual[:, output_column],
                estimate[:, output_column],
                f"{structure_name}: {output_name}",
                unit,
            )

    figure.suptitle(
        "147 mA unseen test: regression comparison with validity flag",
        fontsize=14,
    )

    save_figure(
        figure,
        "08_all_structures_regression_comparison.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )

    # Individual regression figures.
    for structure_name, prediction in structure_items:
        estimate = prediction[evaluation_start:]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.3), squeeze=False)
        for axis, (output_column, output_name, unit) in zip(axes[0], output_definitions):
            plot_panel(
                axis,
                actual[:, output_column],
                estimate[:, output_column],
                f"{structure_name}: {output_name}",
                unit,
            )
        figure.suptitle(
            f"147 mA unseen test: {structure_name} regression analysis",
            fontsize=14,
        )
        safe_name = (
            structure_name.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        save_figure(
            figure,
            f"07_{safe_name}_regression.png",
            report_folder,
            presentation_folder,
            complete_folder,
            for_presentation=True,
        )


def make_accuracy_table(
    metric_rows: list[dict],
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    table_values = []

    for row in metric_rows:
        table_values.append(
            [
                row["evaluation"],
                row["output"],
                f"{row['RMSE']:.6g}",
                f"{row['MAE']:.6g}",
                f"{row['R2']:.5f}",
                f"{row['fit_percent']:.3f}",
            ]
        )

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.axis("off")

    table = axis.table(
        cellText=table_values,
        colLabels=["Structure", "Output", "RMSE", "MAE", "R²", "Fit (%)"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    axis.set_title("147 mA unseen-test comparison of all three structures")

    save_figure(
        figure,
        "05_all_structures_accuracy_table.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )




def make_valid_error_figures(
    time_values: np.ndarray,
    measured: np.ndarray,
    series: np.ndarray,
    series_parallel: np.ndarray,
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    """Create the two valid-model error figures used by the presentation.

    The long parallel rollout is intentionally excluded because a diverged
    recursive result must not determine the physical error-axis limits.
    """
    definitions = [
        (0, "Displacement", "mm", "displacement"),
        (1, "Lorentz force", "N", "force"),
    ]
    t = time_values[WINDOW:]

    for column, output_name, unit, short_name in definitions:
        figure, axis = plt.subplots(figsize=(11.5, 4.8))
        axis.plot(
            t,
            measured[WINDOW:, column] - series[WINDOW:, column],
            label="Series error",
            linewidth=1.2,
        )
        axis.plot(
            t,
            measured[WINDOW:, column] - series_parallel[WINDOW:, column],
            label="Series--parallel error",
            linewidth=1.2,
        )
        axis.axhline(0.0, linewidth=0.9, linestyle="--", alpha=0.7)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(f"{output_name} error ({unit})")
        axis.set_title(
            f"147 mA unseen test: valid one-step {output_name.lower()} errors"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()

        save_figure(
            figure,
            f"04_{short_name}_errors_valid_models.png",
            report_folder,
            presentation_folder,
            complete_folder,
            for_presentation=True,
        )


def make_regression_valid_models_with_baseline(
    measured: np.ndarray,
    persistence: np.ndarray,
    series: np.ndarray,
    series_parallel: np.ndarray,
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    """Create the exact regression figure referenced by the presentation."""
    models = [
        ("Persistence baseline", persistence),
        ("Series", series),
        ("Series--parallel", series_parallel),
    ]
    outputs = [
        (0, "Displacement", "mm"),
        (1, "Lorentz force", "N"),
    ]

    actual = measured[WINDOW:]
    scatter_stride = max(1, len(actual) // 4500)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.4), squeeze=False)

    for column_index, (model_name, prediction) in enumerate(models):
        estimate = prediction[WINDOW:]

        for row_index, (output_column, output_name, unit) in enumerate(outputs):
            axis = axes[row_index, column_index]
            x_all = actual[:, output_column]
            y_all = estimate[:, output_column]
            mask = np.isfinite(x_all) & np.isfinite(y_all)
            x_all = x_all[mask]
            y_all = y_all[mask]

            x = x_all[::scatter_stride]
            y = y_all[::scatter_stride]
            slope, intercept, regression_r2 = _regression_statistics(x_all, y_all)

            lower = float(min(np.min(x_all), np.min(y_all)))
            upper = float(max(np.max(x_all), np.max(y_all)))
            padding = 0.05 * (upper - lower) if upper > lower else 1.0
            line_values = np.linspace(lower - padding, upper + padding, 200)

            axis.scatter(x, y, s=8, alpha=0.30, label="Test samples")
            axis.plot(
                line_values,
                line_values,
                "--",
                linewidth=1.2,
                label="Perfect prediction",
            )
            if np.isfinite(slope):
                axis.plot(
                    line_values,
                    slope * line_values + intercept,
                    ":",
                    linewidth=1.4,
                    label="Regression",
                )

            axis.set_xlim(lower - padding, upper + padding)
            axis.set_ylim(lower - padding, upper + padding)
            axis.set_xlabel(f"Measured ({unit})")
            axis.set_ylabel(f"Predicted ({unit})")
            axis.set_title(
                f"{model_name}: {output_name}\nRegression R² = {regression_r2:.5f}"
            )
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=7)
            axis.set_aspect("equal", adjustable="box")

    figure.suptitle(
        "147 mA unseen test: valid regression comparison with persistence baseline",
        fontsize=14,
    )

    save_figure(
        figure,
        "08_regression_valid_models_with_baseline.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )


def make_parallel_rollout_instability_diagnostic(
    time_values: np.ndarray,
    measured: np.ndarray,
    parallel: np.ndarray,
    parallel_is_valid: bool,
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    """Create a readable diagnostic without hiding the actuator-scale data."""
    t = time_values[WINDOW:]
    actual = measured[WINDOW:]
    estimate = parallel[WINDOW:]

    displacement_range = float(np.ptp(actual[:, 0]))
    displacement_margin = max(0.15, 0.10 * displacement_range)
    displacement_lower = float(np.min(actual[:, 0]) - displacement_margin)
    displacement_upper = float(np.max(actual[:, 0]) + displacement_margin)

    displacement_limit = max(30.0, 10.0 * displacement_range)
    displacement_bad = (
        ~np.isfinite(estimate[:, 0])
        | (np.abs(estimate[:, 0]) > displacement_limit)
    )
    first_bad_index = int(np.argmax(displacement_bad)) if np.any(displacement_bad) else None
    first_bad_time = float(t[first_bad_index]) if first_bad_index is not None else None

    figure, axes = plt.subplots(2, 2, figsize=(14, 8.2))

    axes[0, 0].plot(t, actual[:, 0], label="Measured", linewidth=1.6)
    axes[0, 0].plot(t, estimate[:, 0], label="Parallel", linewidth=1.1)
    axes[0, 0].set_title("Displacement: complete recursive rollout")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Displacement (mm)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(t, actual[:, 0], label="Measured", linewidth=1.6)
    axes[0, 1].plot(t, estimate[:, 0], label="Parallel", linewidth=1.1)
    axes[0, 1].set_ylim(displacement_lower, displacement_upper)
    axes[0, 1].set_title("Displacement on the measured physical scale")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Displacement (mm)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(t, actual[:, 1], label="Measured", linewidth=1.6)
    axes[1, 0].plot(t, estimate[:, 1], label="Parallel", linewidth=1.1)
    axes[1, 0].set_title("Lorentz force: complete recursive rollout")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Lorentz force (N)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    displacement_error = np.maximum(np.abs(actual[:, 0] - estimate[:, 0]), 1e-12)
    force_error = np.maximum(np.abs(actual[:, 1] - estimate[:, 1]), 1e-12)
    axes[1, 1].semilogy(t, displacement_error, label="|Displacement error|")
    axes[1, 1].semilogy(t, force_error, label="|Force error|")
    axes[1, 1].set_title("Recursive error growth")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Absolute error (log scale)")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()

    status_text = "stable over the complete record" if parallel_is_valid else "unstable / diverged"
    if first_bad_time is not None:
        status_text += f"; displacement first exceeds diagnostic limit at {first_bad_time:.3f} s"

    figure.suptitle(
        f"147 mA unseen test: parallel rollout diagnostic — {status_text}",
        fontsize=13,
    )

    save_figure(
        figure,
        "09_parallel_rollout_instability_diagnostic.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )


def make_revised_accuracy_table(
    metric_rows: list[dict],
    parallel_is_valid: bool,
    report_folder: Path,
    presentation_folder: Path,
    complete_folder: Path,
) -> None:
    """Create the accuracy table used by the revised presentation.

    If the parallel rollout diverges, its numerical values are not presented as
    physical accuracy metrics; the row is marked as unstable instead.
    """
    table_values = []

    for row in metric_rows:
        evaluation = row["evaluation"]
        output = f"{row['output']} ({row['unit']})"

        if evaluation == "Parallel" and not parallel_is_valid:
            table_values.append(
                [evaluation, output, "Unstable", "--", "--", "--", "--"]
            )
        else:
            table_values.append(
                [
                    evaluation,
                    output,
                    "Valid",
                    f"{row['RMSE']:.6g}",
                    f"{row['MAE']:.6g}",
                    f"{row['R2']:.5f}",
                    f"{row['fit_percent']:.3f}",
                ]
            )

    figure, axis = plt.subplots(figsize=(12.5, 5.8))
    axis.axis("off")
    table = axis.table(
        cellText=table_values,
        colLabels=[
            "Structure",
            "Output",
            "Status",
            "RMSE",
            "MAE",
            "R²",
            "Fit (%)",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.40)
    axis.set_title(
        "147 mA unseen test: valid metrics and parallel-rollout status",
        pad=14,
    )

    save_figure(
        figure,
        "05_revised_accuracy_table.png",
        report_folder,
        presentation_folder,
        complete_folder,
        for_presentation=True,
    )


def verify_required_presentation_figures(presentation_folder: Path) -> None:
    """Fail clearly if any filename referenced by the Overleaf deck is absent."""
    missing = [
        filename
        for filename in PRESENTATION_REQUIRED_FIGURES
        if not (presentation_folder / filename).exists()
    ]

    if missing:
        formatted = "\n".join(f"  - {name}" for name in missing)
        raise RuntimeError(
            "The simulation finished, but these presentation figures are missing:\n"
            f"{formatted}"
        )

    print()
    print("All figures required by the Overleaf presentation were created:")
    for filename in PRESENTATION_REQUIRED_FIGURES:
        print(f"  - {filename}")


# ---------------------------------------------------------------------
# 11. MAIN PROGRAM
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# GITHUB AUTOMATIC PUSH
# ---------------------------------------------------------------------

def run_git(command: str, working_folder: Path, check: bool = True) -> tuple[int, str]:
    """Run one Git command and print its output."""
    print(f"  $ {command}")

    result = subprocess.run(
        command,
        cwd=working_folder,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    output = (result.stdout + result.stderr).strip()

    if output:
        print(output)

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Git command failed:\n{command}\n\n{output}"
        )

    return result.returncode, output


def git_push_to_github(repository_folder: Path) -> None:
    """Commit and push the code and generated results after a successful run."""
    print()
    print("=" * 76)
    print("Automatic GitHub push")
    print(f"Repository: {GIT_REPOSITORY_SSH}")
    print("=" * 76)

    repository_folder = Path(repository_folder).resolve()
    print(f"Git working folder: {repository_folder}")

    run_git(
        "git --version",
        repository_folder,
    )

    # Initialize Git only when this project folder is not already a repository.
    return_code, _ = run_git(
        "git rev-parse --is-inside-work-tree",
        repository_folder,
        check=False,
    )

    if return_code != 0:
        print("This folder is not a Git repository. Initializing Git now...")

        run_git(
            "git init",
            repository_folder,
        )

        run_git(
            "git branch -M main",
            repository_folder,
            check=False,
        )

    # Use the same Git identity and SSH repository as the supplied code.
    run_git(
        f'git config user.name "{GIT_USER_NAME}"',
        repository_folder,
    )

    run_git(
        f'git config user.email "{GIT_USER_EMAIL}"',
        repository_folder,
    )

    remote_code, current_remote = run_git(
        "git remote get-url origin",
        repository_folder,
        check=False,
    )

    if remote_code != 0:
        run_git(
            f"git remote add origin {GIT_REPOSITORY_SSH}",
            repository_folder,
        )

    elif current_remote.strip() != GIT_REPOSITORY_SSH:
        run_git(
            f"git remote set-url origin {GIT_REPOSITORY_SSH}",
            repository_folder,
        )

    else:
        print(f"Remote origin is already correct: {GIT_REPOSITORY_SSH}")

    run_git(
        "git branch -M main",
        repository_folder,
        check=False,
    )

    # Finish an earlier merge only when all conflicts have already been fixed.
    merge_head = repository_folder / ".git" / "MERGE_HEAD"

    if merge_head.exists():
        _, unresolved = run_git(
            "git diff --name-only --diff-filter=U",
            repository_folder,
            check=False,
        )

        if unresolved.strip():
            raise RuntimeError(
                "Git has an unfinished merge with unresolved conflicts. "
                "Resolve the conflicts and run the simulation again."
            )

        run_git(
            "git add .",
            repository_folder,
        )

        merge_code, merge_output = run_git(
            'git commit -m "Complete previous merge before automatic push"',
            repository_folder,
            check=False,
        )

        if (
            merge_code != 0
            and "nothing to commit" not in merge_output.lower()
        ):
            raise RuntimeError(
                "Could not complete the previous Git merge:\n"
                f"{merge_output}"
            )

    # Add this Python code and all result files produced by the run.
    run_git(
        "git add .",
        repository_folder,
    )

    run_git(
        "git status",
        repository_folder,
        check=False,
    )

    # Commit only when there are staged changes.
    diff_code, _ = run_git(
        "git diff --cached --quiet",
        repository_folder,
        check=False,
    )

    if diff_code != 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        commit_message = (
            "Update three-structure LSTM code and results - "
            f"{timestamp}"
        )

        commit_code, commit_output = run_git(
            f'git commit -m "{commit_message}"',
            repository_folder,
            check=False,
        )

        if commit_code != 0:
            raise RuntimeError(
                "Git commit failed:\n"
                f"{commit_output}"
            )

        print(f"Commit completed successfully:\n{commit_message}")

    else:
        print("No new local changes to commit.")

    # Bring GitHub changes into the local main branch before pushing.
    pull_code, pull_output = run_git(
        "git pull origin main --allow-unrelated-histories --no-rebase --no-edit",
        repository_folder,
        check=False,
    )

    if pull_code != 0:
        raise RuntimeError(
            "Git pull failed. GitHub main has changes that need "
            "manual attention.\n\n"
            f"{pull_output}"
        )

    push_code, push_output = run_git(
        "git push -u origin main",
        repository_folder,
        check=False,
    )

    if push_code != 0:
        raise RuntimeError(
            "Git push failed. Check the SSH connection with:\n"
            "ssh -T git@github.com\n\n"
            f"{push_output}"
        )

    print("Files pushed successfully to GitHub main branch.")
    print("=" * 76)


def main() -> None:
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maximum_rows = 3000 if QUICK_MODE else None
    block_size = 750 if QUICK_MODE else BLOCK_SIZE
    series_epochs = 2 if QUICK_MODE else SERIES_EPOCHS
    series_parallel_epochs = 2 if QUICK_MODE else SERIES_PARALLEL_EPOCHS
    parallel_epochs = 1 if QUICK_MODE else PARALLEL_EPOCHS
    parallel_horizon = 4 if QUICK_MODE else PARALLEL_HORIZON

    workbook_path = find_workbook()

    print("=" * 76)
    print("Series, series-parallel, and parallel system identification")
    print("=" * 76)
    print(f"Workbook: {workbook_path}")
    print(f"Device: {device}")
    print("Development: 67, 87, 107, and 127 mA")
    print("Development split: cyclic train -> validation -> development-test blocks")
    print("Independent final test: 147 mA")
    print()

    workbook = open_workbook_safely(workbook_path)

    sheet_names = DEVELOPMENT_SHEETS + [TEST_SHEET]
    raw_data = {
        sheet_name: load_sheet(workbook, sheet_name, maximum_rows)
        for sheet_name in sheet_names
    }
    workbook.close()

    development_data = {
        sheet_name: raw_data[sheet_name]
        for sheet_name in DEVELOPMENT_SHEETS
    }

    blocks = create_random_blocks(development_data, block_size, parallel_horizon)
    normalizer = create_normalizer(development_data, blocks)
    input_data, output_data = normalize_all_records(raw_data, normalizer)

    training_data = OneStepDataset(
        input_data,
        output_data,
        blocks,
        role="training",
        stride=TRAIN_STRIDE,
    )
    validation_data = OneStepDataset(
        input_data,
        output_data,
        blocks,
        role="validation",
        stride=VALIDATION_STRIDE,
    )

    training_loader = DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    rollout_training_data = RolloutDataset(
        input_data,
        output_data,
        blocks,
        role="training",
        stride=ROLLOUT_STRIDE,
        horizon=parallel_horizon,
    )
    rollout_validation_data = RolloutDataset(
        input_data,
        output_data,
        blocks,
        role="validation",
        stride=ROLLOUT_STRIDE,
        horizon=parallel_horizon,
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

    print(f"One-step training windows: {len(training_data)}")
    print(f"One-step validation windows: {len(validation_data)}")
    print(f"Parallel rollout training sequences: {len(rollout_training_data)}")
    print()

    series_model = SeriesLSTM().to(device)
    series_parallel_model = HammersteinLSTM().to(device)

    training_start = time.perf_counter()

    history = train_series(
        series_model,
        training_loader,
        validation_loader,
        device,
        series_epochs,
    )

    history += train_series_parallel(
        series_parallel_model,
        training_loader,
        validation_loader,
        device,
        series_parallel_epochs,
    )

    # The parallel model starts from the series-parallel solution and is
    # then trained recursively with its own predicted output feedback.
    parallel_model = copy.deepcopy(series_parallel_model).to(device)

    history += train_scheduled_rollout(
        parallel_model,
        rollout_training_loader,
        rollout_validation_loader,
        device,
        parallel_epochs,
        parallel_horizon,
    )

    training_seconds = time.perf_counter() - training_start

    test_input = input_data[TEST_SHEET]
    test_output = output_data[TEST_SHEET]

    series_normalized = predict_series(series_model, test_input, device)
    series_parallel_normalized = predict_one_step(
        series_parallel_model,
        test_input,
        test_output,
        device,
    )
    parallel_normalized = predict_parallel(
        parallel_model,
        test_input,
        test_output,
        device,
    )

    measured = measured_outputs(raw_data[TEST_SHEET])
    series_prediction = normalizer.restore_output(series_normalized)
    series_parallel_prediction = normalizer.restore_output(
        series_parallel_normalized
    )
    parallel_prediction = normalizer.restore_output(parallel_normalized)

    # Physical validity check: do not interpret a diverged recursive rollout
    # as a meaningful actuator prediction. Check both outputs.
    parallel_displacement_status = prediction_status(
        measured[:, [0]],
        parallel_prediction[:, [0]],
    )
    parallel_force_status = prediction_status(
        measured[:, [1]],
        parallel_prediction[:, [1]],
    )
    parallel_is_valid = (
        parallel_displacement_status == "valid"
        and parallel_force_status == "valid"
    )

    if not parallel_is_valid:
        print(
            "WARNING: Parallel rollout diverged or left the physical range. "
            "Treat its metrics and plots as diagnostic only."
        )

    persistence_prediction = make_persistence_baseline(measured)

    metric_rows = []
    metric_rows += calculate_metrics(
        measured[WINDOW:],
        persistence_prediction[WINDOW:],
        "Persistence baseline",
    )
    metric_rows += calculate_metrics(
        measured[WINDOW:],
        series_prediction[WINDOW:],
        "Series",
    )
    metric_rows += calculate_metrics(
        measured[WINDOW:],
        series_parallel_prediction[WINDOW:],
        "Series-parallel",
    )
    metric_rows += calculate_metrics(
        measured[WINDOW:],
        parallel_prediction[WINDOW:],
        "Parallel",
    )

    report_folder, presentation_folder, complete_folder = create_output_folders()

    write_dictionary_rows(
        complete_folder / "training_history.csv",
        history,
    )
    write_dictionary_rows(
        complete_folder / "all_structures_metrics.csv",
        metric_rows,
    )

    block_rows = []
    for block in blocks:
        time_values = raw_data[block.sheet][:, 0]
        block_rows.append(
            {
                "dataset": block.sheet,
                "start_sample": block.start,
                "end_sample_exclusive": block.end,
                "start_time_s": float(time_values[block.start]),
                "end_time_s": float(time_values[block.end - 1]),
                "role": block.role,
            }
        )
    write_dictionary_rows(
        complete_folder / "data_split_blocks.csv",
        block_rows,
    )

    prediction_table = np.column_stack(
        [
            raw_data[TEST_SHEET][:, 0],
            measured[:, 0],
            series_prediction[:, 0],
            series_parallel_prediction[:, 0],
            parallel_prediction[:, 0],
            measured[:, 1],
            series_prediction[:, 1],
            series_parallel_prediction[:, 1],
            parallel_prediction[:, 1],
        ]
    )

    np.savetxt(
        complete_folder / "147mA_all_structures_predictions.csv",
        prediction_table,
        delimiter=",",
        comments="",
        header=(
            "time_s,"
            "measured_displacement_mm,"
            "series_displacement_mm,"
            "series_parallel_displacement_mm,"
            "parallel_displacement_mm,"
            "measured_force_N,"
            "series_force_N,"
            "series_parallel_force_N,"
            "parallel_force_N"
        ),
    )

    # Save normalizer parameters for reproducibility.
    np.savez(
        complete_folder / "normalizer_parameters.npz",
        input_mean=normalizer.input_mean,
        input_std=normalizer.input_std,
        output_mean=normalizer.output_mean,
        output_std=normalizer.output_std,
    )

    torch.save(series_model.state_dict(), complete_folder / "series_model.pt")
    torch.save(
        series_parallel_model.state_dict(),
        complete_folder / "series_parallel_model.pt",
    )
    torch.save(parallel_model.state_dict(), complete_folder / "parallel_model.pt")

    run_information = {
        "device": str(device),
        "development_sheets": DEVELOPMENT_SHEETS,
        "test_sheet": TEST_SHEET,
        "window": WINDOW,
        "hidden_size": HIDDEN_SIZE,
        "static_size": STATIC_SIZE,
        "series_epochs": series_epochs,
        "series_parallel_epochs": series_parallel_epochs,
        "parallel_epochs": parallel_epochs,
        "parallel_horizon": parallel_horizon,
        "training_seconds": training_seconds,
        "split_strategy": "cyclic_blocked_train_validation_development_test",
        "cyclic_train_samples": CYCLIC_TRAIN_SAMPLES,
        "cyclic_validation_samples": CYCLIC_VALIDATION_SAMPLES,
        "cyclic_development_test_samples": CYCLIC_DEVELOPMENT_TEST_SAMPLES,
        "guard_gap_samples": GUARD_GAP_SAMPLES,
        "random_block_size_argument_not_used_in_cyclic_mode": block_size,
        "quick_mode": QUICK_MODE,
    }
    (complete_folder / "run_information.json").write_text(
        json.dumps(run_information, indent=2),
        encoding="utf-8",
    )

    make_cyclic_split_explanation(
        report_folder,
        presentation_folder,
        complete_folder,
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
    make_colored_data_figures(
        blocks,
        raw_data,
        report_folder,
        presentation_folder,
        complete_folder,
    )
    make_prediction_figures(
        raw_data[TEST_SHEET][:, 0],
        measured,
        series_prediction,
        series_parallel_prediction,
        parallel_prediction,
        report_folder,
        presentation_folder,
        complete_folder,
    )
    # Clear measured-versus-predicted overlays for each implemented model.
    prediction_dictionary = {
        "Persistence baseline": persistence_prediction,
        "Series": series_prediction,
        "Series-parallel": series_parallel_prediction,
        "Parallel": parallel_prediction,
    }

    make_individual_measured_prediction_figures(
        raw_data[TEST_SHEET][:, 0],
        measured,
        prediction_dictionary,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    # Regression/parity figures were missing from the previous version.
    make_regression_figures(
        measured,
        prediction_dictionary,
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

    # Exact figures referenced by the current Overleaf presentation.
    make_valid_error_figures(
        raw_data[TEST_SHEET][:, 0],
        measured,
        series_prediction,
        series_parallel_prediction,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    make_regression_valid_models_with_baseline(
        measured,
        persistence_prediction,
        series_prediction,
        series_parallel_prediction,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    make_parallel_rollout_instability_diagnostic(
        raw_data[TEST_SHEET][:, 0],
        measured,
        parallel_prediction,
        parallel_is_valid,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    make_revised_accuracy_table(
        metric_rows,
        parallel_is_valid,
        report_folder,
        presentation_folder,
        complete_folder,
    )

    verify_required_presentation_figures(presentation_folder)

    print()
    print("=" * 76)
    print("Finished successfully.")
    print(f"Training time: {training_seconds:.2f} s")
    print(f"Report figures: {report_folder}")
    print(f"Presentation figures: {presentation_folder}")
    print(f"Complete results: {complete_folder}")
    print("=" * 76)

    # Automatically commit and push this code and the generated results.
    # This is reached only after training, evaluation, and figure generation
    # have all completed successfully.
    if AUTO_GIT_PUSH:
        git_push_to_github(ROOT)
    else:
        print("Automatic GitHub push is disabled (AUTO_GIT_PUSH = False).")

    if os.name == "nt":
        try:
            os.startfile(str(report_folder))
            os.startfile(str(presentation_folder))
            os.startfile(str(complete_folder))
        except OSError:
            pass


if __name__ == "__main__":
    main()
