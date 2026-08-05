"""
Data loading and preparation.

One measured input signal is converted into three features:

    I(k)       = current
    ΔI(k)      = current change
    I_DC       = mean current of the complete experiment

The first four sheets are divided into distributed training and
validation blocks. The fifth sheet is never used here for training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from config import (
    BLOCK_SIZE,
    DOWNSAMPLE_FACTOR,
    DEVELOPMENT_SHEETS,
    FINAL_TEST_SHEET,
    SEED,
    SEQUENCE_LENGTH,
    TEST_STRIDE,
    TRAIN_STRIDE,
    VALIDATION_STRIDE,
)


def find_workbook(folder: Path) -> Path:
    """Find the newest COMSOL workbook beside the code."""

    files = list(folder.glob("COMSOL_07_13_2026*.xlsx"))

    if not files:
        raise FileNotFoundError(
            "Place COMSOL_07_13_2026*.xlsx beside main.py."
        )

    return max(files, key=lambda file: file.stat().st_mtime)


def load_sheet(workbook: Path, sheet_name: str):
    """Load time, current, displacement, and force from one sheet."""

    data = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        header=16,
        usecols="A:D",
        engine="openpyxl",
    )

    data.columns = ["time", "displacement", "current", "force"]

    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    # COMSOL exported repeated time rows; keep the final value.
    data = (
        data.sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )

    time = data["time"].to_numpy(np.float32)
    current = data["current"].to_numpy(np.float32)

    outputs = data[
        ["displacement", "force"]
    ].to_numpy(np.float32)

    # The chirp ends at 31 Hz, but the raw data are sampled at 2000 Hz.
    # Anti-aliased downsampling to 500 Hz preserves the physical signal
    # while giving the LSTM a longer physical memory at lower cost.
    current = resample_poly(
        current,
        up=1,
        down=DOWNSAMPLE_FACTOR,
    ).astype(np.float32)

    outputs = resample_poly(
        outputs,
        up=1,
        down=DOWNSAMPLE_FACTOR,
        axis=0,
    ).astype(np.float32)

    time = np.linspace(
        float(time[0]),
        float(time[-1]),
        num=len(current),
        dtype=np.float32,
    )

    return time, current, outputs


def make_features(current: np.ndarray) -> np.ndarray:
    """
    Convert one measured current signal into three input features.

    Column 0: current I(k)
    Column 1: current change ΔI(k)
    Column 2: experiment operating point I_DC
    """

    delta_current = np.diff(
        current,
        prepend=current[0],
    )

    dc_current = np.full_like(
        current,
        current.mean(),
    )

    return np.column_stack(
        [current, delta_current, dc_current]
    ).astype(np.float32)


def make_blocks(number_of_samples: int):
    """
    Assign every fifth block to validation.

    This places validation blocks throughout the beginning, middle,
    and end of each chirp instead of using one continuous region.
    """

    blocks = []
    number_of_blocks = number_of_samples // BLOCK_SIZE

    for block_number in range(number_of_blocks):
        start = block_number * BLOCK_SIZE
        stop = start + BLOCK_SIZE

        role = (
            "validation"
            if block_number % 5 == 0
            else "training"
        )

        blocks.append(
            (block_number, start, stop, role)
        )

    return blocks


def windows_from_blocks(
    experiment,
    blocks,
    requested_role,
    input_mean,
    input_std,
    output_mean,
    output_std,
):
    """Create LSTM windows that stay inside their assigned blocks."""

    _, current, outputs = experiment

    features = make_features(current)
    features_n = (features - input_mean) / input_std
    outputs_n = (outputs - output_mean) / output_std

    inputs = []
    targets = []

    for _, start, stop, role in blocks:
        if role != requested_role:
            continue

        first_target = start + SEQUENCE_LENGTH - 1

        stride = (
            TRAIN_STRIDE
            if requested_role == "training"
            else VALIDATION_STRIDE
        )

        for target_index in range(
            first_target,
            stop,
            stride,
        ):
            window = features_n[
                target_index - SEQUENCE_LENGTH + 1:
                target_index + 1
            ]

            inputs.append(window)
            targets.append(outputs_n[target_index])

    return (
        torch.tensor(np.asarray(inputs), dtype=torch.float32),
        torch.tensor(np.asarray(targets), dtype=torch.float32),
    )


def final_test_windows(
    experiment,
    input_mean,
    input_std,
    output_mean,
    output_std,
):
    """Create windows from the complete untouched fifth experiment."""

    time, current, outputs = experiment

    features = make_features(current)
    features_n = (features - input_mean) / input_std
    outputs_n = (outputs - output_mean) / output_std

    inputs = []
    targets = []
    target_time = []

    for target_index in range(
        SEQUENCE_LENGTH - 1,
        len(current),
        TEST_STRIDE,
    ):
        window = features_n[
            target_index - SEQUENCE_LENGTH + 1:
            target_index + 1
        ]

        inputs.append(window)
        targets.append(outputs_n[target_index])
        target_time.append(time[target_index])

    return (
        torch.tensor(np.asarray(inputs), dtype=torch.float32),
        torch.tensor(np.asarray(targets), dtype=torch.float32),
        np.asarray(target_time),
    )


def prepare_data(folder: Path):
    """
    Load all sheets, create distributed train/validation windows, and
    keep the fifth sheet untouched for the pure final test.
    """

    workbook = find_workbook(folder)

    experiments = {
        sheet: load_sheet(workbook, sheet)
        for sheet in DEVELOPMENT_SHEETS + [FINAL_TEST_SHEET]
    }

    blocks = {
        sheet: make_blocks(len(experiments[sheet][0]))
        for sheet in DEVELOPMENT_SHEETS
    }

    # Scaling statistics are calculated from training blocks only.
    train_features = []
    train_outputs = []
    split_rows = []

    for sheet in DEVELOPMENT_SHEETS:
        time, current, outputs = experiments[sheet]
        features = make_features(current)

        for block_number, start, stop, role in blocks[sheet]:
            split_rows.append(
                {
                    "sheet": sheet,
                    "block": block_number,
                    "role": role,
                    "start_sample": start,
                    "stop_sample": stop,
                    "start_time": float(time[start]),
                    "end_time": float(time[stop - 1]),
                }
            )

            if role == "training":
                train_features.append(features[start:stop])
                train_outputs.append(outputs[start:stop])

    train_features = np.concatenate(train_features)
    train_outputs = np.concatenate(train_outputs)

    input_mean = train_features.mean(axis=0)
    input_std = train_features.std(axis=0) + 1e-8

    output_mean = train_outputs.mean(axis=0)
    output_std = train_outputs.std(axis=0) + 1e-8

    x_train_list = []
    y_train_list = []
    x_validation_list = []
    y_validation_list = []

    for sheet in DEVELOPMENT_SHEETS:
        x_train, y_train = windows_from_blocks(
            experiments[sheet],
            blocks[sheet],
            "training",
            input_mean,
            input_std,
            output_mean,
            output_std,
        )

        x_validation, y_validation = windows_from_blocks(
            experiments[sheet],
            blocks[sheet],
            "validation",
            input_mean,
            input_std,
            output_mean,
            output_std,
        )

        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_validation_list.append(x_validation)
        y_validation_list.append(y_validation)

    x_test, y_test, test_time = final_test_windows(
        experiments[FINAL_TEST_SHEET],
        input_mean,
        input_std,
        output_mean,
        output_std,
    )

    return {
        "workbook": workbook,
        "experiments": experiments,
        "split_table": pd.DataFrame(split_rows),
        "x_train": torch.cat(x_train_list),
        "y_train": torch.cat(y_train_list),
        "x_validation": torch.cat(x_validation_list),
        "y_validation": torch.cat(y_validation_list),
        "x_test": x_test,
        "y_test": y_test,
        "test_time": test_time,
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
    }
