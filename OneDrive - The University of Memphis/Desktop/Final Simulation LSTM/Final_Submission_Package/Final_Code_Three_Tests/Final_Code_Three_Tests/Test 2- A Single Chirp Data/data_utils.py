from pathlib import Path


import numpy as np

import pandas as pd

import torch


from scipy.signal import resample_poly


from config import (

    AVAILABLE_SHEETS,

    BLOCK_SIZE,

    DOWNSAMPLE_FACTOR,

    SEQUENCE_LENGTH,

    SINGLE_SERIES_SHEET,

    SPLIT_PATTERN,

    TEST_STRIDE,

    TRAIN_STRIDE,

    VALIDATION_STRIDE,

)


def find_workbook(folder: Path) -> Path:


    files = list(folder.glob("COMSOL_07_13_2026*.xlsx"))

    if not files:


        raise FileNotFoundError(

            "Place COMSOL_07_13_2026*.xlsx beside main.py."

        )


    return max(files, key=lambda file: file.stat().st_mtime)


def load_sheet(workbook: Path, sheet_name: str):


    data = pd.read_excel(

        workbook,

        sheet_name=sheet_name,

        header=16,

        usecols="A:D",

        engine="openpyxl",

    )

    data.columns = ["time", "displacement", "current", "force"]

    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    data = (

        data.sort_values("time")

        .drop_duplicates("time", keep="last")

        .reset_index(drop=True)

    )


    time = data["time"].to_numpy(np.float32)

    current = data["current"].to_numpy(np.float32)

    outputs = data[["displacement", "force"]].to_numpy(np.float32)


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


    delta_current = np.diff(current, prepend=current[0])

    dc_current = np.full_like(current, current.mean())

    return np.column_stack(

        [current, delta_current, dc_current]

    ).astype(np.float32)


def make_distributed_blocks(number_of_samples: int):


    blocks = []

    for block_number, start in enumerate(

        range(0, number_of_samples, BLOCK_SIZE)

    ):

        stop = min(start + BLOCK_SIZE, number_of_samples)

        role = (

            SPLIT_PATTERN[block_number % len(SPLIT_PATTERN)]

            if stop - start >= SEQUENCE_LENGTH

            else "unused_short_remainder"

        )

        blocks.append((block_number, start, stop, role))

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


    time, current, outputs = experiment

    features_n = (make_features(current) - input_mean) / input_std

    outputs_n = (outputs - output_mean) / output_std

    stride_by_role = {

        "training": TRAIN_STRIDE,

        "validation": VALIDATION_STRIDE,

        "test": TEST_STRIDE,

    }


    inputs = []

    targets = []

    target_time = []

    for _, start, stop, role in blocks:

        if role != requested_role:

            continue

        for target_index in range(

            start + SEQUENCE_LENGTH - 1,

            stop,

            stride_by_role[requested_role],

        ):

            inputs.append(

                features_n[

                    target_index - SEQUENCE_LENGTH + 1:

                    target_index + 1

                ]

            )

            targets.append(outputs_n[target_index])

            target_time.append(time[target_index])


    if not inputs:

        raise ValueError(

            f"No {requested_role} windows were produced. "

            "Check BLOCK_SIZE and SEQUENCE_LENGTH."

        )


    return (

        torch.tensor(np.asarray(inputs), dtype=torch.float32),

        torch.tensor(np.asarray(targets), dtype=torch.float32),

        np.asarray(target_time, dtype=np.float32),

    )


def full_record_windows(

    experiment,

    input_mean,

    input_std,

    output_mean,

    output_std,

):


    time, current, outputs = experiment

    features_n = (make_features(current) - input_mean) / input_std

    outputs_n = (outputs - output_mean) / output_std


    inputs = []

    targets = []

    target_time = []


    for target_index in range(

        SEQUENCE_LENGTH - 1,

        len(current),

        TEST_STRIDE,

    ):

        inputs.append(

            features_n[

                target_index - SEQUENCE_LENGTH + 1:

                target_index + 1

            ]

        )

        targets.append(outputs_n[target_index])

        target_time.append(time[target_index])


    return (

        torch.tensor(np.asarray(inputs), dtype=torch.float32),

        torch.tensor(np.asarray(targets), dtype=torch.float32),

        np.asarray(target_time, dtype=np.float32),

    )


def prepare_data(folder: Path):


    if SINGLE_SERIES_SHEET not in AVAILABLE_SHEETS:

        raise ValueError(

            f"Unknown SINGLE_SERIES_SHEET={SINGLE_SERIES_SHEET!r}. "

            f"Choose one of: {AVAILABLE_SHEETS}"

        )


    workbook = find_workbook(folder)

    experiment = load_sheet(workbook, SINGLE_SERIES_SHEET)

    time, current, outputs = experiment

    features = make_features(current)

    blocks = make_distributed_blocks(len(time))


    split_rows = []

    train_features = []

    train_outputs = []

    for block_number, start, stop, role in blocks:

        split_rows.append(

            {

                "sheet": SINGLE_SERIES_SHEET,

                "block": block_number,

                "role": role,

                "start_sample": start,

                "stop_sample": stop,

                "start_time_s": float(time[start]),

                "end_time_s": float(time[stop - 1]),

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


    x_train, y_train, train_time = windows_from_blocks(

        experiment,

        blocks,

        "training",

        input_mean,

        input_std,

        output_mean,

        output_std,

    )

    x_validation, y_validation, validation_time = windows_from_blocks(

        experiment,

        blocks,

        "validation",

        input_mean,

        input_std,

        output_mean,

        output_std,

    )

    x_test, y_test, test_time = windows_from_blocks(

        experiment,

        blocks,

        "test",

        input_mean,

        input_std,

        output_mean,

        output_std,

    )

    x_all, y_all, all_time = full_record_windows(

        experiment,

        input_mean,

        input_std,

        output_mean,

        output_std,

    )


    return {

        "workbook": workbook,

        "experiment": experiment,

        "split_table": pd.DataFrame(split_rows),

        "x_train": x_train,

        "y_train": y_train,

        "train_time": train_time,

        "x_validation": x_validation,

        "y_validation": y_validation,

        "validation_time": validation_time,

        "x_test": x_test,

        "y_test": y_test,

        "test_time": test_time,

        "x_all": x_all,

        "y_all": y_all,

        "all_time": all_time,

        "input_mean": input_mean,

        "input_std": input_std,

        "output_mean": output_mean,

        "output_std": output_std,

    }
