# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Load, split, and window the five chirp records."""

# Import selected names from pathlib instead of importing its complete namespace.
from pathlib import Path

# Import NumPy for numerical arrays, normalization, errors, and metrics.
import numpy as np
# Import pandas for reading Excel/CSV files and building result tables.
import pandas as pd
# Import PyTorch for tensors, the LSTM model, training, and prediction.
import torch
# Import SciPy's filtered resampling function. It changes the sample rate
# without simply throwing samples away, which helps prevent aliasing.
from scipy.signal import resample_poly

# Import only the configuration values used in this file. `from config import`
# means these names can be written directly below, such as `BLOCK_SIZE`, instead
# of writing `config.BLOCK_SIZE` each time. The parentheses let the import span
# several readable lines; they do not call a function.
from config import (
    # Number of downsampled samples in one train/validation/test role block.
    BLOCK_SIZE,
    # Amount by which the original sample rate is reduced.
    DOWNSAMPLE_FACTOR,
    # Name of the chirp sheet that must remain completely untouched for final test.
    FINAL_TEST_SHEET,
    # Name of the chirp divided into distributed training, validation, and test blocks.
    MIXED_SHEET,
    # Number of past time samples supplied to the LSTM for one prediction.
    SEQUENCE_LENGTH,
    # Repeating order used to label blocks as training, validation, or test.
    SPLIT_PATTERN,
    # Sliding-window step used for test examples.
    TEST_STRIDE,
    # Chirp sheets whose complete usable records are assigned to training.
    TRAIN_ONLY_SHEETS,
    # Sliding-window step used for training examples.
    TRAIN_STRIDE,
    # Sliding-window step used for validation examples.
    VALIDATION_STRIDE,
# End the multiline list of imported configuration names.
)


# Find the workbook file. This function only locates a file; `load_sheet()`
# below performs the reading and cleaning.
def find_workbook(folder: Path) -> Path:
    # This quoted line is the function's docstring: built-in documentation that
    # tools such as help(find_workbook) can display.
    """Find the newest COMSOL workbook beside the code."""

    # Search `folder` for .xlsx filenames that begin with COMSOL_07_13_2026.
    # The `*` wildcard accepts any extra characters before `.xlsx`. `glob()`
    # produces matching Path objects, and `list()` collects them in one list.
    files = list(folder.glob("COMSOL_07_13_2026*.xlsx"))
    # An empty list is false in Python, so this block runs when no file matched.
    if not files:
        # Stop immediately with a clear error rather than failing later inside
        # pandas with a less helpful missing-file message.
        raise FileNotFoundError(
            # This string becomes the error message shown to the user.
            "Place COMSOL_07_13_2026*.xlsx beside main.py."
        # End the FileNotFoundError(...) call.
        )
    # If several matching copies exist, choose the most recently modified one.
    # `max` compares the modification times returned by the short lambda
    # function. `return` sends the selected Path back to the caller.
    return max(files, key=lambda file: file.stat().st_mtime)


# Define the load_sheet function; its indented lines form the function body.
def load_sheet(workbook: Path, sheet_name: str):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Load time, current, displacement, and force from one sheet."""

    # Store the prepared data or the table returned by the current read operation.
    data = pd.read_excel(
        # Pass `workbook` as the next value required by the surrounding call or collection.
        workbook,
        # Pass `sheet_name` as the `sheet_name` argument of the surrounding function call.
        sheet_name=sheet_name,
        # Tell pandas which zero-based Excel row contains the exported column names; 16 means Excel row 17.
        header=16,
        # Read only the four required Excel columns A:D to avoid loading unrelated fields.
        usecols="A:D",
        # Use openpyxl as the reader for .xlsx workbooks.
        engine="openpyxl",
    # Close the current function call, tuple, or grouped expression.
    )
    # Use the expression `data.columns = ["time", "displacement", "current", "force"]` as the next part of the surrounding Python statement.
    data.columns = ["time", "displacement", "current", "force"]
    # Store the prepared data or the table returned by the current read operation.
    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    # Keep one value for each time so the sequence is unambiguous.
    data = (
        # Call `data.sort_values`; the following indented continuation lines provide its arguments.
        data.sort_values("time")
        # Continue the previous expression by applying its drop_duplicates operation.
        .drop_duplicates("time", keep="last")
        # Continue the previous expression by applying its reset_index operation.
        .reset_index(drop=True)
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate `data["time"].to_numpy(np.float32)` and store the result in `time` for the following steps.
    time = data["time"].to_numpy(np.float32)
    # Evaluate `data["current"].to_numpy(np.float32)` and store the result in `current` for the following steps.
    current = data["current"].to_numpy(np.float32)
    # Evaluate `data[["displacement", "force"]].to_numpy(np.float32)` and store the result in `outputs` for the following steps.
    outputs = data[["displacement", "force"]].to_numpy(np.float32)

    # Reduce 2000 Hz to 500 Hz after anti-alias filtering.
    current = resample_poly(
        # Pass `current` as the next value required by the surrounding call or collection.
        current,
        # Use an upsampling factor of one because the data are only being downsampled.
        up=1,
        # Use the configured decimation factor to reduce the sample rate with anti-alias filtering.
        down=DOWNSAMPLE_FACTOR,
    # Use the expression `).astype(np.float32)` as the next part of the surrounding Python statement.
    ).astype(np.float32)
    # Evaluate `resample_poly(` and store the result in `outputs` for the following steps.
    outputs = resample_poly(
        # Pass `outputs` as the next value required by the surrounding call or collection.
        outputs,
        # Use an upsampling factor of one because the data are only being downsampled.
        up=1,
        # Use the configured decimation factor to reduce the sample rate with anti-alias filtering.
        down=DOWNSAMPLE_FACTOR,
        # Resample along axis 0, the time/sample dimension; keep both output
        # columns aligned and resampled together.
        axis=0,
    # Use the expression `).astype(np.float32)` as the next part of the surrounding Python statement.
    ).astype(np.float32)
    # Evaluate `np.linspace(` and store the result in `time` for the following steps.
    time = np.linspace(
        # Call `float`; the following indented continuation lines provide its arguments.
        float(time[0]),
        # Call `float`; the following indented continuation lines provide its arguments.
        float(time[-1]),
        # Pass `len(current)` as the `num` argument of the surrounding function call.
        num=len(current),
        # Choose the stored numerical data type; float32 matches PyTorch model precision and halves memory versus float64.
        dtype=np.float32,
    # Close the current function call, tuple, or grouped expression.
    )

    # Return this value to the code that called the current function.
    return time, current, outputs


# Convert measured current into the three inputs used by the LSTM
def make_features(current: np.ndarray) -> np.ndarray:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Return [current, current change, experiment DC operating point]."""

    # Calculate the change in current between consecutive samples.
    delta_current = np.diff(current, prepend=current[0])
    # Repeat the record's mean current so the DC operating point is available at every time step.
    dc_current = np.full_like(current, current.mean())
    # Return this value to the code that called the current function.
    return np.column_stack(
        # Begin the grouped expression or collection continued on the following lines.
        [current, delta_current, dc_current]
    # Use the expression `).astype(np.float32)` as the next part of the surrounding Python statement.
    ).astype(np.float32)


# Assign complete records or short blocks to data roles
def continuous_training_block(number_of_samples: int):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Describe one complete training-only experiment."""

    # Return this value to the code that called the current function.
    return [(0, 0, number_of_samples, "training")]


# Define the make_distributed_blocks function; its indented lines form the function body.
def make_distributed_blocks(number_of_samples: int):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Repeat train/validation/test roles across the complete chirp."""

    # Store the time-block assignments used for training, validation, and testing.
    blocks = []
    # Repeat the following indented block once for each item in this iterable.
    for block_number, start in enumerate(
        # Call `range`; the following indented continuation lines provide its arguments.
        range(0, number_of_samples, BLOCK_SIZE)
    # Begin the indented block controlled by this statement.
    ):
        # Store the exclusive ending index of the current block or plotted segment.
        stop = min(start + BLOCK_SIZE, number_of_samples)

        # A short remainder cannot contain one complete history window.
        role = (
            # Select `SPLIT_PATTERN[block_number % len(SPLIT_PATTERN)]` from the current array, tensor, table, or dictionary.
            SPLIT_PATTERN[block_number % len(SPLIT_PATTERN)]
            # Evaluate this condition and run the following indented block only when it is true.
            if stop - start >= SEQUENCE_LENGTH
            # Use the expression `else "unused_short_remainder"` as the next part of the surrounding Python statement.
            else "unused_short_remainder"
        # Close the current function call, tuple, or grouped expression.
        )
        # Call `blocks.append`; the following indented continuation lines provide its arguments.
        blocks.append((block_number, start, stop, role))

    # Return this value to the code that called the current function.
    return blocks


# Define the windows_from_blocks function; its indented lines form the function body.
def windows_from_blocks(
    # Pass `experiment` as the next value required by the surrounding call or collection.
    experiment,
    # Pass `blocks` as the next value required by the surrounding call or collection.
    blocks,
    # Pass `requested_role` as the next value required by the surrounding call or collection.
    requested_role,
    # Pass `input_mean` as the next value required by the surrounding call or collection.
    input_mean,
    # Pass `input_std` as the next value required by the surrounding call or collection.
    input_std,
    # Pass `output_mean` as the next value required by the surrounding call or collection.
    output_mean,
    # Pass `output_std` as the next value required by the surrounding call or collection.
    output_std,
# Begin the indented block controlled by this statement.
):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Create windows that remain fully inside one assigned block."""

    # Use the expression `time, current, outputs = experiment` as the next part of the surrounding Python statement.
    time, current, outputs = experiment
    # Normalize the three input features using training-only statistics.
    features_n = (make_features(current) - input_mean) / input_std
    # Normalize displacement and force using training-only statistics.
    outputs_n = (outputs - output_mean) / output_std

    # Evaluate `{` and store the result in `stride_by_role` for the following steps.
    stride_by_role = {
        # Store the 'training' field in the current dictionary.
        "training": TRAIN_STRIDE,
        # Store the 'validation' field in the current dictionary.
        "validation": VALIDATION_STRIDE,
        # Store the 'test' field in the current dictionary.
        "test": TEST_STRIDE,
    # Close the current dictionary.
    }

    # Each input has shape [120 time steps, 3 current features].
    inputs = []
    # Evaluate `[]` and store the result in `targets` for the following steps.
    targets = []
    # Evaluate `[]` and store the result in `target_time` for the following steps.
    target_time = []

    # Repeat the following indented block once for each item in this iterable.
    for _, start, stop, role in blocks:
        # Evaluate this condition and run the following indented block only when it is true.
        if role != requested_role:
            # Skip the remaining statements in this loop iteration and continue with the next item.
            continue

        # Evaluate `start + SEQUENCE_LENGTH - 1` and store the result in `first_target` for the following steps.
        first_target = start + SEQUENCE_LENGTH - 1
        # Repeat the following indented block once for each item in this iterable.
        for target_index in range(
            # Pass `first_target` as the next value required by the surrounding call or collection.
            first_target,
            # Pass `stop` as the next value required by the surrounding call or collection.
            stop,
            # Select `stride_by_role[requested_role]` from the current array, tensor, table, or dictionary.
            stride_by_role[requested_role],
        # Begin the indented block controlled by this statement.
        ):
            # Call `inputs.append`; the following indented continuation lines provide its arguments.
            inputs.append(
                # Use the expression `features_n[` as the next part of the surrounding Python statement.
                features_n[
                    # Begin the indented block controlled by this statement.
                    target_index - SEQUENCE_LENGTH + 1:
                    # Use the expression `target_index + 1` as the next part of the surrounding Python statement.
                    target_index + 1
                # Close the current list or index expression.
                ]
            # Close the current function call, tuple, or grouped expression.
            )
            # Call `targets.append`; the following indented continuation lines provide its arguments.
            targets.append(outputs_n[target_index])
            # Call `target_time.append`; the following indented continuation lines provide its arguments.
            target_time.append(time[target_index])

    # Evaluate this condition and run the following indented block only when it is true.
    if not inputs:
        # Stop this operation and report the stated error condition.
        raise ValueError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"No {requested_role} windows were produced. "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Check BLOCK_SIZE and SEQUENCE_LENGTH."
        # Close the current function call, tuple, or grouped expression.
        )

    # Return this value to the code that called the current function.
    return (
        # Call `torch.tensor`; the following indented continuation lines provide its arguments.
        torch.tensor(np.asarray(inputs), dtype=torch.float32),
        # Call `torch.tensor`; the following indented continuation lines provide its arguments.
        torch.tensor(np.asarray(targets), dtype=torch.float32),
        # Call `np.asarray`; the following indented continuation lines provide its arguments.
        np.asarray(target_time, dtype=np.float32),
    # Close the current function call, tuple, or grouped expression.
    )


# Build dense windows for the untouched fifth record
def full_test_windows(
    # Pass `experiment` as the next value required by the surrounding call or collection.
    experiment,
    # Pass `input_mean` as the next value required by the surrounding call or collection.
    input_mean,
    # Pass `input_std` as the next value required by the surrounding call or collection.
    input_std,
    # Pass `output_mean` as the next value required by the surrounding call or collection.
    output_mean,
    # Pass `output_std` as the next value required by the surrounding call or collection.
    output_std,
# Begin the indented block controlled by this statement.
):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Create dense windows from the untouched fifth experiment."""

    # Use the expression `time, current, outputs = experiment` as the next part of the surrounding Python statement.
    time, current, outputs = experiment
    # Normalize the three input features using training-only statistics.
    features_n = (make_features(current) - input_mean) / input_std
    # Normalize displacement and force using training-only statistics.
    outputs_n = (outputs - output_mean) / output_std

    # Evaluate `[]` and store the result in `inputs` for the following steps.
    inputs = []
    # Evaluate `[]` and store the result in `targets` for the following steps.
    targets = []
    # Evaluate `[]` and store the result in `target_time` for the following steps.
    target_time = []

    # Repeat the following indented block once for each item in this iterable.
    for target_index in range(
        # Use the expression `SEQUENCE_LENGTH - 1` as the next part of the surrounding Python statement.
        SEQUENCE_LENGTH - 1,
        # Call `len`; the following indented continuation lines provide its arguments.
        len(current),
        # Pass `TEST_STRIDE` as the next value required by the surrounding call or collection.
        TEST_STRIDE,
    # Begin the indented block controlled by this statement.
    ):
        # Call `inputs.append`; the following indented continuation lines provide its arguments.
        inputs.append(
            # Use the expression `features_n[` as the next part of the surrounding Python statement.
            features_n[
                # Begin the indented block controlled by this statement.
                target_index - SEQUENCE_LENGTH + 1:
                # Use the expression `target_index + 1` as the next part of the surrounding Python statement.
                target_index + 1
            # Close the current list or index expression.
            ]
        # Close the current function call, tuple, or grouped expression.
        )
        # Call `targets.append`; the following indented continuation lines provide its arguments.
        targets.append(outputs_n[target_index])
        # Call `target_time.append`; the following indented continuation lines provide its arguments.
        target_time.append(time[target_index])

    # Return this value to the code that called the current function.
    return (
        # Call `torch.tensor`; the following indented continuation lines provide its arguments.
        torch.tensor(np.asarray(inputs), dtype=torch.float32),
        # Call `torch.tensor`; the following indented continuation lines provide its arguments.
        torch.tensor(np.asarray(targets), dtype=torch.float32),
        # Call `np.asarray`; the following indented continuation lines provide its arguments.
        np.asarray(target_time, dtype=np.float32),
    # Close the current function call, tuple, or grouped expression.
    )


# Define the prepare_data function; its indented lines form the function body.
def prepare_data(folder: Path):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Build train, validation, internal-test, and pure-test datasets."""

    # Store the workbook path selected for this simulation.
    workbook = find_workbook(folder)
    # Evaluate `TRAIN_ONLY_SHEETS + [MIXED_SHEET, FINAL_TEST_SHEET]` and store the result in `all_sheets` for the following steps.
    all_sheets = TRAIN_ONLY_SHEETS + [MIXED_SHEET, FINAL_TEST_SHEET]
    # Build a dictionary that keeps each loaded experiment under its identifier.
    experiments = {
        # Use the expression `sheet: load_sheet(workbook, sheet)` as the next part of the surrounding Python statement.
        sheet: load_sheet(workbook, sheet)
        # Repeat the following indented block once for each item in this iterable.
        for sheet in all_sheets
    # Close the current dictionary.
    }

    # Store the time-block assignments used for training, validation, and testing.
    blocks = {
        # Use the expression `sheet: continuous_training_block(len(experiments[sheet][0]))` as the next part of the surrounding Python statement.
        sheet: continuous_training_block(len(experiments[sheet][0]))
        # Repeat the following indented block once for each item in this iterable.
        for sheet in TRAIN_ONLY_SHEETS
    # Close the current dictionary.
    }
    # Use the expression `blocks[MIXED_SHEET] = make_distributed_blocks(` as the next part of the surrounding Python statement.
    blocks[MIXED_SHEET] = make_distributed_blocks(
        # Call `len`; the following indented continuation lines provide its arguments.
        len(experiments[MIXED_SHEET][0])
    # Close the current function call, tuple, or grouped expression.
    )

    # Fit scaling only on training samples to avoid test-data leakage.
    train_features = []
    # Evaluate `[]` and store the result in `train_outputs` for the following steps.
    train_outputs = []
    # Evaluate `[]` and store the result in `split_rows` for the following steps.
    split_rows = []

    # Repeat the following indented block once for each item in this iterable.
    for sheet in TRAIN_ONLY_SHEETS + [MIXED_SHEET]:
        # Use the expression `time, current, outputs = experiments[sheet]` as the next part of the surrounding Python statement.
        time, current, outputs = experiments[sheet]
        # Evaluate `make_features(current)` and store the result in `features` for the following steps.
        features = make_features(current)

        # Repeat the following indented block once for each item in this iterable.
        for block_number, start, stop, role in blocks[sheet]:
            # Call `split_rows.append`; the following indented continuation lines provide its arguments.
            split_rows.append(
                # Begin the grouped expression or collection continued on the following lines.
                {
                    # Store the 'sheet' field in the current dictionary.
                    "sheet": sheet,
                    # Store the 'block' field in the current dictionary.
                    "block": block_number,
                    # Store the 'role' field in the current dictionary.
                    "role": role,
                    # Store the 'start_sample' field in the current dictionary.
                    "start_sample": start,
                    # Store the 'stop_sample' field in the current dictionary.
                    "stop_sample": stop,
                    # Store the 'start_time_s' field in the current dictionary.
                    "start_time_s": float(time[start]),
                    # Store the 'end_time_s' field in the current dictionary.
                    "end_time_s": float(time[stop - 1]),
                # Close the current dictionary.
                }
            # Close the current function call, tuple, or grouped expression.
            )
            # Evaluate this condition and run the following indented block only when it is true.
            if role == "training":
                # Call `train_features.append`; the following indented continuation lines provide its arguments.
                train_features.append(features[start:stop])
                # Call `train_outputs.append`; the following indented continuation lines provide its arguments.
                train_outputs.append(outputs[start:stop])

    # Evaluate `np.concatenate(train_features)` and store the result in `train_features` for the following steps.
    train_features = np.concatenate(train_features)
    # Evaluate `np.concatenate(train_outputs)` and store the result in `train_outputs` for the following steps.
    train_outputs = np.concatenate(train_outputs)
    # Store the training-only mean used to center each input feature.
    input_mean = train_features.mean(axis=0)
    # Store the training-only input standard deviation. Add 1e-8 solely to
    # prevent division by zero if a feature is constant; it is too small to
    # materially change an ordinary nonzero standard deviation.
    input_std = train_features.std(axis=0) + 1e-8
    # Store the training-only mean used to center displacement and force.
    output_mean = train_outputs.mean(axis=0)
    # Store the two training-only output standard deviations and add the same
    # 1e-8 numerical-safety term before normalization.
    output_std = train_outputs.std(axis=0) + 1e-8

    # Convert the assigned blocks into tensors for the shared model.
    x_train_parts = []
    # Evaluate `[]` and store the result in `y_train_parts` for the following steps.
    y_train_parts = []
    # Repeat the following indented block once for each item in this iterable.
    for sheet in TRAIN_ONLY_SHEETS + [MIXED_SHEET]:
        # Use the expression `x_part, y_part, _ = windows_from_blocks(` as the next part of the surrounding Python statement.
        x_part, y_part, _ = windows_from_blocks(
            # Select `experiments[sheet]` from the current array, tensor, table, or dictionary.
            experiments[sheet],
            # Select `blocks[sheet]` from the current array, tensor, table, or dictionary.
            blocks[sheet],
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "training",
            # Pass `input_mean` as the next value required by the surrounding call or collection.
            input_mean,
            # Pass `input_std` as the next value required by the surrounding call or collection.
            input_std,
            # Pass `output_mean` as the next value required by the surrounding call or collection.
            output_mean,
            # Pass `output_std` as the next value required by the surrounding call or collection.
            output_std,
        # Close the current function call, tuple, or grouped expression.
        )
        # Call `x_train_parts.append`; the following indented continuation lines provide its arguments.
        x_train_parts.append(x_part)
        # Call `y_train_parts.append`; the following indented continuation lines provide its arguments.
        y_train_parts.append(y_part)

    # Use the expression `x_validation, y_validation, _ = windows_from_blocks(` as the next part of the surrounding Python statement.
    x_validation, y_validation, _ = windows_from_blocks(
        # Select `experiments[MIXED_SHEET]` from the current array, tensor, table, or dictionary.
        experiments[MIXED_SHEET],
        # Select `blocks[MIXED_SHEET]` from the current array, tensor, table, or dictionary.
        blocks[MIXED_SHEET],
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        "validation",
        # Pass `input_mean` as the next value required by the surrounding call or collection.
        input_mean,
        # Pass `input_std` as the next value required by the surrounding call or collection.
        input_std,
        # Pass `output_mean` as the next value required by the surrounding call or collection.
        output_mean,
        # Pass `output_std` as the next value required by the surrounding call or collection.
        output_std,
    # Close the current function call, tuple, or grouped expression.
    )
    # Use the expression `x_development_test, y_development_test, development_test_time = (` as the next part of the surrounding Python statement.
    x_development_test, y_development_test, development_test_time = (
        # Call `windows_from_blocks`; the following indented continuation lines provide its arguments.
        windows_from_blocks(
            # Select `experiments[MIXED_SHEET]` from the current array, tensor, table, or dictionary.
            experiments[MIXED_SHEET],
            # Select `blocks[MIXED_SHEET]` from the current array, tensor, table, or dictionary.
            blocks[MIXED_SHEET],
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "test",
            # Pass `input_mean` as the next value required by the surrounding call or collection.
            input_mean,
            # Pass `input_std` as the next value required by the surrounding call or collection.
            input_std,
            # Pass `output_mean` as the next value required by the surrounding call or collection.
            output_mean,
            # Pass `output_std` as the next value required by the surrounding call or collection.
            output_std,
        # Close the current function call, tuple, or grouped expression.
        )
    # Close the current function call, tuple, or grouped expression.
    )
    # Use the expression `x_final_test, y_final_test, final_test_time = full_test_windows(` as the next part of the surrounding Python statement.
    x_final_test, y_final_test, final_test_time = full_test_windows(
        # Select `experiments[FINAL_TEST_SHEET]` from the current array, tensor, table, or dictionary.
        experiments[FINAL_TEST_SHEET],
        # Pass `input_mean` as the next value required by the surrounding call or collection.
        input_mean,
        # Pass `input_std` as the next value required by the surrounding call or collection.
        input_std,
        # Pass `output_mean` as the next value required by the surrounding call or collection.
        output_mean,
        # Pass `output_std` as the next value required by the surrounding call or collection.
        output_std,
    # Close the current function call, tuple, or grouped expression.
    )

    # Return this value to the code that called the current function.
    return {
        # Store the 'workbook' field in the current dictionary.
        "workbook": workbook,
        # Store the 'experiments' field in the current dictionary.
        "experiments": experiments,
        # Store the 'split_table' field in the current dictionary.
        "split_table": pd.DataFrame(split_rows),
        # Store the 'x_train' field in the current dictionary.
        "x_train": torch.cat(x_train_parts),
        # Store the 'y_train' field in the current dictionary.
        "y_train": torch.cat(y_train_parts),
        # Store the 'x_validation' field in the current dictionary.
        "x_validation": x_validation,
        # Store the 'y_validation' field in the current dictionary.
        "y_validation": y_validation,
        # Store the 'x_development_test' field in the current dictionary.
        "x_development_test": x_development_test,
        # Store the 'y_development_test' field in the current dictionary.
        "y_development_test": y_development_test,
        # Store the 'development_test_time' field in the current dictionary.
        "development_test_time": development_test_time,
        # Store the 'x_final_test' field in the current dictionary.
        "x_final_test": x_final_test,
        # Store the 'y_final_test' field in the current dictionary.
        "y_final_test": y_final_test,
        # Store the 'final_test_time' field in the current dictionary.
        "final_test_time": final_test_time,
        # Store the 'input_mean' field in the current dictionary.
        "input_mean": input_mean,
        # Store the 'input_std' field in the current dictionary.
        "input_std": input_std,
        # Store the 'output_mean' field in the current dictionary.
        "output_mean": output_mean,
        # Store the 'output_std' field in the current dictionary.
        "output_std": output_std,
    # Close the current dictionary.
    }
