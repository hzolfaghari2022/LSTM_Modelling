"""
Beginner-friendly LSTM training code for TWO chirp experiments
=============================================================

Purpose
-------
This script trains an LSTM model for your actuator/system data.

Your current dataset has only TWO experiments:
    1) Chirp_1
    2) Chirp_2

Because there are only two experiments, the code does NOT train only on one
experiment and validate on the other one. That split is too harsh and usually
makes the validation result bad. Instead, the code uses both chirps and splits
EACH chirp internally into training/validation/testing windows.

System-identification model
---------------------------
Default mode is "series_parallel":

    Inputs at each time step:
        coil current, measured displacement, measured force

    Outputs at the next time step:
        displacement, force

This is a one-step-ahead system-identification model. It is usually the best
first model because it uses the recent measured output history as part of the
state information.

Important improvements in this version
--------------------------------------
1) It supports exactly two chirp experiments.
2) It resamples both chirps to a common time step by default.
   This is important because Chirp_1 and Chirp_2 have different sample times.
3) It uses block-based train/validation/test splitting so all frequency regions
   can appear in train, validation, and test.
4) It balances the number of training windows from Chirp_1 and Chirp_2.
   This prevents the longer chirp from dominating the training process.
5) It uses early stopping, learning-rate scheduling, gradient clipping,
   AdamW weight decay, and weighted multi-output loss.
6) It saves prediction plots, error plots, loss curves, metrics, and the model.

How to run
----------
Basic run:

python lstm_two_chirp_balanced_training_COMMENTED.py --open

After each run, the script automatically commits and pushes the updated files to GitHub.
To run without pushing, add --no-push.

If Excel is not found automatically, pass the full Excel path:

python lstm_two_chirp_balanced_training_COMMENTED.py --excel_path "C:\\full\\path\\Chrip_Input.xlsx" --open

More accurate but slower run:

python lstm_two_chirp_balanced_training_COMMENTED.py --epochs 400 --window_stride 1 --open
"""

# ---------------------------------------------------------------------------
# Imports: these libraries give Python the tools needed for this project.
# ---------------------------------------------------------------------------

from __future__ import annotations  # Allows modern type hints on older Python versions.

import argparse  # Lets the user change settings from the command line.
import json  # Used to save normalization parameters and summary information.
import os  # Used to open files automatically on Windows.
import subprocess  # Used to open files automatically on macOS/Linux.
import sys  # Used to detect the operating system.
from dataclasses import dataclass  # Used to create a simple scaler class.
from datetime import datetime  # Used to create timestamped Git commit messages.
from pathlib import Path  # Makes Windows file paths safer and easier to use.
from typing import Dict, List, Tuple  # Makes function inputs/outputs easier to read.

import matplotlib.pyplot as plt  # Used for plotting figures.
import numpy as np  # Used for numerical arrays and math.
import pandas as pd  # Used for reading Excel files and saving CSV files.
import torch  # Main PyTorch library for machine learning.
torch.set_num_threads(1)  # Keep CPU LSTM fast and avoid OpenMP thread overhead/deadlock on some systems.
torch.backends.mkldnn.enabled = False  # Disable MKLDNN because it can make small CPU LSTM training very slow on some machines.
import torch.nn as nn  # Neural network layers and loss functions.
from torch.utils.data import DataLoader, TensorDataset  # Converts arrays into mini-batches.

# ---------------------------------------------------------------------------
# Default folders for your Windows computer.
# ---------------------------------------------------------------------------

DEFAULT_SAVE_DIR = Path(  # This is the default folder where results will be saved.
    r"C:\Users\hzlfghri\OneDrive - The University of Memphis\The University of Memphis\My Research\New Project on LSTM\Code\LSTM_Gans_Simulation"
)

DEFAULT_EXCEL_PATH = DEFAULT_SAVE_DIR / "Chrip_Input.xlsx"  # This is the expected Excel filename.

POSSIBLE_EXCEL_NAMES = [  # The code will search for these possible file names.
    "Chrip_Input.xlsx",  # Your filename has this spelling.
    "Chirp_Input.xlsx",  # This is the standard spelling of chirp.
    "Chrip_Input(1).xlsx",  # This is the uploaded filename in ChatGPT.
    "Chirp_Input(1).xlsx",  # Alternative spelling.
]

# ---------------------------------------------------------------------------
# Small utility functions.
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Make training more repeatable by setting random seeds."""
    np.random.seed(seed)  # Fix NumPy randomness.
    torch.manual_seed(seed)  # Fix PyTorch CPU randomness.
    torch.cuda.manual_seed_all(seed)  # Fix PyTorch GPU randomness if a GPU exists.
    try:  # Some systems may not support these deterministic settings.
        torch.backends.cudnn.deterministic = True  # Make CUDA operations more repeatable.
        torch.backends.cudnn.benchmark = False  # Disable automatic nondeterministic speed tuning.
    except Exception:  # If deterministic settings fail, ignore the error.
        pass  # Continue normally.


def open_file(path: Path) -> None:
    """Open a saved file using the default application on your computer."""
    try:  # Try to open the file.
        if sys.platform.startswith("win"):  # Check if you are on Windows.
            os.startfile(str(path))  # Open file on Windows.
        elif sys.platform == "darwin":  # Check if you are on macOS.
            subprocess.run(["open", str(path)], check=False)  # Open file on macOS.
        else:  # Otherwise assume Linux.
            subprocess.run(["xdg-open", str(path)], check=False)  # Open file on Linux.
    except Exception as exc:  # If opening fails, show the reason.
        print(f"Could not open {path}: {exc}")  # Print helpful message.


def resolve_excel_path(requested_path: Path, save_dir: Path) -> Path:
    """Find the Excel file even if it is not exactly where the script expects."""
    if requested_path.exists():  # If the path provided by the user exists.
        return requested_path  # Use that Excel file.

    script_dir = Path(__file__).resolve().parent  # Folder where this Python file is stored.
    current_dir = Path.cwd()  # Folder where PowerShell is currently running.

    candidates: List[Path] = []  # List of possible Excel paths.
    candidates.append(requested_path)  # Add the originally requested path.

    for name in POSSIBLE_EXCEL_NAMES:  # Try all possible file names.
        candidates.append(save_dir / name)  # Try inside the save folder.
        candidates.append(script_dir / name)  # Try beside the Python script.
        candidates.append(current_dir / name)  # Try inside the current PowerShell folder.
        candidates.append(Path("/mnt/data") / name)  # Try the ChatGPT sandbox folder.

    for candidate in candidates:  # Loop over candidate paths.
        if candidate.exists():  # If this file exists.
            return candidate  # Use it.

    raise FileNotFoundError(  # Stop with a clear error if no file was found.
        "Excel file was not found. Copy Chrip_Input.xlsx beside this Python file "
        "or run with --excel_path followed by the full path to your Excel file."
    )

# ---------------------------------------------------------------------------
# Normalization helper.
# ---------------------------------------------------------------------------

@dataclass  # This makes a simple class that stores mean and standard deviation.
class Scaler:
    """Standard scaler: normalized_value = (physical_value - mean) / std."""

    mean: np.ndarray  # Mean of each data column.
    std: np.ndarray  # Standard deviation of each data column.

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Convert physical values to normalized values."""
        return (x - self.mean) / self.std  # Apply standard scaling.

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized values back to physical values."""
        return x * self.std + self.mean  # Reverse standard scaling.


def fit_scaler(arrays: List[np.ndarray]) -> Scaler:
    """Fit a scaler using only training data."""
    data = np.vstack(arrays).astype(np.float32)  # Stack all arrays vertically.
    mean = data.mean(axis=0)  # Calculate mean for each column.
    std = data.std(axis=0)  # Calculate standard deviation for each column.
    std[std < 1e-12] = 1.0  # Prevent division by zero.
    return Scaler(mean=mean, std=std)  # Return scaler object.

# ---------------------------------------------------------------------------
# Excel loading and cleaning.
# ---------------------------------------------------------------------------

def find_header_row(raw: pd.DataFrame) -> int:
    """Find the row that contains Time, Displacement, Current, and Force."""
    for row_index in range(len(raw)):  # Check every row in the raw sheet.
        raw_cells = raw.iloc[row_index].tolist()  # Convert current row to a list.
        text_cells: List[str] = []  # Store text versions of non-empty cells.
        for cell in raw_cells:  # Check each cell in this row.
            if pd.isna(cell):  # If the cell is empty.
                continue  # Ignore it.
            text_cells.append(str(cell).strip().lower())  # Convert cell to lowercase text.
        row_text = " ".join(text_cells)  # Join row cells into one string.
        has_time = "time" in row_text  # Detect time column.
        has_disp = "displacement" in row_text  # Detect displacement column.
        has_current = "current" in row_text  # Detect current column.
        has_force = "force" in row_text  # Detect force column.
        if has_time and has_disp and has_current and has_force:  # If all headers are present.
            return row_index  # Return the header row index.
    raise ValueError("Could not find the header row in one Excel sheet.")  # Stop if not found.


def standardize_column_names(columns: List[object]) -> Dict[object, str]:
    """Rename messy Excel column names to simple names used in the code."""
    mapping: Dict[object, str] = {}  # Dictionary from original name to clean name.
    for column in columns:  # Loop through all original columns.
        clean = str(column).lower().strip()  # Convert column name to lowercase text.
        if "time" in clean:  # If column is time.
            mapping[column] = "time"  # Rename it to time.
        elif "displacement" in clean:  # If column is displacement.
            mapping[column] = "displacement"  # Rename it to displacement.
        elif "current" in clean:  # If column is current.
            mapping[column] = "current"  # Rename it to current.
        elif "force" in clean:  # If column is force.
            mapping[column] = "force"  # Rename it to force.
    return mapping  # Return rename dictionary.


def resample_experiment(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    """Resample one experiment to a fixed time step using linear interpolation."""
    if dt <= 0:  # If dt is zero or negative.
        return df.copy()  # Do not resample.

    t = df["time"].to_numpy(dtype=np.float64)  # Original time vector.
    t0 = float(t[0])  # Start time.
    t1 = float(t[-1])  # End time.
    new_t = np.arange(t0, t1 + 0.5 * dt, dt)  # New uniform time vector.

    new_df = pd.DataFrame()  # Create empty dataframe.
    new_df["time"] = new_t.astype(np.float32)  # Save new time.
    for col in ["displacement", "current", "force"]:  # Resample each signal.
        new_df[col] = np.interp(new_t, t, df[col].to_numpy(dtype=np.float64)).astype(np.float32)  # Interpolate.
    return new_df  # Return resampled dataframe.


def load_experiments(excel_path: Path, resample_dt: float) -> Dict[str, pd.DataFrame]:
    """Load Chirp_1 and Chirp_2 sheets from the Excel file."""
    excel_file = pd.ExcelFile(excel_path)  # Open Excel workbook.
    experiments: Dict[str, pd.DataFrame] = {}  # Store experiments here.

    for sheet_name in excel_file.sheet_names:  # Loop over every sheet.
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)  # Read raw sheet.
        header_row = find_header_row(raw)  # Find row with real column names.
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header_row)  # Read using real headers.
        df = df.dropna(axis=1, how="all")  # Remove fully empty columns.
        df = df.rename(columns=standardize_column_names(list(df.columns)))  # Rename columns.

        required = ["time", "displacement", "current", "force"]  # Required columns.
        missing = [col for col in required if col not in df.columns]  # Find missing columns.
        if missing:  # If any column is missing.
            print(f"Skipping {sheet_name}: missing columns {missing}")  # Print warning.
            continue  # Skip this sheet.

        df = df[required].copy()  # Keep only needed columns.
        for col in required:  # Convert columns to numbers.
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Bad cells become NaN.
        df = df.dropna().reset_index(drop=True)  # Remove rows with missing values.

        if len(df) < 500:  # Check that enough samples exist.
            print(f"Skipping {sheet_name}: only {len(df)} valid rows")  # Print warning.
            continue  # Skip short sheet.

        original_len = len(df)  # Store original number of samples.
        df = resample_experiment(df, resample_dt)  # Resample to common sample time.
        experiments[sheet_name] = df  # Save cleaned experiment.

        if resample_dt > 0:  # If resampling was used.
            print(f"Loaded {sheet_name}: {original_len} rows -> {len(df)} rows after dt={resample_dt:g} s resampling")  # Info.
        else:  # If no resampling was used.
            print(f"Loaded {sheet_name}: {len(df)} rows without resampling")  # Info.

    if not experiments:  # If no sheets were loaded.
        raise RuntimeError("No valid experiments were loaded from the Excel file.")  # Stop.

    return experiments  # Return dictionary of experiments.

# ---------------------------------------------------------------------------
# Block-based split for only two chirp experiments.
# ---------------------------------------------------------------------------

def split_one_experiment_into_blocks(
    df: pd.DataFrame,
    block_len: int,
    train_ratio: float,
    val_ratio: float,
    seq_len: int,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Split one long experiment into several train/validation/test blocks."""
    train_blocks: List[pd.DataFrame] = []  # Store training segments.
    val_blocks: List[pd.DataFrame] = []  # Store validation segments.
    test_blocks: List[pd.DataFrame] = []  # Store test segments.

    n = len(df)  # Number of rows in this experiment.
    min_segment_len = seq_len + 5  # Minimum segment length needed to create windows.

    for start in range(0, n, block_len):  # Move through the experiment block by block.
        block = df.iloc[start:start + block_len].reset_index(drop=True)  # Extract one block.
        if len(block) < 3 * min_segment_len:  # If the block is too short.
            continue  # Skip it.

        train_end = int(train_ratio * len(block))  # End index of train part.
        val_end = int((train_ratio + val_ratio) * len(block))  # End index of validation part.

        train_df = block.iloc[:train_end].copy()  # Training part of the block.
        val_df = block.iloc[train_end:val_end].copy()  # Validation part of the block.
        test_df = block.iloc[val_end:].copy()  # Test part of the block.

        if len(train_df) > min_segment_len:  # If training part is long enough.
            train_blocks.append(train_df)  # Store training part.
        if len(val_df) > min_segment_len:  # If validation part is long enough.
            val_blocks.append(val_df)  # Store validation part.
        if len(test_df) > min_segment_len:  # If test part is long enough.
            test_blocks.append(test_df)  # Store test part.

    return train_blocks, val_blocks, test_blocks  # Return lists of segments.


def build_balanced_block_splits(
    experiments: Dict[str, pd.DataFrame],
    block_len: int,
    seq_len: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Create train/validation/test dictionaries from all chirps using blocks."""
    train: Dict[str, pd.DataFrame] = {}  # Training segments.
    val: Dict[str, pd.DataFrame] = {}  # Validation segments.
    test: Dict[str, pd.DataFrame] = {}  # Test segments.

    for exp_name, df in experiments.items():  # Loop over Chirp_1 and Chirp_2.
        tr_blocks, va_blocks, te_blocks = split_one_experiment_into_blocks(  # Split this experiment.
            df=df,  # Current experiment dataframe.
            block_len=block_len,  # Number of rows per block.
            train_ratio=0.70,  # 70 percent of each block for training.
            val_ratio=0.15,  # 15 percent of each block for validation.
            seq_len=seq_len,  # LSTM sequence length.
        )
        for i, part in enumerate(tr_blocks):  # Store training blocks.
            train[f"{exp_name}_block{i:02d}_train"] = part  # Name the segment.
        for i, part in enumerate(va_blocks):  # Store validation blocks.
            val[f"{exp_name}_block{i:02d}_val"] = part  # Name the segment.
        for i, part in enumerate(te_blocks):  # Store test blocks.
            test[f"{exp_name}_block{i:02d}_test"] = part  # Name the segment.

    if not train or not val or not test:  # Check that all groups have data.
        raise RuntimeError("Block split failed. Try smaller --seq_len or larger --block_len.")  # Stop if split failed.

    return train, val, test  # Return split dictionaries.

# ---------------------------------------------------------------------------
# Window creation.
# ---------------------------------------------------------------------------

def base_experiment_name(segment_name: str) -> str:
    """Extract Chirp_1 or Chirp_2 from a segment name."""
    if segment_name.startswith("Chirp_1"):  # If this segment belongs to Chirp_1.
        return "Chirp_1"  # Return base name.
    if segment_name.startswith("Chirp_2"):  # If this segment belongs to Chirp_2.
        return "Chirp_2"  # Return base name.
    return segment_name.split("_")[0]  # Fallback for other names.


def make_windows_from_dataframe(
    df: pd.DataFrame,
    seq_len: int,
    window_stride: int,
    input_scaler: Scaler,
    output_scaler: Scaler,
    input_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one time-series segment into many LSTM windows."""
    current = df[["current"]].to_numpy(dtype=np.float32)  # Current signal as [N, 1].
    outputs = df[["displacement", "force"]].to_numpy(dtype=np.float32)  # Output signals as [N, 2].

    current_scaled = input_scaler.transform(current)  # Normalize current.
    outputs_scaled = output_scaler.transform(outputs)  # Normalize outputs.

    if input_mode == "current_only":  # If only current is used as input.
        features = current_scaled  # Input features are [current].
    elif input_mode == "series_parallel":  # If measured outputs are also used as past state information.
        features = np.hstack([current_scaled, outputs_scaled])  # Input features are [current, displacement, force].
    else:  # If input mode is invalid.
        raise ValueError("input_mode must be current_only or series_parallel")  # Stop.

    X_list: List[np.ndarray] = []  # Store input windows.
    Y_list: List[np.ndarray] = []  # Store target outputs.
    idx_list: List[int] = []  # Store target row indices.

    last_start = len(df) - seq_len - 1  # Last valid start index.
    for start in range(0, last_start + 1, window_stride):  # Slide a window through the segment.
        end = start + seq_len  # End of input window.
        target_index = end  # Predict the next sample after the window.
        X_list.append(features[start:end])  # Add input window.
        Y_list.append(outputs_scaled[target_index])  # Add normalized target.
        idx_list.append(target_index)  # Add target index.

    if not X_list:  # If no windows were created.
        return (  # Return empty arrays.
            np.empty((0, seq_len, features.shape[1]), dtype=np.float32),  # Empty X.
            np.empty((0, 2), dtype=np.float32),  # Empty Y.
            np.empty((0,), dtype=np.int64),  # Empty indices.
        )

    X = np.asarray(X_list, dtype=np.float32)  # Convert X list to array.
    Y = np.asarray(Y_list, dtype=np.float32)  # Convert Y list to array.
    idx = np.asarray(idx_list, dtype=np.int64)  # Convert indices to array.
    return X, Y, idx  # Return windows, targets, and target indices.


def make_windows_for_group(
    group: Dict[str, pd.DataFrame],
    seq_len: int,
    window_stride: int,
    input_scaler: Scaler,
    output_scaler: Scaler,
    input_mode: str,
    balance_by_experiment: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windows for a whole group and optionally balance Chirp_1/Chirp_2."""
    rng = np.random.default_rng(seed)  # Random generator for reproducible balancing.
    by_exp_X: Dict[str, List[np.ndarray]] = {}  # Store X windows grouped by experiment.
    by_exp_Y: Dict[str, List[np.ndarray]] = {}  # Store Y windows grouped by experiment.
    by_exp_id: Dict[str, List[str]] = {}  # Store labels grouped by experiment.

    for segment_name, df in group.items():  # Loop over all segments in this group.
        X, Y, _ = make_windows_from_dataframe(  # Make windows for this segment.
            df=df,  # Segment data.
            seq_len=seq_len,  # LSTM sequence length.
            window_stride=window_stride,  # Window stride.
            input_scaler=input_scaler,  # Current scaler.
            output_scaler=output_scaler,  # Output scaler.
            input_mode=input_mode,  # Input mode.
        )
        if len(X) == 0:  # If this segment produced no windows.
            continue  # Skip it.
        exp_name = base_experiment_name(segment_name)  # Get Chirp_1 or Chirp_2.
        by_exp_X.setdefault(exp_name, []).append(X)  # Add X to this experiment.
        by_exp_Y.setdefault(exp_name, []).append(Y)  # Add Y to this experiment.
        by_exp_id.setdefault(exp_name, []).extend([segment_name] * len(X))  # Add labels.

    if not by_exp_X:  # If no windows were created.
        raise RuntimeError("No windows were created. Reduce --seq_len or check data.")  # Stop.

    X_parts: List[np.ndarray] = []  # Final X arrays.
    Y_parts: List[np.ndarray] = []  # Final Y arrays.
    labels: List[str] = []  # Final segment labels.

    counts = {exp: sum(len(a) for a in arrays) for exp, arrays in by_exp_X.items()}  # Count windows per experiment.
    min_count = min(counts.values()) if balance_by_experiment else None  # Target count for balancing.

    for exp_name in sorted(by_exp_X.keys()):  # Loop over experiments in stable order.
        X_exp = np.vstack(by_exp_X[exp_name])  # Stack X windows for this experiment.
        Y_exp = np.vstack(by_exp_Y[exp_name])  # Stack Y windows for this experiment.
        id_exp = np.asarray(by_exp_id[exp_name], dtype=object)  # Convert labels to array.
        if balance_by_experiment and min_count is not None and len(X_exp) > min_count:  # If this experiment has too many windows.
            chosen = rng.choice(len(X_exp), size=min_count, replace=False)  # Randomly choose windows.
            chosen.sort()  # Sort indices for cleaner order.
            X_exp = X_exp[chosen]  # Keep selected X.
            Y_exp = Y_exp[chosen]  # Keep selected Y.
            id_exp = id_exp[chosen]  # Keep selected labels.
        X_parts.append(X_exp)  # Add experiment X.
        Y_parts.append(Y_exp)  # Add experiment Y.
        labels.extend(id_exp.tolist())  # Add labels.

    X_all = np.vstack(X_parts)  # Combine all X windows.
    Y_all = np.vstack(Y_parts)  # Combine all Y windows.
    labels_all = np.asarray(labels, dtype=object)  # Convert labels to array.
    return X_all, Y_all, labels_all  # Return final arrays.

# ---------------------------------------------------------------------------
# Model and loss.
# ---------------------------------------------------------------------------

class MultiOutputLSTM(nn.Module):
    """LSTM model that predicts displacement and force."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()  # Initialize PyTorch module.
        lstm_dropout = dropout if num_layers > 1 else 0.0  # PyTorch uses LSTM dropout only when layers > 1.
        self.lstm = nn.LSTM(  # Create LSTM layer.
            input_size=input_dim,  # Number of input features per time step.
            hidden_size=hidden_size,  # Number of hidden units.
            num_layers=num_layers,  # Number of stacked LSTM layers.
            dropout=lstm_dropout,  # Dropout between LSTM layers.
            batch_first=True,  # Input shape is [batch, sequence, features].
        )
        self.head = nn.Sequential(  # Create prediction head.
            nn.Linear(hidden_size, hidden_size),  # Fully connected layer.
            nn.Tanh(),  # Smooth nonlinear activation.
            nn.Dropout(dropout),  # Dropout for regularization.
            nn.Linear(hidden_size, 2),  # Two outputs: displacement and force.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next displacement and force."""
        lstm_out, _ = self.lstm(x)  # Run the LSTM over the input sequence.
        last_state = lstm_out[:, -1, :]  # Use the final hidden state.
        prediction = self.head(last_state)  # Map hidden state to two outputs.
        return prediction  # Return normalized prediction.


class WeightedMSELoss(nn.Module):
    """MSE loss with different weights for displacement and force."""

    def __init__(self, disp_weight: float, force_weight: float):
        super().__init__()  # Initialize PyTorch module.
        self.register_buffer("weights", torch.tensor([disp_weight, force_weight], dtype=torch.float32))  # Store weights.

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate weighted MSE."""
        weights = self.weights.to(pred.device)  # Move weights to CPU/GPU.
        squared_error = (pred - target) ** 2  # Calculate squared error.
        weighted_error = squared_error * weights  # Apply output weights.
        return weighted_error.mean()  # Return average weighted MSE.

# ---------------------------------------------------------------------------
# Evaluation and plotting.
# ---------------------------------------------------------------------------

def fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate system-identification fit percentage."""
    numerator = np.linalg.norm(y_true - y_pred)  # Size of prediction error.
    denominator = np.linalg.norm(y_true - np.mean(y_true, axis=0, keepdims=True))  # Size of signal variation.
    if denominator < 1e-12:  # Avoid division by zero.
        return float("nan")  # Return NaN if fit cannot be computed.
    return float(100.0 * (1.0 - numerator / denominator))  # Return fit percentage.


def evaluate_model(model: nn.Module, X: np.ndarray, Y: np.ndarray, output_scaler: Scaler, device: torch.device, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Predict normalized windows and convert predictions back to physical units."""
    model.eval()  # Evaluation mode disables dropout.
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)), batch_size=batch_size, shuffle=False)  # Evaluation loader.
    pred_batches: List[np.ndarray] = []  # Store predictions.
    true_batches: List[np.ndarray] = []  # Store true values.
    with torch.no_grad():  # Do not calculate gradients during evaluation.
        for xb, yb in loader:  # Loop through batches.
            xb = xb.to(device).float()  # Move input to CPU/GPU.
            pred = model(xb).cpu().numpy()  # Predict and move result to NumPy.
            pred_batches.append(pred)  # Save predictions.
            true_batches.append(yb.numpy())  # Save true normalized targets.
    y_pred_scaled = np.vstack(pred_batches)  # Stack normalized predictions.
    y_true_scaled = np.vstack(true_batches)  # Stack normalized true values.
    y_pred = output_scaler.inverse_transform(y_pred_scaled)  # Convert prediction to physical units.
    y_true = output_scaler.inverse_transform(y_true_scaled)  # Convert truth to physical units.
    return y_true, y_pred  # Return physical values.


def plot_loss(history: Dict[str, List[float]], save_path: Path) -> None:
    """Save train/validation loss plot."""
    fig, ax = plt.subplots(figsize=(9, 5))  # Create figure.
    ax.semilogy(history["train"], label="Train MSE")  # Plot training loss.
    ax.semilogy(history["val"], label="Validation MSE")  # Plot validation loss.
    ax.set_xlabel("Epoch")  # x-axis label.
    ax.set_ylabel("Weighted MSE on normalized outputs")  # y-axis label.
    ax.set_title("LSTM training and validation loss")  # Plot title.
    ax.grid(True, linestyle=":", alpha=0.6)  # Add grid.
    ax.legend()  # Add legend.
    fig.tight_layout()  # Improve layout.
    fig.savefig(save_path, dpi=180, bbox_inches="tight")  # Save figure.
    plt.close(fig)  # Close figure.


def plot_prediction(time: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path) -> None:
    """Save measured-vs-predicted displacement and force plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)  # Create two subplots.
    axes[0].plot(time, y_true[:, 0], label="Measured displacement", linewidth=1.2)  # True displacement.
    axes[0].plot(time, y_pred[:, 0], "--", label="Predicted displacement", linewidth=1.2)  # Predicted displacement.
    axes[0].set_ylabel("Displacement (mm)")  # y-axis label.
    axes[0].grid(True, linestyle=":", alpha=0.6)  # Add grid.
    axes[0].legend()  # Add legend.
    axes[1].plot(time, y_true[:, 1], label="Measured / computed force", linewidth=1.0)  # True force.
    axes[1].plot(time, y_pred[:, 1], "--", label="Predicted force", linewidth=1.0)  # Predicted force.
    axes[1].set_xlabel("Time (s)")  # x-axis label.
    axes[1].set_ylabel("Lorentz force (N)")  # y-axis label.
    axes[1].grid(True, linestyle=":", alpha=0.6)  # Add grid.
    axes[1].legend()  # Add legend.
    fig.suptitle(title)  # Main title.
    fig.tight_layout()  # Improve spacing.
    fig.savefig(save_path, dpi=180, bbox_inches="tight")  # Save PNG.
    plt.close(fig)  # Close figure.


def plot_error(time: np.ndarray, error: np.ndarray, title: str, save_path: Path) -> None:
    """Save prediction-error plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # Create two subplots.
    axes[0].plot(time, error[:, 0], linewidth=1.0)  # Displacement error.
    axes[0].axhline(0.0, color="black", linewidth=0.8)  # Zero-error line.
    axes[0].set_ylabel("Disp. error (mm)")  # y-axis label.
    axes[0].grid(True, linestyle=":", alpha=0.6)  # Add grid.
    axes[1].plot(time, error[:, 1], linewidth=1.0)  # Force error.
    axes[1].axhline(0.0, color="black", linewidth=0.8)  # Zero-error line.
    axes[1].set_xlabel("Time (s)")  # x-axis label.
    axes[1].set_ylabel("Force error (N)")  # y-axis label.
    axes[1].grid(True, linestyle=":", alpha=0.6)  # Add grid.
    fig.suptitle(title)  # Main title.
    fig.tight_layout()  # Improve spacing.
    fig.savefig(save_path, dpi=180, bbox_inches="tight")  # Save PNG.
    plt.close(fig)  # Close figure.


# ---------------------------------------------------------------------------
# GitHub automatic push helper.
# ---------------------------------------------------------------------------

REPO_SSH = "git@github.com:hzolfaghari2022/LSTM_Modelling.git"


def run_git(cmd_str: str, check: bool = True) -> Tuple[int, str]:
    """Run one Git command and print its output."""
    print(f"  $ {cmd_str}")
    result = subprocess.run(
        cmd_str,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output)
    if check and result.returncode != 0:
        print(f"\nERROR: Git command failed:\n{cmd_str}")
        sys.exit(1)
    return result.returncode, output


def git_push_to_github(repo_dir: Path) -> None:
    """Commit and push the saved results to GitHub after each run."""
    print("\n" + "=" * 78)
    print("Automatic GitHub push")
    print("Repository: hzolfaghari2022/LSTM_Modelling")
    print("=" * 78)

    repo_dir = Path(repo_dir).resolve()
    repo_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(repo_dir)
    print(f"Git working folder: {repo_dir}")

    run_git("git --version")

    # Initialize the folder as a Git repository if needed.
    rc_repo, _ = run_git("git rev-parse --is-inside-work-tree", check=False)
    if rc_repo != 0:
        print("This folder is not a Git repository. Initializing Git now...")
        run_git("git init")
        run_git("git branch -M main", check=False)

    # Always use the correct identity and SSH remote.
    run_git('git config user.name "Hussein Zolfaghari"')
    run_git('git config user.email "h.zolfaghari2015@gmail.com"')

    rc_remote, current_remote = run_git("git remote get-url origin", check=False)
    if rc_remote != 0:
        run_git(f"git remote add origin {REPO_SSH}")
    elif current_remote.strip() != REPO_SSH:
        run_git(f"git remote set-url origin {REPO_SSH}")
    else:
        print(f"Remote origin is already correct: {REPO_SSH}")

    # Keep the local branch name consistent with GitHub main.
    run_git("git branch -M main", check=False)

    # If a previous merge is already open and conflicts are fixed, finish it first.
    merge_head = repo_dir / ".git" / "MERGE_HEAD"
    if merge_head.exists():
        rc_unmerged, unmerged = run_git("git diff --name-only --diff-filter=U", check=False)
        if unmerged.strip():
            print("\nGit has an unfinished merge with unresolved conflicts.")
            print("Resolve the conflicts first, then run this code again.")
            sys.exit(1)
        run_git("git add .")
        rc_merge_commit, merge_out = run_git(
            'git commit -m "Complete previous merge before automatic push"',
            check=False,
        )
        if rc_merge_commit != 0 and "nothing to commit" not in merge_out.lower():
            print(f"\nCould not complete the previous merge:\n{merge_out}")
            sys.exit(1)

    # Stage everything generated by the run.
    run_git("git add .")
    run_git("git status", check=False)

    # Commit only if there are staged changes.
    rc_diff, _ = run_git("git diff --cached --quiet", check=False)
    if rc_diff != 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Update two-chirp LSTM results - {timestamp}"
        rc_commit, commit_out = run_git(f'git commit -m "{commit_message}"', check=False)
        if rc_commit != 0:
            print(f"\nGit commit failed:\n{commit_out}")
            sys.exit(1)
        print(f"\nCommit completed successfully:\n{commit_message}")
    else:
        print("\nNo new local changes to commit.")

    # Pull first so push is not rejected when GitHub main already has commits.
    rc_pull, pull_out = run_git(
        "git pull origin main --allow-unrelated-histories --no-rebase --no-edit",
        check=False,
    )
    if rc_pull != 0:
        print("\nGit pull failed. GitHub main has changes that need manual attention.")
        print("Run this in PowerShell from the same folder and send me the output if it fails:")
        print("  git status")
        print("  git pull origin main --allow-unrelated-histories --no-rebase")
        print("\nGit message:")
        print(pull_out)
        sys.exit(1)

    rc_push, push_out = run_git("git push -u origin main", check=False)
    if rc_push != 0:
        print("\nGit push failed.")
        print("Check your SSH connection with: ssh -T git@github.com")
        print("\nGit message:")
        print(push_out)
        sys.exit(1)

    print("\nFiles pushed successfully to GitHub main branch.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main program.
# ---------------------------------------------------------------------------

def main() -> None:
    """Run complete training and testing pipeline."""
    parser = argparse.ArgumentParser(description="Balanced LSTM training for two chirp experiments")  # Command-line parser.
    parser.add_argument("--excel_path", type=str, default=str(DEFAULT_EXCEL_PATH), help="Path to Chrip_Input.xlsx")  # Excel path.
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR), help="Folder where results are saved")  # Output folder.
    parser.add_argument("--input_mode", type=str, default="series_parallel", choices=["series_parallel", "current_only"], help="Input features")  # Model input mode.
    parser.add_argument("--seq_len", type=int, default=120, help="Number of past samples in each LSTM window")  # Sequence length.
    parser.add_argument("--window_stride", type=int, default=2, help="Sliding-window step; 1 is most accurate but slower")  # Window stride.
    parser.add_argument("--block_len", type=int, default=2500, help="Rows per block for balanced time split")  # Block length.
    parser.add_argument("--resample_dt", type=float, default=0.002, help="Common sample time in seconds; 0 disables resampling")  # Resampling dt.
    parser.add_argument("--hidden_size", type=int, default=64, help="LSTM hidden units")  # Hidden size.
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")  # Number of layers.
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout probability")  # Dropout.
    parser.add_argument("--epochs", type=int, default=300, help="Maximum number of epochs")  # Epochs.
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")  # Batch size.
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")  # Learning rate.
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")  # Weight decay.
    parser.add_argument("--disp_loss_weight", type=float, default=2.0, help="Loss weight for displacement")  # Displacement weight.
    parser.add_argument("--force_loss_weight", type=float, default=1.0, help="Loss weight for force")  # Force weight.
    parser.add_argument("--patience", type=int, default=60, help="Early stopping patience")  # Early stopping patience.
    parser.add_argument("--balance_experiments", action="store_true", default=True, help="Balance Chirp_1 and Chirp_2 window counts")  # Balance option.
    parser.add_argument("--seed", type=int, default=7, help="Random seed")  # Random seed.
    parser.add_argument("--open", action="store_true", help="Open generated plots at the end")  # Open option.
    parser.add_argument("--no-push", action="store_true", help="Run without committing/pushing results to GitHub")  # Disable GitHub push.
    args = parser.parse_args()  # Parse arguments.

    set_seed(args.seed)  # Fix random seeds.
    save_dir = Path(args.save_dir)  # Convert save_dir to Path.
    save_dir.mkdir(parents=True, exist_ok=True)  # Create folder if needed.
    excel_path = resolve_excel_path(Path(args.excel_path), save_dir)  # Locate Excel file.

    print("=" * 78)  # Print separator.
    print("Balanced LSTM training for TWO chirp experiments")  # Print title.
    print(f"Excel file : {excel_path}")  # Print Excel path.
    print(f"Save folder: {save_dir}")  # Print save folder.
    print(f"Input mode : {args.input_mode}")  # Print input mode.
    print(f"Resample dt: {args.resample_dt:g} s")  # Print resample dt.
    print("Outputs    : displacement and Lorentz force")  # Print outputs.
    print("=" * 78)  # Print separator.

    experiments = load_experiments(excel_path, resample_dt=args.resample_dt)  # Load Excel sheets.
    train_group, val_group, test_group = build_balanced_block_splits(experiments, args.block_len, args.seq_len)  # Split data.

    print(f"Train segments: {len(train_group)}")  # Print number of train segments.
    print(f"Val segments  : {len(val_group)}")  # Print number of validation segments.
    print(f"Test segments : {len(test_group)}")  # Print number of test segments.

    train_currents = [df[["current"]].to_numpy(dtype=np.float32) for df in train_group.values()]  # Current arrays for scaler.
    train_outputs = [df[["displacement", "force"]].to_numpy(dtype=np.float32) for df in train_group.values()]  # Output arrays for scaler.
    input_scaler = fit_scaler(train_currents)  # Fit current scaler from training data only.
    output_scaler = fit_scaler(train_outputs)  # Fit output scaler from training data only.

    input_dim = 1 if args.input_mode == "current_only" else 3  # Number of input features.
    print(f"Input dimension per time step: {input_dim}")  # Print input dimension.

    X_train, Y_train, train_labels = make_windows_for_group(  # Create training windows.
        train_group, args.seq_len, args.window_stride, input_scaler, output_scaler, args.input_mode, args.balance_experiments, args.seed
    )
    X_val, Y_val, val_labels = make_windows_for_group(  # Create validation windows.
        val_group, args.seq_len, 1, input_scaler, output_scaler, args.input_mode, args.balance_experiments, args.seed + 1
    )

    print(f"Training windows  : X={X_train.shape}, Y={Y_train.shape}")  # Print training shape.
    print(f"Validation windows: X={X_val.shape}, Y={Y_val.shape}")  # Print validation shape.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available.
    print(f"Device: {device}")  # Print device.

    train_loader = DataLoader(  # Create training mini-batches.
        TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),  # Convert arrays to tensors.
        batch_size=args.batch_size,  # Batch size.
        shuffle=True,  # Shuffle training windows.
    )
    val_loader = DataLoader(  # Create validation mini-batches.
        TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)),  # Convert arrays to tensors.
        batch_size=args.batch_size,  # Batch size.
        shuffle=False,  # Do not shuffle validation data.
    )

    model = MultiOutputLSTM(input_dim=input_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)  # Create LSTM.
    criterion = WeightedMSELoss(args.disp_loss_weight, args.force_loss_weight)  # Weighted multi-output loss.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # AdamW optimizer.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)  # LR scheduler.

    history: Dict[str, List[float]] = {"train": [], "val": []}  # Save loss history.
    best_val_loss = float("inf")  # Best validation loss.
    best_state = None  # Best model weights.
    bad_epochs = 0  # Counter for early stopping.

    for epoch in range(1, args.epochs + 1):  # Loop through epochs.
        model.train()  # Training mode enables dropout.
        train_losses: List[float] = []  # Store training batch losses.

        for xb, yb in train_loader:  # Loop over mini-batches.
            xb = xb.to(device).float()  # Move input to device.
            yb = yb.to(device).float()  # Move target to device.
            optimizer.zero_grad()  # Clear previous gradients.
            pred = model(xb)  # Predict displacement and force.
            loss = criterion(pred, yb)  # Calculate weighted loss.
            loss.backward()  # Backpropagation.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients.
            optimizer.step()  # Update model weights.
            train_losses.append(float(loss.item()))  # Save training loss.

        model.eval()  # Evaluation mode disables dropout.
        val_losses: List[float] = []  # Store validation losses.
        with torch.no_grad():  # Do not calculate gradients for validation.
            for xb, yb in val_loader:  # Loop over validation batches.
                xb = xb.to(device).float()  # Move input to device.
                yb = yb.to(device).float()  # Move target to device.
                pred = model(xb)  # Predict validation data.
                loss = criterion(pred, yb)  # Calculate validation loss.
                val_losses.append(float(loss.item()))  # Save validation loss.

        train_loss = float(np.mean(train_losses))  # Average training loss.
        val_loss = float(np.mean(val_losses))  # Average validation loss.
        history["train"].append(train_loss)  # Save training loss.
        history["val"].append(val_loss)  # Save validation loss.
        scheduler.step(val_loss)  # Reduce learning rate if validation stops improving.

        if val_loss < best_val_loss:  # If this is the best validation loss so far.
            best_val_loss = val_loss  # Update best validation loss.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # Copy best weights.
            bad_epochs = 0  # Reset early-stopping counter.
        else:  # If validation did not improve.
            bad_epochs += 1  # Increase counter.

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:  # Print progress sometimes.
            current_lr = optimizer.param_groups[0]["lr"]  # Read current learning rate.
            print(f"Epoch {epoch:4d}/{args.epochs} | train loss={train_loss:.6g} | val loss={val_loss:.6g} | lr={current_lr:.2e}")  # Print.

        if bad_epochs >= args.patience:  # If validation did not improve for too long.
            print(f"Early stopping at epoch {epoch}; best val loss={best_val_loss:.6g}")  # Print stop message.
            break  # Stop training.

    if best_state is not None:  # If best weights were saved.
        model.load_state_dict(best_state)  # Restore best validation model.

    loss_path = save_dir / "training_validation_loss_two_chirps_balanced.png"  # Loss plot path.
    plot_loss(history, loss_path)  # Save loss curve.

    split_rows: List[dict] = []  # Store split summary rows.
    for group_name, group in [("train", train_group), ("validation", val_group), ("test", test_group)]:  # Loop groups.
        for name, df in group.items():  # Loop segments.
            split_rows.append({  # Add one summary row.
                "group": group_name,  # Group name.
                "segment": name,  # Segment name.
                "rows": len(df),  # Number of rows.
                "time_start": float(df["time"].iloc[0]),  # Start time.
                "time_end": float(df["time"].iloc[-1]),  # End time.
                "current_min": float(df["current"].min()),  # Current minimum.
                "current_max": float(df["current"].max()),  # Current maximum.
                "disp_min": float(df["displacement"].min()),  # Displacement minimum.
                "disp_max": float(df["displacement"].max()),  # Displacement maximum.
                "force_min": float(df["force"].min()),  # Force minimum.
                "force_max": float(df["force"].max()),  # Force maximum.
            })
    split_path = save_dir / "experiment_split_summary_two_chirps_balanced.csv"  # Split summary path.
    pd.DataFrame(split_rows).to_csv(split_path, index=False)  # Save split summary.

    metrics_rows: List[dict] = []  # Store metrics rows.
    generated_figures: List[Path] = [loss_path]  # Store figures.

    # Evaluate every test segment separately so you can see performance over different frequency regions.
    for segment_name, df in test_group.items():  # Loop over test segments.
        X_test, Y_test, target_idx = make_windows_from_dataframe(  # Create test windows with stride 1.
            df, args.seq_len, 1, input_scaler, output_scaler, args.input_mode
        )
        if len(X_test) == 0:  # If no windows exist.
            continue  # Skip.
        y_true, y_pred = evaluate_model(model, X_test, Y_test, output_scaler, device, args.batch_size)  # Predict test data.
        error = y_true - y_pred  # Physical error = measured - predicted.
        time = df["time"].to_numpy(dtype=np.float32)[target_idx]  # Target time values.

        disp_rmse = float(np.sqrt(np.mean(error[:, 0] ** 2)))  # Displacement RMSE.
        force_rmse = float(np.sqrt(np.mean(error[:, 1] ** 2)))  # Force RMSE.
        disp_fit = fit_percent(y_true[:, [0]], y_pred[:, [0]])  # Displacement fit.
        force_fit = fit_percent(y_true[:, [1]], y_pred[:, [1]])  # Force fit.
        total_fit = fit_percent(y_true, y_pred)  # Overall fit.

        metrics_rows.append({  # Store metrics row.
            "segment": segment_name,  # Segment name.
            "disp_rmse_mm": disp_rmse,  # Displacement RMSE.
            "force_rmse_N": force_rmse,  # Force RMSE.
            "disp_fit_percent": disp_fit,  # Displacement fit.
            "force_fit_percent": force_fit,  # Force fit.
            "total_fit_percent": total_fit,  # Overall fit.
            "n_test_windows": len(X_test),  # Number of windows.
        })

        safe_name = segment_name.replace("/", "_").replace("\\", "_")  # Safe filename.
        pred_path = save_dir / f"prediction_{safe_name}_two_chirps_balanced.png"  # Prediction plot path.
        err_path = save_dir / f"error_{safe_name}_two_chirps_balanced.png"  # Error plot path.
        csv_path = save_dir / f"predictions_{safe_name}_two_chirps_balanced.csv"  # CSV path.

        plot_prediction(time, y_true, y_pred, f"LSTM prediction on {segment_name}\nmode={args.input_mode}, two-chirp balanced split", pred_path)  # Save prediction plot.
        plot_error(time, error, f"Prediction error on {segment_name}: measured - predicted", err_path)  # Save error plot.

        pd.DataFrame({  # Save numeric predictions.
            "time_s": time,  # Time.
            "measured_displacement_mm": y_true[:, 0],  # True displacement.
            "predicted_displacement_mm": y_pred[:, 0],  # Predicted displacement.
            "measured_force_N": y_true[:, 1],  # True force.
            "predicted_force_N": y_pred[:, 1],  # Predicted force.
            "displacement_error_mm": error[:, 0],  # Displacement error.
            "force_error_N": error[:, 1],  # Force error.
        }).to_csv(csv_path, index=False)  # Write CSV.

        generated_figures.extend([pred_path, err_path])  # Save figure paths.

    metrics_path = save_dir / "metrics_by_segment_two_chirps_balanced.csv"  # Metrics CSV path.
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)  # Save metrics.

    model_path = save_dir / "lstm_two_chirps_balanced.pt"  # Model path.
    torch.save(model.state_dict(), model_path)  # Save trained model weights.

    norm_path = save_dir / "normalization_parameters_two_chirps_balanced.json"  # Normalization path.
    with open(norm_path, "w", encoding="utf-8") as f:  # Open JSON file.
        json.dump({  # Save normalization and training settings.
            "input_mean": input_scaler.mean.tolist(),  # Input mean.
            "input_std": input_scaler.std.tolist(),  # Input std.
            "output_mean": output_scaler.mean.tolist(),  # Output mean.
            "output_std": output_scaler.std.tolist(),  # Output std.
            "input_mode": args.input_mode,  # Input mode.
            "seq_len": args.seq_len,  # Sequence length.
            "resample_dt": args.resample_dt,  # Resample time.
            "block_len": args.block_len,  # Block length.
            "hidden_size": args.hidden_size,  # Hidden size.
            "num_layers": args.num_layers,  # Number of layers.
            "dropout": args.dropout,  # Dropout.
            "best_val_loss": best_val_loss,  # Best validation loss.
        }, f, indent=2)  # Pretty JSON.

    print("\nSaved files:")  # Print saved files.
    for path in [loss_path, split_path, metrics_path, model_path, norm_path, *generated_figures[1:]]:  # Loop over saved files.
        print(f"  {path}")  # Print path.

    if args.open:  # If user requested opening files.
        open_file(loss_path)  # Open loss plot.
        for fig_path in generated_figures[1:3]:  # Open first two result plots only.
            open_file(fig_path)  # Open plot.

    print("\nDone.")  # Final message.

    if args.no_push:  # If user wants to test without uploading to GitHub.
        print("\nGitHub push is disabled because you used --no-push.")
    else:  # Otherwise push the saved results automatically after each run.
        git_push_to_github(save_dir)


if __name__ == "__main__":  # If this file is run directly.
    main()  # Start the program.
