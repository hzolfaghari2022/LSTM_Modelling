#!/usr/bin/env python3
"""
Improved LSTM/NARX model for the chirp-input actuator dataset.

Why this version is different
-----------------------------
The previous pure input-to-output LSTM often collapsed to an average output.
For this actuator data, the displacement and force are dynamic states. The same
current value can correspond to many different displacement values depending on
previous motion, load, and excitation history. This script therefore uses a
NARX-style LSTM:

    [current history, dcurrent/dt history, time, load, case parameters,
     previous output history]  ->  next output change

The model predicts output increments, not absolute output directly:

    y_hat(k) = y(k-1) + Delta_y_hat(k)

This is usually much easier and more physically meaningful for dynamic system
identification.

Workbook expected
-----------------
The workbook should contain sheets named Load_1, Load_2, Load_3, etc. Each sheet
contains four side-by-side cases:
    Case 1: DC offset + sine
    Case 2: DC offset only
    Case 3: sine only
    Case 4: no current

Run
---
    python Improved_Chirp_LSTM_NARX_Model_FINAL.py
    python Improved_Chirp_LSTM_NARX_Model_FINAL.py --excel "path/to/Chirp_Input.xlsx"
    python Improved_Chirp_LSTM_NARX_Model_FINAL.py --epochs 400 --split within_case

GitHub
------
By default, the script pushes the code and the latest results to:
    git@github.com:hzolfaghari2022/LSTM_Modelling.git
Use --no-github to skip the push for a local-only run.

Split modes
-----------
within_case:
    Every experiment contributes to train, validation, and test using a
    chronological split. This is the right first check to verify that the model
    can learn the measured system trajectories.

leave_load3_out:
    Train mostly on Load_1 and Load_2 and test on Load_3 / unseen cases. This is
    a much harder generalization test and may require more data.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# User settings
# ============================================================
EXCEL_FILE = "Chirp_Input.xlsx"
OUTPUT_FOLDER = "Chirp_LSTM_NARX_Results"

RESAMPLE_DT = 0.001
SEQUENCE_LENGTH = 30                 # 30 samples at 0.001 s = 0.03 s memory
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

BATCH_SIZE = 1024
MAX_EPOCHS = 250
INITIAL_LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-6
GRADIENT_THRESHOLD = 1.0
PATIENCE = 90

NUM_LSTM_LAYERS = 1
NUM_HIDDEN_UNITS = 48
DROPOUT = 0.0
FC_HIDDEN_UNITS = 32

RANDOM_SEED = 42
SHOW_FIGURES = False

TARGET_COLUMNS = ["Displacement_mm", "Force_N"]

# GitHub settings. This is the repo used in your previous LSTM workflow.
DO_GITHUB_PUSH = True
REPO_SSH = "git@github.com:hzolfaghari2022/LSTM_Modelling.git"
REPO_NAME = "LSTM_Modelling"
BRANCH_NAME = "main"
TARGET_FOLDER_NAME = "Improved_Chirp_LSTM_NARX_Model"
INCLUDE_EXCEL_IN_GITHUB = True


# ============================================================
# General helpers
# ============================================================
def find_excel_file(script_dir: Path, file_name: str) -> Path:
    candidates = [script_dir / file_name, Path.cwd() / file_name]
    for parent in script_dir.parents:
        candidates.append(parent / file_name)
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find {file_name}. Put it beside this Python file or pass --excel path/to/file.xlsx"
    )




def make_excel_working_copy(excel_path: Path) -> Path:
    """
    Copy the workbook to a temporary local folder before pandas reads it.

    This avoids common Windows/OneDrive permission problems such as:
        PermissionError: [Errno 13] Permission denied: '...Chirp_Input.xlsx'

    If the original file is exclusively locked by Excel, even copying can fail.
    In that case, close Excel completely or manually copy the workbook to a
    local folder such as C:\\LSTM_Test and pass that path with --excel.
    """
    excel_path = Path(excel_path).resolve()
    cache_dir = Path(tempfile.gettempdir()) / "Chirp_LSTM_Excel_Cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use a stable output name so the latest run is easy to inspect.
    local_path = cache_dir / excel_path.name
    try:
        shutil.copy2(excel_path, local_path)
    except PermissionError as exc:
        raise PermissionError(
            "Python found the Excel file but Windows/OneDrive denied access to it.\n"
            "Fix: close Chirp_Input.xlsx in Excel, close all Excel windows, and run again.\n"
            "Alternative: copy Chirp_Input.xlsx to C:\\LSTM_Test and run with:\n"
            "  --excel C:\\LSTM_Test\\Chirp_Input.xlsx"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Could not copy the Excel workbook to a local cache. Original file:\n{excel_path}\n"
            "Try copying the file manually to C:\\LSTM_Test and pass that path with --excel."
        ) from exc

    # Confirm pandas can open the local copy.
    try:
        with pd.ExcelFile(local_path) as _:
            pass
    except PermissionError as exc:
        raise PermissionError(
            f"The local cached Excel copy is still not readable:\n{local_path}\n"
            "Close Excel completely and try again."
        ) from exc

    if local_path != excel_path:
        print(f"Using local cached copy of Excel file: {local_path}")
    return local_path


def run_command(command: List[str], cwd: Path | None = None) -> Tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout


def copy_tree_contents(src: Path, dst: Path) -> None:
    """Copy all files/folders from src into dst without deleting dst."""
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree_contents(item, target)
        elif item.is_file():
            try:
                shutil.copy2(item, target)
            except PermissionError:
                print(f"Warning: could not copy locked file: {item}")


def github_push(run_folder: Path, script_path: Path, excel_path: Path | None = None) -> None:
    """Copy the current code/results to GitHub and push by SSH."""
    print("\n====================================================")
    print("Preparing GitHub update")
    print("====================================================")
    print(f"Repository: {REPO_SSH}")

    github_root = Path(tempfile.gettempdir()) / "Python_GitHub_Repos"
    repo_folder = github_root / REPO_NAME
    target_folder = repo_folder / TARGET_FOLDER_NAME
    github_root.mkdir(parents=True, exist_ok=True)

    if not shutil.which("git"):
        print("Warning: git was not found on this computer. Results were saved locally but not pushed.")
        return

    if not repo_folder.exists():
        status, out = run_command(["git", "clone", REPO_SSH, str(repo_folder)])
        if status != 0:
            print("Warning: Git clone failed. Results were saved locally but not pushed.")
            print(out)
            return

    run_command(["git", "remote", "set-url", "origin", REPO_SSH], cwd=repo_folder)
    run_command(["git", "config", "user.name", "Hussein Zolfaghari"], cwd=repo_folder)
    run_command(["git", "config", "user.email", "h.zolfaghari2015@gmail.com"], cwd=repo_folder)
    run_command(["git", "branch", "-M", BRANCH_NAME], cwd=repo_folder)

    # Keep local work safe before pulling.
    run_command(["git", "stash", "push", "-u", "-m", "auto-stash-before-lstm-results-update"], cwd=repo_folder)
    status, out = run_command(["git", "pull", "origin", BRANCH_NAME, "--rebase"], cwd=repo_folder)
    if status != 0:
        print("Warning: git pull had an issue. The script will still try to commit/push the latest files.")
        print(out)

    target_folder.mkdir(parents=True, exist_ok=True)

    # Copy the final code.
    shutil.copy2(script_path, target_folder / script_path.name)

    # Copy the latest run outputs.
    if run_folder.is_dir():
        copy_tree_contents(run_folder, target_folder / run_folder.name)

    # Copy Excel for reproducibility if requested and readable.
    if INCLUDE_EXCEL_IN_GITHUB and excel_path is not None and Path(excel_path).is_file():
        try:
            shutil.copy2(excel_path, target_folder / Path(excel_path).name)
        except PermissionError:
            print("Warning: Excel file was locked and was not copied to GitHub folder.")

    # Write/update a simple run manifest.
    manifest = target_folder / "latest_run_manifest.txt"
    manifest.write_text(
        "Improved Chirp LSTM/NARX Model Latest Run\n"
        "=========================================\n"
        f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Script: {script_path.name}\n"
        f"Results folder: {run_folder.name}\n"
        f"Excel source: {excel_path}\n"
        f"Repository: {REPO_SSH}\n"
        f"Target folder: {TARGET_FOLDER_NAME}\n",
        encoding="utf-8",
    )

    status, out = run_command(["git", "add", "."], cwd=repo_folder)
    if status != 0:
        print("Warning: git add failed. Results were saved locally but not pushed.")
        print(out)
        return

    status, _ = run_command(["git", "diff", "--cached", "--quiet"], cwd=repo_folder)
    if status == 0:
        print("No new GitHub changes to commit.")
        return

    commit_message = f"Update improved Chirp LSTM NARX results {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    status, out = run_command(["git", "commit", "-m", commit_message], cwd=repo_folder)
    if status != 0:
        print("Warning: git commit failed. Results were saved locally but not pushed.")
        print(out)
        return

    status, out = run_command(["git", "push", "-u", "origin", BRANCH_NAME], cwd=repo_folder)
    if status != 0:
        print("Warning: git push failed. Your local results are safe, but GitHub was not updated.")
        print(out)
        print("Most common fix: make sure your SSH key is connected to GitHub.")
        return

    print("GitHub updated successfully.")

def safe_std(x: np.ndarray) -> np.ndarray:
    sig = x.std(axis=0)
    sig[sig < 1e-12] = 1.0
    return sig


def normalize(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (x - mu) / sig


def denormalize(xn: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return xn * sig + mu


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def r2_score(y: np.ndarray, yp: np.ndarray) -> float:
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def nrmse_percent(e: np.ndarray, y: np.ndarray) -> float:
    rng = float(np.max(y) - np.min(y))
    return float("nan") if rng < 1e-12 else 100.0 * rmse(e) / rng


def fit_percent(y: np.ndarray, yp: np.ndarray) -> float:
    denom = np.linalg.norm(y - np.mean(y))
    return float("nan") if denom < 1e-12 else 100.0 * (1.0 - np.linalg.norm(y - yp) / denom)


def save_fig(fig: plt.Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show(block=True)
    else:
        plt.close(fig)


# ============================================================
# Workbook parsing
# ============================================================
def parse_load_mass_grams(sheet_note: str) -> float:
    if not isinstance(sheet_note, str):
        return float("nan")
    m = re.search(r"Load\s*Mass\s*=\s*([0-9.]+)\s*gram", sheet_note, flags=re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")


def parse_coil_mass_grams(sheet_note: str) -> float:
    if not isinstance(sheet_note, str):
        return float("nan")
    m = re.search(r"Coil\s*Mass\s*=\s*([0-9.]+)\s*gram", sheet_note, flags=re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")


def parse_case_description(desc: str) -> Tuple[float, float, str]:
    if not isinstance(desc, str):
        return 0.0, 0.0, "Unknown"

    dc = 0.0
    amp = 0.0
    m_dc = re.search(r"DC_Offset\s*=\s*([0-9.]+)\s*A", desc, flags=re.IGNORECASE)
    if m_dc:
        dc = float(m_dc.group(1))
    m_amp = re.search(r"Sine\s*Amplitude\s*=\s*([0-9.]+)\s*A", desc, flags=re.IGNORECASE)
    if m_amp:
        amp = float(m_amp.group(1))

    if "DC_Offset+Sine" in desc:
        case_type = "DC_plus_sine"
    elif "DC_Offset only" in desc:
        case_type = "DC_only"
    elif "Sine only" in desc:
        case_type = "Sine_only"
    elif "No Current" in desc:
        case_type = "No_current"
    else:
        case_type = "Unknown"
    return dc, amp, case_type


def load_all_cases(excel_path: Path) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(excel_path)
    experiments: Dict[str, pd.DataFrame] = {}
    case_starts = [0, 5, 10, 15]

    print("Available worksheet names:", xls.sheet_names)
    for sheet_name in xls.sheet_names:
        if not str(sheet_name).lower().startswith("load"):
            continue

        raw_head = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, nrows=17)
        sheet_note = raw_head.iloc[0, 0]
        load_mass_g = parse_load_mass_grams(sheet_note)
        coil_mass_g = parse_coil_mass_grams(sheet_note)

        for case_id, start_col in enumerate(case_starts, start=1):
            case_desc = raw_head.iloc[15, start_col]
            dc_offset, sine_amp, case_type = parse_case_description(case_desc)

            block = pd.read_excel(
                excel_path,
                sheet_name=sheet_name,
                header=16,
                usecols=list(range(start_col, start_col + 4)),
            )
            block.columns = ["Time_s", "Displacement_mm", "Current_A", "Force_N"]
            for c in ["Time_s", "Displacement_mm", "Current_A", "Force_N"]:
                block[c] = pd.to_numeric(block[c], errors="coerce")
            block = block.dropna().sort_values("Time_s")
            block = block.groupby("Time_s", as_index=False).mean()
            if len(block) < SEQUENCE_LENGTH + 10:
                continue

            block["Sheet"] = sheet_name
            block["CaseID"] = case_id
            block["CaseType"] = case_type
            block["DCOffset_A"] = dc_offset
            block["SineAmplitude_A"] = sine_amp
            block["CoilMass_g"] = coil_mass_g
            block["LoadMass_g"] = load_mass_g

            label = f"{sheet_name}_Case_{case_id}_{case_type}"
            experiments[label] = block.reset_index(drop=True)

    if not experiments:
        raise RuntimeError("No valid experiments were parsed from the Excel workbook.")
    return experiments


def resample_uniform(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    t0 = float(df["Time_s"].iloc[0])
    t1 = float(df["Time_s"].iloc[-1])
    t_new = np.arange(t0, t1 + 0.5 * dt, dt)
    out = pd.DataFrame({"Time_s": t_new})
    for col in ["Displacement_mm", "Current_A", "Force_N"]:
        out[col] = np.interp(t_new, df["Time_s"].to_numpy(), df[col].to_numpy())
    for col in ["Sheet", "CaseID", "CaseType", "DCOffset_A", "SineAmplitude_A", "CoilMass_g", "LoadMass_g"]:
        out[col] = df[col].iloc[0]
    return out


# ============================================================
# Feature engineering
# ============================================================
def static_and_current_features(df: pd.DataFrame) -> np.ndarray:
    t = df["Time_s"].to_numpy()
    current = df["Current_A"].to_numpy()
    dc_dt = np.gradient(current, t)
    time_norm = (t - t.min()) / max(t.max() - t.min(), 1e-12)

    load = np.full_like(t, float(df["LoadMass_g"].iloc[0]), dtype=float)
    dc = np.full_like(t, float(df["DCOffset_A"].iloc[0]), dtype=float)
    amp = np.full_like(t, float(df["SineAmplitude_A"].iloc[0]), dtype=float)
    case_id = np.full_like(t, float(df["CaseID"].iloc[0]), dtype=float)

    return np.column_stack([current, dc_dt, time_norm, load, dc, amp, case_id])


def build_rows_for_experiment(
    label: str,
    df: pd.DataFrame,
    norm: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build a row table. Each row corresponds to one target time index."""
    Y = df[TARGET_COLUMNS].to_numpy()
    base = static_and_current_features(df)

    base_n = normalize(base, norm["mu_base"], norm["sig_base"])
    prev_y = np.vstack([Y[0:1], Y[:-1]])
    prev_y_n = normalize(prev_y, norm["mu_y"], norm["sig_y"])
    X_step = np.column_stack([base_n, prev_y_n])

    delta_y = Y - prev_y
    delta_y_n = normalize(delta_y, norm["mu_delta"], norm["sig_delta"])

    rows = []
    for i in range(SEQUENCE_LENGTH, len(df)):
        rows.append({
            "Experiment": label,
            "Index": i,
            "Time_s": float(df["Time_s"].iloc[i]),
            "Current_A": float(df["Current_A"].iloc[i]),
            "LoadMass_g": float(df["LoadMass_g"].iloc[i]),
            "CaseID": int(df["CaseID"].iloc[i]),
            "CaseType": str(df["CaseType"].iloc[i]),
            "X_window": X_step[i - SEQUENCE_LENGTH + 1:i + 1].astype(np.float32),
            "Y_delta_norm": delta_y_n[i].astype(np.float32),
            "Y_prev": prev_y[i].astype(np.float32),
            "Y_true": Y[i].astype(np.float32),
        })
    return pd.DataFrame(rows)


def make_normalization(data: Dict[str, pd.DataFrame], train_labels: List[str]) -> Dict[str, np.ndarray]:
    base_list, y_list, delta_list = [], [], []
    for lab in train_labels:
        df = data[lab]
        Y = df[TARGET_COLUMNS].to_numpy()
        base_list.append(static_and_current_features(df))
        y_list.append(Y)
        prev_y = np.vstack([Y[0:1], Y[:-1]])
        delta_list.append(Y - prev_y)
    base_all = np.vstack(base_list)
    y_all = np.vstack(y_list)
    delta_all = np.vstack(delta_list)
    return {
        "mu_base": base_all.mean(axis=0),
        "sig_base": safe_std(base_all),
        "mu_y": y_all.mean(axis=0),
        "sig_y": safe_std(y_all),
        "mu_delta": delta_all.mean(axis=0),
        "sig_delta": safe_std(delta_all),
    }


def split_row_table(rows: pd.DataFrame, split_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts, val_parts, test_parts = [], [], []

    if split_mode == "within_case":
        for _, g in rows.groupby("Experiment", sort=True):
            g = g.sort_values("Index").reset_index(drop=True)
            n = len(g)
            n_train = max(1, int(math.floor(TRAIN_FRACTION * n)))
            n_val = max(1, int(math.floor(VALIDATION_FRACTION * n)))
            train_parts.append(g.iloc[:n_train])
            val_parts.append(g.iloc[n_train:n_train + n_val])
            test_parts.append(g.iloc[n_train + n_val:])

    elif split_mode == "leave_load3_out":
        for _, g in rows.groupby("Experiment", sort=True):
            lab = g["Experiment"].iloc[0]
            if lab.startswith("Load_1") or lab.startswith("Load_2"):
                if "_Case_4_" not in lab:
                    train_parts.append(g)
                else:
                    test_parts.append(g)
            elif lab.startswith("Load_3") and "_Case_1_" in lab:
                val_parts.append(g)
            else:
                test_parts.append(g)
    else:
        raise ValueError("split_mode must be either 'within_case' or 'leave_load3_out'")

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def rows_to_tensors(df_rows: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    X = np.stack(df_rows["X_window"].to_numpy())
    Yd = np.stack(df_rows["Y_delta_norm"].to_numpy())
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Yd, dtype=torch.float32)


# ============================================================
# Model
# ============================================================
class NARXLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2,
                 num_layers: int = 2, dropout: float = 0.05, fc_hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.Tanh(),
            nn.Linear(fc_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ============================================================
# Prediction and metrics
# ============================================================
def predict_teacher_forced(model: nn.Module, rows: pd.DataFrame, norm: Dict[str, np.ndarray], device: torch.device) -> pd.DataFrame:
    X, _ = rows_to_tensors(rows)
    model.eval()
    preds = []
    batch = 2048
    with torch.no_grad():
        for i in range(0, len(X), batch):
            ydn = model(X[i:i + batch].to(device)).detach().cpu().numpy()
            delta = denormalize(ydn, norm["mu_delta"], norm["sig_delta"])
            prev = np.stack(rows.iloc[i:i + batch]["Y_prev"].to_numpy())
            preds.append(prev + delta)
    yp = np.vstack(preds)
    yt = np.stack(rows["Y_true"].to_numpy())

    out = rows[["Experiment", "Index", "Time_s", "Current_A", "LoadMass_g", "CaseID", "CaseType"]].copy()
    out["TrueDisplacement_mm"] = yt[:, 0]
    out["PredictedDisplacement_mm"] = yp[:, 0]
    out["DisplacementError_mm"] = yt[:, 0] - yp[:, 0]
    out["TrueForce_N"] = yt[:, 1]
    out["PredictedForce_N"] = yp[:, 1]
    out["ForceError_N"] = yt[:, 1] - yp[:, 1]
    return out


def metric_rows(pred: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    for exp, g in pred.groupby("Experiment", sort=True):
        de = g["DisplacementError_mm"].to_numpy()
        fe = g["ForceError_N"].to_numpy()
        yd = g["TrueDisplacement_mm"].to_numpy()
        ypd = g["PredictedDisplacement_mm"].to_numpy()
        yf = g["TrueForce_N"].to_numpy()
        ypf = g["PredictedForce_N"].to_numpy()
        rows.append({
            "Experiment": exp,
            "Split": split_name,
            "LoadMass_g": float(g["LoadMass_g"].iloc[0]),
            "CaseID": int(g["CaseID"].iloc[0]),
            "CaseType": str(g["CaseType"].iloc[0]),
            "Disp_RMSE_mm": rmse(de),
            "Disp_MAE_mm": mae(de),
            "Disp_NRMSE_percent": nrmse_percent(de, yd),
            "Disp_R2": r2_score(yd, ypd),
            "Disp_Fit_percent": fit_percent(yd, ypd),
            "Force_RMSE_N": rmse(fe),
            "Force_MAE_N": mae(fe),
            "Force_NRMSE_percent": nrmse_percent(fe, yf),
            "Force_R2": r2_score(yf, ypf),
            "Force_Fit_percent": fit_percent(yf, ypf),
        })
    return pd.DataFrame(rows)


# ============================================================
# Figures
# ============================================================
def plot_training_history(history: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    axes[0].plot(history["Epoch"], history["TrainingRMSE"], label="Training")
    axes[0].plot(history["Epoch"], history["ValidationRMSE"], "--", label="Validation")
    axes[0].set_ylabel("Normalized RMSE")
    axes[0].set_title("LSTM/NARX Training History")
    axes[0].legend()
    axes[1].plot(history["Epoch"], history["TrainingLoss"], label="Training")
    axes[1].plot(history["Epoch"], history["ValidationLoss"], "--", label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE loss")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_fig(fig, fig_dir, "Fig01_Training_History")


def plot_prediction_examples(predictions: Dict[str, pd.DataFrame], metrics: pd.DataFrame, fig_dir: Path) -> None:
    # Plot up to three representative experiments per split: best, median, worst displacement RMSE.
    for split, pred in predictions.items():
        split_metrics = metrics[metrics["Split"] == split].copy()
        if split_metrics.empty:
            continue
        split_metrics = split_metrics.sort_values("Disp_RMSE_mm")
        selected = []
        selected.append(split_metrics.iloc[0]["Experiment"])
        selected.append(split_metrics.iloc[len(split_metrics) // 2]["Experiment"])
        selected.append(split_metrics.iloc[-1]["Experiment"])
        selected = list(dict.fromkeys(selected))

        for exp in selected:
            g = pred[pred["Experiment"] == exp].sort_values("Time_s")
            m = split_metrics[split_metrics["Experiment"] == exp].iloc[0]
            fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True)
            axes[0].plot(g["Time_s"], g["Current_A"], linewidth=1.2)
            axes[0].set_ylabel("Current (A)")
            axes[0].set_title(f"{split}: {exp}")
            axes[1].plot(g["Time_s"], g["TrueDisplacement_mm"], label="COMSOL", linewidth=1.4)
            axes[1].plot(g["Time_s"], g["PredictedDisplacement_mm"], "--", label="LSTM/NARX", linewidth=1.4)
            axes[1].set_ylabel("Disp. (mm)")
            axes[1].set_title(f"Displacement: RMSE={m['Disp_RMSE_mm']:.4g} mm, R2={m['Disp_R2']:.4f}")
            axes[1].legend()
            axes[2].plot(g["Time_s"], g["TrueForce_N"], label="COMSOL", linewidth=1.4)
            axes[2].plot(g["Time_s"], g["PredictedForce_N"], "--", label="LSTM/NARX", linewidth=1.4)
            axes[2].set_ylabel("Force (N)")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_title(f"Force: RMSE={m['Force_RMSE_N']:.4g} N, R2={m['Force_R2']:.4f}")
            axes[2].legend()
            for ax in axes:
                ax.grid(True, alpha=0.3)
            safe_exp = re.sub(r"[^A-Za-z0-9_]+", "_", exp)
            save_fig(fig, fig_dir, f"Fig02_{split}_{safe_exp}_Prediction")


def plot_metric_summary(metrics: pd.DataFrame, fig_dir: Path) -> None:
    for split in metrics["Split"].unique():
        g = metrics[metrics["Split"] == split].sort_values("Experiment")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        x = np.arange(len(g))
        labels = g["Experiment"].str.replace("Load_", "L", regex=False).str.replace("_Case_", " C", regex=False)
        axes[0].bar(x, g["Disp_RMSE_mm"])
        axes[0].set_ylabel("Disp. RMSE (mm)")
        axes[0].set_title(f"{split}: RMSE Summary")
        axes[1].bar(x, g["Force_RMSE_N"])
        axes[1].set_ylabel("Force RMSE (N)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=35, ha="right")
        for ax in axes:
            ax.grid(True, axis="y", alpha=0.3)
        save_fig(fig, fig_dir, f"Fig03_{split}_Metric_Summary")


def plot_parity(predictions: Dict[str, pd.DataFrame], fig_dir: Path) -> None:
    for split, pred in predictions.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, true_col, pred_col, title, unit in [
            (axes[0], "TrueDisplacement_mm", "PredictedDisplacement_mm", "Displacement", "mm"),
            (axes[1], "TrueForce_N", "PredictedForce_N", "Force", "N"),
        ]:
            y = pred[true_col].to_numpy()
            yp = pred[pred_col].to_numpy()
            ax.scatter(y, yp, s=8, alpha=0.6)
            mn = min(float(np.min(y)), float(np.min(yp)))
            mx = max(float(np.max(y)), float(np.max(yp)))
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2)
            ax.set_xlabel(f"COMSOL ({unit})")
            ax.set_ylabel(f"LSTM/NARX ({unit})")
            ax.set_title(f"{split}: {title} parity, R2={r2_score(y, yp):.4f}")
            ax.grid(True, alpha=0.3)
        save_fig(fig, fig_dir, f"Fig04_{split}_Parity")


# ============================================================
# Main workflow
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, default=None, help="Path to Chirp_Input.xlsx")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--split", type=str, default="within_case", choices=["within_case", "leave_load3_out"])
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--no-github", action="store_true", help="Skip automatic GitHub push after the run")
    args = parser.parse_args()

    global SHOW_FIGURES
    SHOW_FIGURES = bool(args.show)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(2)

    script_dir = Path(__file__).resolve().parent
    script_path = Path(__file__).resolve()
    original_excel_path = Path(args.excel).resolve() if args.excel else find_excel_file(script_dir, EXCEL_FILE)
    excel_path = make_excel_working_copy(original_excel_path)
    out_dir = script_dir / OUTPUT_FOLDER
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all_cases(excel_path)
    data = {lab: resample_uniform(df, RESAMPLE_DT) for lab, df in raw.items()}
    labels = sorted(data.keys())

    # For normalization, use all labels that are allowed to train in the selected split.
    if args.split == "within_case":
        norm_train_labels = labels
    else:
        norm_train_labels = [lab for lab in labels if (lab.startswith("Load_1") or lab.startswith("Load_2")) and "_Case_4_" not in lab]
    norm = make_normalization(data, norm_train_labels)

    all_rows = pd.concat([build_rows_for_experiment(lab, data[lab], norm) for lab in labels], ignore_index=True)
    train_rows, val_rows, test_rows = split_row_table(all_rows, args.split)

    print("\n====================================================")
    print("Improved LSTM/NARX dataset summary")
    print("====================================================")
    print(f"Excel file: {excel_path}")
    print(f"Output folder: {out_dir}")
    print(f"Split mode: {args.split}")
    print(f"Resampling dt: {RESAMPLE_DT} s")
    print(f"Sequence length: {SEQUENCE_LENGTH} samples = {SEQUENCE_LENGTH * RESAMPLE_DT:.4f} s")
    print(f"Input dimension per time step: {train_rows['X_window'].iloc[0].shape[1]}")
    print(f"Train windows: {len(train_rows)}")
    print(f"Validation windows: {len(val_rows)}")
    print(f"Test windows: {len(test_rows)}")
    print("====================================================\n")

    X_train, Yd_train = rows_to_tensors(train_rows)
    X_val, Yd_val = rows_to_tensors(val_rows)
    X_test, Yd_test = rows_to_tensors(test_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NARXLSTM(
        input_size=X_train.shape[2],
        hidden_size=NUM_HIDDEN_UNITS,
        output_size=2,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
        fc_hidden=FC_HIDDEN_UNITS,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, Yd_train),
        batch_size=min(BATCH_SIZE, len(X_train)),
        shuffle=True,
        drop_last=False,
    )
    X_val = X_val.to(device)
    Yd_val = Yd_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARN_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)
    loss_fn = nn.MSELoss()

    history = []
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_rmses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            yp = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_THRESHOLD)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_rmses.append(float(torch.sqrt(nn.functional.mse_loss(yp, yb)).detach().cpu()))

        model.eval()
        with torch.no_grad():
            yp_val = model(X_val)
            val_loss_t = loss_fn(yp_val, Yd_val)
            val_rmse = float(torch.sqrt(val_loss_t).detach().cpu())
            val_loss = float(val_loss_t.detach().cpu())
        scheduler.step(val_loss)

        train_loss = float(np.mean(train_losses))
        train_rmse = float(np.mean(train_rmses))
        history.append({
            "Epoch": epoch,
            "TrainingLoss": train_loss,
            "ValidationLoss": val_loss,
            "TrainingRMSE": train_rmse,
            "ValidationRMSE": val_rmse,
            "LearningRate": optimizer.param_groups[0]["lr"],
        })

        if val_rmse < best_val - 1e-7:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:4d} | Train RMSE {train_rmse:.6f} | Val RMSE {val_rmse:.6f}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best validation RMSE = {best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    predictions = {
        "Train": predict_teacher_forced(model, train_rows, norm, device),
        "Validation": predict_teacher_forced(model, val_rows, norm, device),
        "Test": predict_teacher_forced(model, test_rows, norm, device),
    }

    metrics = pd.concat([metric_rows(pred, split) for split, pred in predictions.items()], ignore_index=True)
    metrics.to_csv(out_dir / "metrics_summary.csv", index=False)
    for split, pred in predictions.items():
        pred.to_csv(out_dir / f"{split}_teacher_forced_predictions.csv", index=False)

    plot_training_history(history_df, fig_dir)
    plot_prediction_examples(predictions, metrics, fig_dir)
    plot_metric_summary(metrics, fig_dir)
    plot_parity(predictions, fig_dir)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_type": "NARX_LSTM_delta_output",
            "input_size": int(X_train.shape[2]),
            "hidden_units": NUM_HIDDEN_UNITS,
            "num_lstm_layers": NUM_LSTM_LAYERS,
            "dropout": DROPOUT,
            "fc_hidden_units": FC_HIDDEN_UNITS,
            "sequence_length": SEQUENCE_LENGTH,
            "resample_dt": RESAMPLE_DT,
            "outputs": TARGET_COLUMNS,
            "split_mode": args.split,
        },
        "normalization": norm,
        "metrics": metrics.to_dict(orient="records"),
    }, out_dir / "improved_chirp_lstm_narx_model.pt")

    with open(out_dir / "README_results.txt", "w", encoding="utf-8") as f:
        f.write("Improved Chirp LSTM/NARX Modeling Results\n")
        f.write("========================================\n\n")
        f.write("This model uses past measured outputs and predicts output increments.\n")
        f.write("This is a one-step-ahead NARX-style system-identification model.\n\n")
        f.write(f"Split mode: {args.split}\n")
        f.write(f"Sequence length: {SEQUENCE_LENGTH} samples = {SEQUENCE_LENGTH * RESAMPLE_DT:.4f} s\n")
        f.write(f"Resampling dt: {RESAMPLE_DT} s\n\n")
        f.write("Generated figures:\n")
        f.write("Fig01_Training_History: training and validation convergence.\n")
        f.write("Fig02_*_Prediction: representative true vs predicted trajectories.\n")
        f.write("Fig03_*_Metric_Summary: RMSE per experiment.\n")
        f.write("Fig04_*_Parity: predicted-vs-true plots.\n\n")
        f.write("Important interpretation:\n")
        f.write("The default within_case split verifies that the model can learn the measured trajectories.\n")
        f.write("For harder generalization to unseen loads/cases, rerun with --split leave_load3_out.\n")

    print("\nFinal split-level metrics")
    summary = metrics.groupby("Split")[["Disp_RMSE_mm", "Disp_R2", "Force_RMSE_N", "Force_R2"]].mean(numeric_only=True)
    print(summary.to_string())
    print(f"\nResults saved to: {out_dir}")
    print(f"Figures saved to: {fig_dir}")

    if DO_GITHUB_PUSH and not args.no_github:
        github_push(out_dir, script_path, original_excel_path)
    else:
        print("GitHub push skipped.")


if __name__ == "__main__":
    main()
    plt.close("all")
