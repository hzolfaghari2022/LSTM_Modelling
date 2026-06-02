#!/usr/bin/env python3
"""
Enhanced ESA COMSOL Single Actuator LSTM Modeling in Python

What is improved in this version
-------------------------------
1. The Excel file is always searched in the same folder as this script.
2. A more robust LSTM is used for noisy data:
   - stacked LSTM with 2 recurrent layers
   - dropout regularization
   - sliding-window training, which creates many training samples
   - optional moving-average smoothing for raw signals
3. The code generates a clean network architecture figure for the report.
4. The code saves only a small set of meaningful report-ready figures.
5. It writes a compact LaTeX metrics table and a summary text file.
6. It can still push the outputs to GitHub using SSH.

Learned mapping
---------------
current history  ->  displacement and coil force
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# User settings
# ============================================================
EXCEL_FILE = "ESA-COMSOL_Data_05_22_2026.xlsx"
FIG_FOLDER = "figures"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEQUENCE_LENGTH = 25
BATCH_SIZE = 32
MAX_EPOCHS = 800
INITIAL_LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRADIENT_THRESHOLD = 1.0

# Optional denoising to improve robustness against noise
USE_SMOOTHING = True
SMOOTHING_WINDOW = 5

# Improved architecture
NUM_LSTM_LAYERS = 2
NUM_HIDDEN_UNITS = 64
DROPOUT = 0.20
FC_HIDDEN_UNITS = 32

SHOW_FIGURES = False
DO_GITHUB_PUSH = True
REPO_SSH = "git@github.com:hzolfaghari2022/LSTM_Modelling.git"
REPO_NAME = "LSTM_Modelling"
BRANCH_NAME = "main"
TARGET_FOLDER_NAME = "ESA_Single_Actuator_LSTM_Model_Python_Enhanced"

RANDOM_SEED = 42

# ============================================================
# Utility functions
# ============================================================
def run_command(command: list[str], cwd: Path | None = None) -> Tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout


def clean_time_table(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    df = df.dropna().copy()
    df = df.sort_values("Time_s")
    return df.groupby("Time_s", as_index=False)[value_columns].mean()


def smooth_signal(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def save_figure(fig: plt.Figure, fig_folder: Path, base_name: str) -> None:
    fig_folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_folder / f"{base_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_folder / f"{base_name}.pdf", bbox_inches="tight")
    if SHOW_FIGURES:
        fig.show()
    else:
        plt.close(fig)


def normalize(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (data - mu) / sigma


def denormalize(data_norm: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return data_norm * sigma + mu


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error ** 2)))


def mae(error: np.ndarray) -> float:
    return float(np.mean(np.abs(error)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def write_metrics_table_tex(file_path: Path, metrics: dict[str, float]) -> None:
    with file_path.open("w", encoding="utf-8") as fid:
        fid.write("\\begin{table}[H]\n")
        fid.write("\\centering\n")
        fid.write("\\caption{Prediction performance of the enhanced single actuator LSTM model.}\n")
        fid.write("\\label{tab:lstm_prediction_metrics_enhanced}\n")
        fid.write("\\begin{tabular}{l c}\n")
        fid.write("\\hline\n")
        fid.write("Metric & Value \\\\ \n")
        fid.write("\\hline\n")
        fid.write(f"Displacement RMSE, training & {metrics['rmse_disp_train']:.6g} mm \\\\ \n")
        fid.write(f"Displacement RMSE, validation & {metrics['rmse_disp_val']:.6g} mm \\\\ \n")
        fid.write(f"Displacement RMSE, testing & {metrics['rmse_disp_test']:.6g} mm \\\\ \n")
        fid.write(f"Displacement MAE, testing & {metrics['mae_disp_test']:.6g} mm \\\\ \n")
        fid.write(f"Displacement $R^2$, testing & {metrics['r2_disp_test']:.6f} \\\\ \n")
        fid.write(f"Coil force RMSE, training & {metrics['rmse_force_train']:.6g} N \\\\ \n")
        fid.write(f"Coil force RMSE, validation & {metrics['rmse_force_val']:.6g} N \\\\ \n")
        fid.write(f"Coil force RMSE, testing & {metrics['rmse_force_test']:.6g} N \\\\ \n")
        fid.write(f"Coil force MAE, testing & {metrics['mae_force_test']:.6g} N \\\\ \n")
        fid.write(f"Coil force $R^2$, testing & {metrics['r2_force_test']:.6f} \\\\ \n")
        fid.write("\\hline\n")
        fid.write("\\end{tabular}\n")
        fid.write("\\end{table}\n")


def create_windows(X: np.ndarray, Y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, Y_list, end_index = [], [], []
    for i in range(seq_len - 1, len(X)):
        start = i - seq_len + 1
        X_list.append(X[start:i + 1])
        Y_list.append(Y[i])
        end_index.append(i)
    return np.stack(X_list), np.stack(Y_list), np.array(end_index)


def plot_network_architecture(fig_folder: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def add_box(x, y, w, h, text, fc="0.95"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=fc, edgecolor="black")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, fontweight="bold")

    def add_arrow(x1, y1, x2, y2):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=15, linewidth=2)
        ax.add_patch(arr)

    add_box(0.03, 0.32, 0.13, 0.34, "Input\nCurrent history\nsequence")
    add_box(0.22, 0.32, 0.14, 0.34, f"LSTM layer 1\n{NUM_HIDDEN_UNITS} hidden units")
    add_box(0.42, 0.32, 0.14, 0.34, f"LSTM layer 2\n{NUM_HIDDEN_UNITS} hidden units")
    add_box(0.62, 0.32, 0.12, 0.34, f"Fully connected\n{FC_HIDDEN_UNITS} units")
    add_box(0.79, 0.32, 0.16, 0.34, "Output\nDisplacement\nCoil force")

    add_arrow(0.16, 0.49, 0.22, 0.49)
    add_arrow(0.36, 0.49, 0.42, 0.49)
    add_arrow(0.56, 0.49, 0.62, 0.49)
    add_arrow(0.74, 0.49, 0.79, 0.49)

    ax.text(0.29, 0.16, "Dropout regularization", ha="center", va="center", fontsize=10)
    ax.text(0.49, 0.16, "Stacked memory for nonlinear dynamics", ha="center", va="center", fontsize=10)
    ax.text(0.80, 0.16, "Regression output", ha="center", va="center", fontsize=10)
    ax.set_title("Enhanced LSTM Architecture for Single ESA Actuator", fontsize=13, fontweight="bold")
    save_figure(fig, fig_folder, "Fig03_LSTM_Architecture")


# ============================================================
# Model
# ============================================================
class EnhancedSingleActuatorLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, output_size: int = 2,
                 num_layers: int = 2, dropout: float = 0.20, fc_hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        z = out[:, -1, :]
        z = self.fc1(z)
        z = self.relu(z)
        y = self.fc2(z)
        return y


# ============================================================
# Main
# ============================================================
def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    script_dir = Path(__file__).resolve().parent
    working_folder = script_dir
    fig_folder = working_folder / FIG_FOLDER
    fig_folder.mkdir(parents=True, exist_ok=True)

    # Generate the LSTM topology figure immediately. This figure does not need
    # the Excel data, so it will still be created even if the data path is wrong.
    plot_network_architecture(fig_folder)

    # Search for the Excel file in useful locations. This avoids errors when
    # PowerShell is open in a different folder or when the code is inside a Code folder.
    possible_excel_paths = [
        script_dir / EXCEL_FILE,
        script_dir.parent / EXCEL_FILE,
        script_dir.parent.parent / EXCEL_FILE,
        Path.cwd() / EXCEL_FILE,
    ]
    excel_path = None
    for candidate in possible_excel_paths:
        if candidate.is_file():
            excel_path = candidate
            break

    if excel_path is None:
        searched = "\n".join(str(p) for p in possible_excel_paths)
        raise FileNotFoundError(
            f"The Excel file was not found. The code searched these locations:\n{searched}\n\n"
            f"Put {EXCEL_FILE} in the same folder as this Python file, or one folder above it."
        )

    print(f"Using Excel file: {excel_path}")

    # Read Excel
    D = pd.read_excel(excel_path, sheet_name="Displacement", skiprows=4, usecols="A:B")
    F = pd.read_excel(excel_path, sheet_name="Force", skiprows=1, usecols="A:C")
    C = pd.read_excel(excel_path, sheet_name="Current", skiprows=1, usecols="A:B")

    D.columns = ["Time_s", "Displacement_mm"]
    F.columns = ["Time_s", "CoilForce_N", "WeightLoad_N"]
    C.columns = ["Time_s", "Current_A"]

    for df in (D, F, C):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    D = clean_time_table(D, ["Displacement_mm"])
    F = clean_time_table(F, ["CoilForce_N", "WeightLoad_N"])
    C = clean_time_table(C, ["Current_A"])

    t = F["Time_s"].to_numpy()
    current_on_grid = np.interp(t, C["Time_s"].to_numpy(), C["Current_A"].to_numpy())
    disp_on_grid = np.interp(t, D["Time_s"].to_numpy(), D["Displacement_mm"].to_numpy())
    coil_force = F["CoilForce_N"].to_numpy()
    weight_load = F["WeightLoad_N"].to_numpy()
    net_force = coil_force - weight_load

    if USE_SMOOTHING:
        current_for_model = smooth_signal(current_on_grid, SMOOTHING_WINDOW)
        disp_for_model = smooth_signal(disp_on_grid, SMOOTHING_WINDOW)
        force_for_model = smooth_signal(coil_force, SMOOTHING_WINDOW)
    else:
        current_for_model = current_on_grid.copy()
        disp_for_model = disp_on_grid.copy()
        force_for_model = coil_force.copy()

    # Basic summary figure
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(t, current_on_grid, linewidth=1.6, label="Raw current")
    if USE_SMOOTHING:
        axes[0].plot(t, current_for_model, "--", linewidth=1.4, label="Smoothed current")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("Single Actuator ESA Dataset Overview")
    axes[0].legend()
    axes[1].plot(t, coil_force, linewidth=1.6, label="Raw coil force")
    if USE_SMOOTHING:
        axes[1].plot(t, force_for_model, "--", linewidth=1.4, label="Smoothed coil force")
    axes[1].set_ylabel("Force (N)")
    axes[1].legend()
    axes[2].plot(t, disp_on_grid, linewidth=1.6, label="Raw displacement")
    if USE_SMOOTHING:
        axes[2].plot(t, disp_for_model, "--", linewidth=1.4, label="Smoothed displacement")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Displacement (mm)")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_figure(fig, fig_folder, "Fig01_Data_Overview")

    cleaned_data = pd.DataFrame({
        "Time_s": t,
        "Current_A": current_on_grid,
        "CurrentForModel_A": current_for_model,
        "CoilForce_N": coil_force,
        "CoilForceForModel_N": force_for_model,
        "WeightLoad_N": weight_load,
        "NetForce_N": net_force,
        "Displacement_mm": disp_on_grid,
        "DisplacementForModel_mm": disp_for_model,
    })
    cleaned_data.to_csv(working_folder / "ESA_cleaned_synchronized_data_enhanced.csv", index=False)

    # Prepare learning data
    Xraw = current_for_model.reshape(-1, 1)
    Yraw = np.column_stack([disp_for_model, force_for_model])

    num_samples = len(t)
    n_train_time = max(SEQUENCE_LENGTH + 10, int(math.floor(TRAIN_RATIO * num_samples)))
    n_val_time = max(5, int(math.floor(VAL_RATIO * num_samples)))
    n_test_start = min(num_samples - 1, n_train_time + n_val_time)

    mu_x = Xraw[:n_train_time].mean(axis=0)
    sig_x = Xraw[:n_train_time].std(axis=0)
    sig_x[sig_x == 0] = 1.0
    mu_y = Yraw[:n_train_time].mean(axis=0)
    sig_y = Yraw[:n_train_time].std(axis=0)
    sig_y[sig_y == 0] = 1.0

    Xnorm = normalize(Xraw, mu_x, sig_x)
    Ynorm = normalize(Yraw, mu_y, sig_y)

    Xseq, Yseq, end_idx = create_windows(Xnorm, Ynorm, SEQUENCE_LENGTH)
    t_seq = t[end_idx]
    true_disp_seq = disp_on_grid[end_idx]
    true_force_seq = coil_force[end_idx]

    train_mask = end_idx < n_train_time
    val_mask = (end_idx >= n_train_time) & (end_idx < n_test_start)
    test_mask = end_idx >= n_test_start

    X_train = Xseq[train_mask]
    Y_train = Yseq[train_mask]
    X_val = Xseq[val_mask]
    Y_val = Yseq[val_mask]
    X_test = Xseq[test_mask]
    Y_test = Yseq[test_mask]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(t, Xnorm[:, 0], linewidth=1.5)
    axes[1].plot(t, Ynorm[:, 0], linewidth=1.5)
    axes[2].plot(t, Ynorm[:, 1], linewidth=1.5)
    for ax in axes:
        ax.axvline(t[n_train_time - 1], color="k", linestyle=":", linewidth=1.2)
        ax.axvline(t[n_test_start - 1], color="r", linestyle=":", linewidth=1.2)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Norm. current")
    axes[0].set_title("Normalized Input and Outputs with Train/Validation/Test Split")
    axes[1].set_ylabel("Norm. disp.")
    axes[2].set_ylabel("Norm. force")
    axes[2].set_xlabel("Time (s)")
    save_figure(fig, fig_folder, "Fig02_Train_Test_Normalized_Sequences")

    # Architecture figure was already generated before data loading.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedSingleActuatorLSTM(
        input_size=1,
        hidden_size=NUM_HIDDEN_UNITS,
        output_size=2,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
        fc_hidden=FC_HIDDEN_UNITS,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)),
        batch_size=min(BATCH_SIZE, len(X_train)),
        shuffle=False,
    )

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)
    X_all_tensor = torch.tensor(Xseq, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARN_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(delta=1.0)

    history = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        train_rmses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
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
            yp_val = model(X_val_tensor)
            val_loss = float(loss_fn(yp_val, Y_val_tensor).detach().cpu())
            val_rmse = float(torch.sqrt(nn.functional.mse_loss(yp_val, Y_val_tensor)).detach().cpu())

        train_loss_epoch = float(np.mean(train_losses))
        train_rmse_epoch = float(np.mean(train_rmses))

        history.append({
            "Epoch": epoch,
            "TrainingLoss": train_loss_epoch,
            "ValidationLoss": val_loss,
            "TrainingRMSE": train_rmse_epoch,
            "ValidationRMSE": val_rmse,
        })

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0 or epoch == MAX_EPOCHS:
            print(f"Epoch {epoch:4d} | Train RMSE {train_rmse_epoch:.6f} | Val RMSE {val_rmse:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    training_history = pd.DataFrame(history)
    training_history.to_csv(working_folder / "ESA_single_actuator_LSTM_training_history_enhanced.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    axes[0].plot(training_history["Epoch"], training_history["TrainingRMSE"], linewidth=1.8, label="Training RMSE")
    axes[0].plot(training_history["Epoch"], training_history["ValidationRMSE"], "--", linewidth=1.8, label="Validation RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Enhanced LSTM Training History")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(training_history["Epoch"], training_history["TrainingLoss"], linewidth=1.8, label="Training loss")
    axes[1].plot(training_history["Epoch"], training_history["ValidationLoss"], "--", linewidth=1.8, label="Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Huber loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    save_figure(fig, fig_folder, "Fig04_Training_History")

    model.eval()
    with torch.no_grad():
        Ypred_norm = model(X_all_tensor).cpu().numpy()
    Ypred = denormalize(Ypred_norm, mu_y, sig_y)

    pred_disp = Ypred[:, 0]
    pred_force = Ypred[:, 1]
    disp_err = true_disp_seq - pred_disp
    force_err = true_force_seq - pred_force

    metrics = {
        "rmse_disp_train": rmse(disp_err[train_mask]),
        "rmse_disp_val": rmse(disp_err[val_mask]),
        "rmse_disp_test": rmse(disp_err[test_mask]),
        "mae_disp_test": mae(disp_err[test_mask]),
        "r2_disp_test": r2_score(true_disp_seq[test_mask], pred_disp[test_mask]),
        "rmse_force_train": rmse(force_err[train_mask]),
        "rmse_force_val": rmse(force_err[val_mask]),
        "rmse_force_test": rmse(force_err[test_mask]),
        "mae_force_test": mae(force_err[test_mask]),
        "r2_force_test": r2_score(true_force_seq[test_mask], pred_force[test_mask]),
    }

    print("\nEnhanced LSTM metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.8g}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t_seq, true_disp_seq, linewidth=1.8, label="COMSOL")
    axes[0].plot(t_seq, pred_disp, "--", linewidth=1.8, label="LSTM")
    axes[0].axvline(t[n_train_time - 1], color="k", linestyle=":", linewidth=1.2)
    axes[0].axvline(t[n_test_start - 1], color="r", linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title(
        f"Prediction Accuracy Over Time, displacement test RMSE = {metrics['rmse_disp_test']:.4g} mm, "
        f"R2 = {metrics['r2_disp_test']:.4f}"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(t_seq, true_force_seq, linewidth=1.8, label="COMSOL")
    axes[1].plot(t_seq, pred_force, "--", linewidth=1.8, label="LSTM")
    axes[1].axvline(t[n_train_time - 1], color="k", linestyle=":", linewidth=1.2)
    axes[1].axvline(t[n_test_start - 1], color="r", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Coil force (N)")
    axes[1].set_title(
        f"Force test RMSE = {metrics['rmse_force_test']:.4g} N, R2 = {metrics['r2_force_test']:.4f}"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    save_figure(fig, fig_folder, "Fig05_Prediction_Accuracy_Time_Response")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(true_disp_seq[test_mask], pred_disp[test_mask], s=28)
    mn, mx = min(true_disp_seq[test_mask].min(), pred_disp[test_mask].min()), max(true_disp_seq[test_mask].max(), pred_disp[test_mask].max())
    axes[0].plot([mn, mx], [mn, mx], "k--", linewidth=1.5)
    axes[0].set_xlabel("COMSOL displacement (mm)")
    axes[0].set_ylabel("LSTM displacement (mm)")
    axes[0].set_title(f"Displacement parity, R2 = {metrics['r2_disp_test']:.4f}")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(true_force_seq[test_mask], pred_force[test_mask], s=28)
    mn, mx = min(true_force_seq[test_mask].min(), pred_force[test_mask].min()), max(true_force_seq[test_mask].max(), pred_force[test_mask].max())
    axes[1].plot([mn, mx], [mn, mx], "k--", linewidth=1.5)
    axes[1].set_xlabel("COMSOL coil force (N)")
    axes[1].set_ylabel("LSTM coil force (N)")
    axes[1].set_title(f"Force parity, R2 = {metrics['r2_force_test']:.4f}")
    axes[1].grid(True, alpha=0.3)
    save_figure(fig, fig_folder, "Fig06_Test_Parity_Plots")

    results = pd.DataFrame({
        "Time_s": t_seq,
        "TrueDisplacement_mm": true_disp_seq,
        "PredictedDisplacement_mm": pred_disp,
        "DisplacementError_mm": disp_err,
        "TrueCoilForce_N": true_force_seq,
        "PredictedCoilForce_N": pred_force,
        "CoilForceError_N": force_err,
        "IsTrainingSample": train_mask,
        "IsValidationSample": val_mask,
        "IsTestingSample": test_mask,
    })
    results.to_csv(working_folder / "ESA_single_actuator_LSTM_results_enhanced.csv", index=False)
    pd.DataFrame([metrics]).to_csv(working_folder / "ESA_single_actuator_LSTM_metrics_enhanced.csv", index=False)
    write_metrics_table_tex(working_folder / "ESA_LSTM_metrics_table_enhanced.tex", metrics)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": 1,
            "hidden_size": NUM_HIDDEN_UNITS,
            "output_size": 2,
            "num_lstm_layers": NUM_LSTM_LAYERS,
            "dropout": DROPOUT,
            "fc_hidden": FC_HIDDEN_UNITS,
            "sequence_length": SEQUENCE_LENGTH,
        },
        "normalization": {
            "mu_x": mu_x,
            "sig_x": sig_x,
            "mu_y": mu_y,
            "sig_y": sig_y,
        },
        "metrics": metrics,
    }, working_folder / "ESA_single_actuator_LSTM_model_enhanced.pt")

    with (working_folder / "ESA_single_actuator_LSTM_architecture_summary.txt").open("w", encoding="utf-8") as fid:
        fid.write("Enhanced Single Actuator LSTM Architecture Summary\n")
        fid.write("===============================================\n\n")
        fid.write("Conceptual layer structure:\n")
        fid.write("1. Sequence input layer with 1 feature: current\n")
        fid.write(f"2. LSTM recurrent layer 1 with {NUM_HIDDEN_UNITS} hidden units\n")
        fid.write(f"3. LSTM recurrent layer 2 with {NUM_HIDDEN_UNITS} hidden units\n")
        fid.write(f"4. Fully connected hidden layer with {FC_HIDDEN_UNITS} units\n")
        fid.write("5. ReLU activation layer\n")
        fid.write("6. Fully connected output layer with 2 outputs\n\n")
        fid.write("Why two LSTM layers are used:\n")
        fid.write("- The first recurrent layer learns short term temporal behavior and local memory effects.\n")
        fid.write("- The second recurrent layer refines higher level nonlinear dynamic patterns.\n")
        fid.write("- A single LSTM layer may be too limited when the actuator response contains delays, transient memory, and noisy behavior.\n")
        fid.write("- More than two recurrent layers would likely be unnecessary for this small dataset and could increase overfitting risk.\n\n")
        fid.write(f"Sequence length: {SEQUENCE_LENGTH}\n")
        fid.write(f"Training ratio: {TRAIN_RATIO}\n")
        fid.write(f"Validation ratio: {VAL_RATIO}\n")
        fid.write(f"Smoothing used: {USE_SMOOTHING}\n")
        fid.write(f"Smoothing window: {SMOOTHING_WINDOW}\n")

    print("Enhanced single actuator LSTM training completed successfully.")
    print("Important report figures saved in the figures folder:")
    print(f"  {fig_folder / 'Fig01_Data_Overview.png'}")
    print(f"  {fig_folder / 'Fig02_Train_Test_Normalized_Sequences.png'}")
    print(f"  {fig_folder / 'Fig03_LSTM_Architecture.png'}")
    print(f"  {fig_folder / 'Fig04_Training_History.png'}")
    print(f"  {fig_folder / 'Fig05_Prediction_Accuracy_Time_Response.png'}")
    print(f"  {fig_folder / 'Fig06_Test_Parity_Plots.png'}")

    if DO_GITHUB_PUSH:
        github_push(working_folder)


# ============================================================
# GitHub push
# ============================================================
def github_push(source_folder: Path) -> None:
    print("====================================================")
    print("Preparing GitHub SSH push for LSTM_Modelling repository")
    print("====================================================")

    github_root = Path(tempfile.gettempdir()) / "Python_GitHub_Repos"
    repo_folder = github_root / REPO_NAME
    target_folder = repo_folder / TARGET_FOLDER_NAME
    github_root.mkdir(parents=True, exist_ok=True)

    if not repo_folder.exists():
        status, out = run_command(["git", "clone", REPO_SSH, str(repo_folder)])
        if status != 0:
            raise RuntimeError(f"Git clone failed using SSH.\n{out}")

    run_command(["git", "remote", "set-url", "origin", REPO_SSH], cwd=repo_folder)
    run_command(["git", "branch", "-M", BRANCH_NAME], cwd=repo_folder)
    run_command(["git", "pull", "origin", BRANCH_NAME, "--rebase"], cwd=repo_folder)

    target_folder.mkdir(parents=True, exist_ok=True)
    patterns = ["*.py", "*.xlsx", "*.csv", "*.pt", "*.txt", "*.tex", "*.pdf", "*.png"]
    for pattern in patterns:
        for file in source_folder.glob(pattern):
            if file.is_file():
                shutil.copy2(file, target_folder / file.name)

    source_fig = source_folder / FIG_FOLDER
    target_fig = target_folder / FIG_FOLDER
    if source_fig.is_dir():
        if target_fig.exists():
            shutil.rmtree(target_fig)
        shutil.copytree(source_fig, target_fig)

    run_command(["git", "config", "user.name", "Hussein Zolfaghari"], cwd=repo_folder)
    run_command(["git", "config", "user.email", "h.zolfaghari2015@gmail.com"], cwd=repo_folder)
    status, out = run_command(["git", "add", "."], cwd=repo_folder)
    if status != 0:
        raise RuntimeError(f"Git add failed:\n{out}")
    status, _ = run_command(["git", "diff", "--cached", "--quiet"], cwd=repo_folder)
    if status != 0:
        run_command(["git", "commit", "-m", "Update enhanced Python ESA LSTM model"], cwd=repo_folder)
    status, out = run_command(["git", "push", "-u", "origin", BRANCH_NAME], cwd=repo_folder)
    if status != 0:
        raise RuntimeError(f"Git push failed using SSH.\n{out}")
    print("Files pushed successfully to GitHub.")


if __name__ == "__main__":
    main()
