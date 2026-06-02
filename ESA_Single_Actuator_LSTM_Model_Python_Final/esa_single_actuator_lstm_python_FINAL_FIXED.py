#!/usr/bin/env python3
"""
Final fixed Python implementation for ESA COMSOL single actuator LSTM modeling.

Main improvements compared with the previous version
----------------------------------------------------
1. The Excel file is searched in the script folder and its parent folders.
2. The LSTM no longer uses one single train segment only. It uses overlapping
   current history windows, which creates many supervised learning samples from
   the one available COMSOL trajectory.
3. The train, validation, and test sets are selected from those windows using a
   fixed random seed. This improves the ability to evaluate interpolation within
   the available experiment.
4. The architecture is kept honest and simple: current history -> stacked LSTM
   -> fully connected regression head -> displacement and force.
5. A report-ready LSTM topology figure is generated automatically.
6. Only a compact set of meaningful figures is saved for the supervisor report.
7. The report metrics table is written as a LaTeX file, but the report also
   compiles without it.
8. Results can be pushed to GitHub by SSH.

Important scientific note
-------------------------
With only one COMSOL experiment, this code demonstrates trajectory learning and
within-trajectory interpolation. For true generalization, additional excitation
profiles and loading cases are still needed.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
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

# Learning settings
SEQUENCE_LENGTH = 30                 # number of past current samples used by LSTM
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
BATCH_SIZE = 32
MAX_EPOCHS = 1200
INITIAL_LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-6
GRADIENT_THRESHOLD = 1.0
PATIENCE = 180                       # early stopping patience

# Signal treatment
USE_SMOOTHING_FOR_TRAINING = False   # keep False to compare with raw COMSOL outputs
SMOOTHING_WINDOW = 5

# Architecture
NUM_LSTM_LAYERS = 2
NUM_HIDDEN_UNITS = 96
DROPOUT = 0.10
FC_HIDDEN_UNITS = 48

# Display and GitHub settings
SHOW_FIGURES = False
DO_GITHUB_PUSH = True
REPO_SSH = "git@github.com:hzolfaghari2022/LSTM_Modelling.git"
REPO_NAME = "LSTM_Modelling"
BRANCH_NAME = "main"
TARGET_FOLDER_NAME = "ESA_Single_Actuator_LSTM_Model_Python_Final"

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


def find_excel_file(script_dir: Path) -> Path:
    """Search the script folder and several parent folders for the Excel file."""
    candidates = [script_dir / EXCEL_FILE]
    for parent in script_dir.parents:
        candidates.append(parent / EXCEL_FILE)
    candidates.append(Path.cwd() / EXCEL_FILE)

    for path in candidates:
        if path.is_file():
            return path

    msg = ["The Excel file was not found.", "Searched these paths:"]
    msg.extend(str(p) for p in candidates[:8])
    msg.append(f"Put {EXCEL_FILE} beside this Python file or one folder above it.")
    raise FileNotFoundError("\n".join(msg))


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


def nrmse_percent(error: np.ndarray, truth: np.ndarray) -> float:
    dynamic_range = float(np.max(truth) - np.min(truth))
    if dynamic_range == 0:
        return float("nan")
    return 100.0 * rmse(error) / dynamic_range


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def create_windows(X: np.ndarray, Y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create overlapping windows. The output is the response at the window end."""
    X_list, Y_list, end_idx = [], [], []
    for i in range(seq_len - 1, len(X)):
        start = i - seq_len + 1
        X_list.append(X[start:i + 1])
        Y_list.append(Y[i])
        end_idx.append(i)
    return np.stack(X_list), np.stack(Y_list), np.array(end_idx)


def split_indices(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(math.floor(TRAIN_FRACTION * n))
    n_val = int(math.floor(VALIDATION_FRACTION * n))

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val:])
    return train_idx, val_idx, test_idx


def write_metrics_table_tex(file_path: Path, metrics: dict[str, float]) -> None:
    with file_path.open("w", encoding="utf-8") as fid:
        fid.write("\\begin{table}[H]\n")
        fid.write("\\centering\n")
        fid.write("\\caption{Prediction performance of the final single actuator LSTM model.}\n")
        fid.write("\\label{tab:lstm_prediction_metrics_final}\n")
        fid.write("\\begin{tabular}{l c}\n")
        fid.write("\\toprule\n")
        fid.write("Metric & Value \\\\ \n")
        fid.write("\\midrule\n")
        fid.write(f"Displacement RMSE, training & {metrics['rmse_disp_train']:.6g} mm \\\\ \n")
        fid.write(f"Displacement RMSE, validation & {metrics['rmse_disp_val']:.6g} mm \\\\ \n")
        fid.write(f"Displacement RMSE, testing & {metrics['rmse_disp_test']:.6g} mm \\\\ \n")
        fid.write(f"Displacement MAE, testing & {metrics['mae_disp_test']:.6g} mm \\\\ \n")
        fid.write(f"Displacement NRMSE, testing & {metrics['nrmse_disp_test']:.4f}\\% \\\\ \n")
        fid.write(f"Displacement $R^2$, testing & {metrics['r2_disp_test']:.6f} \\\\ \n")
        fid.write(f"Coil force RMSE, training & {metrics['rmse_force_train']:.6g} N \\\\ \n")
        fid.write(f"Coil force RMSE, validation & {metrics['rmse_force_val']:.6g} N \\\\ \n")
        fid.write(f"Coil force RMSE, testing & {metrics['rmse_force_test']:.6g} N \\\\ \n")
        fid.write(f"Coil force MAE, testing & {metrics['mae_force_test']:.6g} N \\\\ \n")
        fid.write(f"Coil force NRMSE, testing & {metrics['nrmse_force_test']:.4f}\\% \\\\ \n")
        fid.write(f"Coil force $R^2$, testing & {metrics['r2_force_test']:.6f} \\\\ \n")
        fid.write("\\bottomrule\n")
        fid.write("\\end{tabular}\n")
        fid.write("\\end{table}\n")


def plot_lstm_topology(fig_folder: Path) -> None:
    """Create a supervisor-ready topology figure matching the style requested by the user."""
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def vertical_box(x, w, text, color="white", edge="steelblue"):
        box = FancyBboxPatch((x, 0.45), w, 3.4, boxstyle="round,pad=0.02", facecolor=color,
                             edgecolor=edge, linewidth=2.2)
        ax.add_patch(box)
        ax.text(x + w / 2, 2.15, text, ha="center", va="center", rotation=90,
                fontsize=13, fontweight="bold")

    def wide_box(x, w, title, subtitle):
        box = FancyBboxPatch((x, 0.45), w, 3.4, boxstyle="round,pad=0.02", facecolor="white",
                             edgecolor="steelblue", linewidth=2.2)
        ax.add_patch(box)
        ax.text(x + w / 2, 3.35, title, ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(x + w / 2, 3.00, subtitle, ha="center", va="center", fontsize=11)
        # small neural graph
        xs_in = x + 0.35
        xs_mid = x + w / 2
        xs_out = x + w - 0.35
        ys = [2.55, 2.15, 1.75]
        for y in ys:
            ax.add_patch(Circle((xs_in, y), 0.08, color="steelblue"))
            ax.add_patch(Circle((xs_mid, y), 0.20, facecolor="gold", edgecolor="orange", linewidth=1.2))
            ax.add_patch(Circle((xs_out, y), 0.08, color="red"))
        for y1 in ys:
            for y2 in ys:
                ax.add_patch(FancyArrowPatch((xs_in + 0.08, y1), (xs_mid - 0.20, y2),
                                             arrowstyle="->", mutation_scale=8, linewidth=0.7, color="black"))
                ax.add_patch(FancyArrowPatch((xs_mid + 0.20, y1), (xs_out - 0.08, y2),
                                             arrowstyle="->", mutation_scale=8, linewidth=0.7, color="black"))
        for y in ys:
            ax.add_patch(FancyArrowPatch((xs_mid, y + 0.20), (xs_mid + 0.01, y + 0.55),
                                         connectionstyle="arc3,rad=0.8", arrowstyle="->",
                                         mutation_scale=8, linewidth=0.8, color="black"))

    def arrow(x1, x2):
        ax.add_patch(FancyArrowPatch((x1, 2.15), (x2, 2.15), arrowstyle="->", mutation_scale=16,
                                     linewidth=1.6, color="black"))

    ax.text(0.35, 2.15, "Input\ncurrent\nsequence", ha="center", va="center", rotation=90,
            fontsize=14, fontweight="bold", color="darkred")
    arrow(0.70, 1.00)

    vertical_box(1.00, 0.90, "Preprocessing\nnormalization")
    arrow(1.90, 2.15)

    vertical_box(2.15, 0.90, "Sliding\nwindow")
    arrow(3.05, 3.35)

    wide_box(3.35, 2.10, "LSTM Layer 1", "64 or 96 hidden units")
    arrow(5.45, 5.75)

    vertical_box(5.75, 0.80, "Dropout\nregularization")
    arrow(6.55, 6.85)

    wide_box(6.85, 2.10, "LSTM Layer 2", "stacked temporal memory")
    arrow(8.95, 9.25)

    vertical_box(9.25, 0.95, "Fully connected\n+ ReLU")
    arrow(10.20, 10.50)

    vertical_box(10.50, 1.15, "Regression\noutput layer")
    arrow(11.65, 11.95)

    ax.text(12.35, 2.15, "Predicted\ndisplacement\nand force", ha="center", va="center", rotation=90,
            fontsize=14, fontweight="bold", color="darkred")

    # dashed outer frame
    outer = FancyBboxPatch((0.85, 0.25), 10.95, 3.85, boxstyle="round,pad=0.03",
                           facecolor="none", edgecolor="0.65", linewidth=2.0, linestyle="--")
    ax.add_patch(outer)
    ax.set_title("Topology of the Stacked LSTM Regression Model", fontsize=16, fontweight="bold", pad=10)
    save_figure(fig, fig_folder, "Fig03_LSTM_Topology")


# ============================================================
# Model
# ============================================================
class StackedLSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 96, output_size: int = 2,
                 num_layers: int = 2, dropout: float = 0.10, fc_hidden: int = 48):
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
            nn.ReLU(),
            nn.Linear(fc_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


# ============================================================
# Main workflow
# ============================================================
def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    script_dir = Path(__file__).resolve().parent
    working_folder = script_dir
    excel_path = find_excel_file(script_dir)
    fig_folder = working_folder / FIG_FOLDER
    fig_folder.mkdir(exist_ok=True)

    # Create topology figure immediately so it is always available for report.
    plot_lstm_topology(fig_folder)

    # ------------------------------------------------------------
    # Load and synchronize data
    # ------------------------------------------------------------
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
    current = np.interp(t, C["Time_s"].to_numpy(), C["Current_A"].to_numpy())
    displacement = np.interp(t, D["Time_s"].to_numpy(), D["Displacement_mm"].to_numpy())
    force = F["CoilForce_N"].to_numpy()
    weight_load = F["WeightLoad_N"].to_numpy()
    net_force = force - weight_load

    current_model = smooth_signal(current, SMOOTHING_WINDOW) if USE_SMOOTHING_FOR_TRAINING else current.copy()
    displacement_model = smooth_signal(displacement, SMOOTHING_WINDOW) if USE_SMOOTHING_FOR_TRAINING else displacement.copy()
    force_model = smooth_signal(force, SMOOTHING_WINDOW) if USE_SMOOTHING_FOR_TRAINING else force.copy()

    print("\n====================================================")
    print("ESA single actuator dataset summary")
    print("====================================================")
    print(f"Excel file used: {excel_path}")
    print(f"Number of synchronized samples: {len(t)}")
    print(f"Time range: {t.min():.6f} s to {t.max():.6f} s")
    print(f"Mean sampling time: {np.mean(np.diff(t)) * 1000:.3f} ms")
    print(f"Current range: {current.min():.6f} A to {current.max():.6f} A")
    print(f"Displacement range: {displacement.min():.6f} mm to {displacement.max():.6f} mm")
    print(f"Coil force range: {force.min():.6f} N to {force.max():.6f} N")
    print("====================================================")

    # Figure 1: data overview
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    axes[0].plot(t, current, linewidth=1.7, label="Current")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("Synchronized Single Actuator Dataset")
    axes[1].plot(t, force, linewidth=1.7, label="Coil force")
    axes[1].plot(t, weight_load, "--", linewidth=1.3, label="Weight/load")
    axes[1].set_ylabel("Force (N)")
    axes[1].legend()
    axes[2].plot(t, displacement, linewidth=1.7, label="Displacement")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Displacement (mm)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_figure(fig, fig_folder, "Fig01_Data_Overview")

    pd.DataFrame({
        "Time_s": t,
        "Current_A": current,
        "CurrentForModel_A": current_model,
        "CoilForce_N": force,
        "CoilForceForModel_N": force_model,
        "WeightLoad_N": weight_load,
        "NetForce_N": net_force,
        "Displacement_mm": displacement,
        "DisplacementForModel_mm": displacement_model,
    }).to_csv(working_folder / "ESA_cleaned_synchronized_data_final.csv", index=False)

    # ------------------------------------------------------------
    # Create window dataset
    # ------------------------------------------------------------
    X_raw = current_model.reshape(-1, 1)
    Y_raw = np.column_stack([displacement_model, force_model])

    # Normalize from full trajectory for within trajectory reconstruction.
    # This is appropriate here because the goal is fitting the one available COMSOL response.
    mu_x = X_raw.mean(axis=0)
    sig_x = X_raw.std(axis=0)
    sig_x[sig_x == 0] = 1.0
    mu_y = Y_raw.mean(axis=0)
    sig_y = Y_raw.std(axis=0)
    sig_y[sig_y == 0] = 1.0

    X_norm = normalize(X_raw, mu_x, sig_x)
    Y_norm = normalize(Y_raw, mu_y, sig_y)
    X_seq, Y_seq, end_idx = create_windows(X_norm, Y_norm, SEQUENCE_LENGTH)

    train_idx, val_idx, test_idx = split_indices(len(X_seq))

    # Figure 2: split visualization
    split_label = np.full(len(X_seq), 0)
    split_label[val_idx] = 1
    split_label[test_idx] = 2
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    axes[0].plot(t, X_norm[:, 0], linewidth=1.5)
    axes[1].plot(t, Y_norm[:, 0], linewidth=1.5)
    axes[2].plot(t, Y_norm[:, 1], linewidth=1.5)
    axes[0].set_title("Normalized LSTM Input and Outputs")
    axes[0].set_ylabel("Norm. current")
    axes[1].set_ylabel("Norm. disp.")
    axes[2].set_ylabel("Norm. force")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_figure(fig, fig_folder, "Fig02_Normalized_Sequences")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackedLSTMRegressor(
        input_size=1,
        hidden_size=NUM_HIDDEN_UNITS,
        output_size=2,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
        fc_hidden=FC_HIDDEN_UNITS,
    ).to(device)

    X_train = torch.tensor(X_seq[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y_seq[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_seq[val_idx], dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_seq[val_idx], dtype=torch.float32).to(device)
    X_all = torch.tensor(X_seq, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=min(BATCH_SIZE, len(X_train)),
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARN_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    history = []
    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses, batch_rmses = [], []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            yp = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_THRESHOLD)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            batch_rmses.append(float(torch.sqrt(nn.functional.mse_loss(yp, yb)).detach().cpu()))

        model.eval()
        with torch.no_grad():
            yp_val = model(X_val)
            val_mse = nn.functional.mse_loss(yp_val, Y_val)
            val_rmse = float(torch.sqrt(val_mse).detach().cpu())
            val_loss = float(val_mse.detach().cpu())

        train_loss = float(np.mean(batch_losses))
        train_rmse = float(np.mean(batch_rmses))

        history.append({
            "Epoch": epoch,
            "TrainingLoss": train_loss,
            "ValidationLoss": val_loss,
            "TrainingRMSE": train_rmse,
            "ValidationRMSE": val_rmse,
        })

        if val_rmse < best_val_rmse - 1e-7:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Train RMSE {train_rmse:.6f} | Val RMSE {val_rmse:.6f}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best validation RMSE: {best_val_rmse:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_history = pd.DataFrame(history)
    training_history.to_csv(working_folder / "ESA_single_actuator_LSTM_training_history_final.csv", index=False)

    # Figure 4: training history
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.5), sharex=True)
    axes[0].plot(training_history["Epoch"], training_history["TrainingRMSE"], linewidth=1.7, label="Training RMSE")
    axes[0].plot(training_history["Epoch"], training_history["ValidationRMSE"], "--", linewidth=1.7, label="Validation RMSE")
    axes[0].set_ylabel("Normalized RMSE")
    axes[0].set_title("LSTM Training History")
    axes[0].legend()
    axes[1].plot(training_history["Epoch"], training_history["TrainingLoss"], linewidth=1.7, label="Training loss")
    axes[1].plot(training_history["Epoch"], training_history["ValidationLoss"], "--", linewidth=1.7, label="Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE loss")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_figure(fig, fig_folder, "Fig04_Training_History")

    # Prediction on all windows
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_all).detach().cpu().numpy()
    y_pred = denormalize(y_pred_norm, mu_y, sig_y)

    t_pred = t[end_idx]
    true_disp = displacement[end_idx]
    true_force = force[end_idx]
    pred_disp = y_pred[:, 0]
    pred_force = y_pred[:, 1]

    disp_err = true_disp - pred_disp
    force_err = true_force - pred_force

    train_mask = np.zeros(len(X_seq), dtype=bool)
    val_mask = np.zeros(len(X_seq), dtype=bool)
    test_mask = np.zeros(len(X_seq), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    metrics = {
        "rmse_disp_train": rmse(disp_err[train_mask]),
        "rmse_disp_val": rmse(disp_err[val_mask]),
        "rmse_disp_test": rmse(disp_err[test_mask]),
        "mae_disp_test": mae(disp_err[test_mask]),
        "nrmse_disp_test": nrmse_percent(disp_err[test_mask], true_disp[test_mask]),
        "r2_disp_test": r2_score(true_disp[test_mask], pred_disp[test_mask]),
        "rmse_force_train": rmse(force_err[train_mask]),
        "rmse_force_val": rmse(force_err[val_mask]),
        "rmse_force_test": rmse(force_err[test_mask]),
        "mae_force_test": mae(force_err[test_mask]),
        "nrmse_force_test": nrmse_percent(force_err[test_mask], true_force[test_mask]),
        "r2_force_test": r2_score(true_force[test_mask], pred_force[test_mask]),
    }

    print("\nFinal LSTM metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.8g}")

    # Figure 5: prediction accuracy over time
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7), sharex=True)
    axes[0].plot(t_pred, true_disp, linewidth=1.8, label="COMSOL")
    axes[0].plot(t_pred, pred_disp, "--", linewidth=1.8, label="LSTM")
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title(
        f"Displacement Prediction, test RMSE = {metrics['rmse_disp_test']:.4g} mm, "
        f"R2 = {metrics['r2_disp_test']:.4f}"
    )
    axes[0].legend()
    axes[1].plot(t_pred, true_force, linewidth=1.8, label="COMSOL")
    axes[1].plot(t_pred, pred_force, "--", linewidth=1.8, label="LSTM")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Coil force (N)")
    axes[1].set_title(
        f"Force Prediction, test RMSE = {metrics['rmse_force_test']:.4g} N, "
        f"R2 = {metrics['r2_force_test']:.4f}"
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_figure(fig, fig_folder, "Fig05_Prediction_Accuracy_Time_Response")

    # Figure 6: parity plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].scatter(true_disp[test_mask], pred_disp[test_mask], s=30)
    mn = min(true_disp[test_mask].min(), pred_disp[test_mask].min())
    mx = max(true_disp[test_mask].max(), pred_disp[test_mask].max())
    axes[0].plot([mn, mx], [mn, mx], "k--", linewidth=1.4)
    axes[0].set_xlabel("COMSOL displacement (mm)")
    axes[0].set_ylabel("LSTM displacement (mm)")
    axes[0].set_title(f"Displacement parity, R2 = {metrics['r2_disp_test']:.4f}")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(true_force[test_mask], pred_force[test_mask], s=30)
    mn = min(true_force[test_mask].min(), pred_force[test_mask].min())
    mx = max(true_force[test_mask].max(), pred_force[test_mask].max())
    axes[1].plot([mn, mx], [mn, mx], "k--", linewidth=1.4)
    axes[1].set_xlabel("COMSOL coil force (N)")
    axes[1].set_ylabel("LSTM coil force (N)")
    axes[1].set_title(f"Force parity, R2 = {metrics['r2_force_test']:.4f}")
    axes[1].grid(True, alpha=0.3)
    save_figure(fig, fig_folder, "Fig06_Test_Parity_Plots")

    # Save outputs
    pd.DataFrame({
        "Time_s": t_pred,
        "TrueDisplacement_mm": true_disp,
        "PredictedDisplacement_mm": pred_disp,
        "DisplacementError_mm": disp_err,
        "TrueCoilForce_N": true_force,
        "PredictedCoilForce_N": pred_force,
        "CoilForceError_N": force_err,
        "IsTrainingSample": train_mask,
        "IsValidationSample": val_mask,
        "IsTestingSample": test_mask,
    }).to_csv(working_folder / "ESA_single_actuator_LSTM_results_final.csv", index=False)

    pd.DataFrame([metrics]).to_csv(working_folder / "ESA_single_actuator_LSTM_metrics_final.csv", index=False)
    write_metrics_table_tex(working_folder / "ESA_LSTM_metrics_table_final.tex", metrics)

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
        "normalization": {"mu_x": mu_x, "sig_x": sig_x, "mu_y": mu_y, "sig_y": sig_y},
        "metrics": metrics,
    }, working_folder / "ESA_single_actuator_LSTM_model_final.pt")

    with (working_folder / "ESA_single_actuator_LSTM_summary_final.txt").open("w", encoding="utf-8") as fid:
        fid.write("Final Single Actuator ESA LSTM Model Summary\n")
        fid.write("===========================================\n\n")
        fid.write("Input: current history window\n")
        fid.write("Outputs: displacement and coil force\n")
        fid.write(f"Sequence length: {SEQUENCE_LENGTH}\n")
        fid.write(f"LSTM layers: {NUM_LSTM_LAYERS}\n")
        fid.write(f"Hidden units per LSTM layer: {NUM_HIDDEN_UNITS}\n")
        fid.write(f"Fully connected hidden units: {FC_HIDDEN_UNITS}\n")
        fid.write(f"Dropout: {DROPOUT}\n")
        fid.write(f"Smoothing used for training: {USE_SMOOTHING_FOR_TRAINING}\n\n")
        fid.write("Scientific note: With only one COMSOL experiment, this model demonstrates within-trajectory learning. Additional input profiles are required for generalization.\n\n")
        for key, value in metrics.items():
            fid.write(f"{key}: {value:.8g}\n")

    print("\nFinal fixed Python LSTM workflow completed successfully.")
    print(f"Figures saved to: {fig_folder}")

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
    status, out = run_command(["git", "pull", "origin", BRANCH_NAME, "--rebase"], cwd=repo_folder)
    if status != 0:
        print("Warning: git pull had an issue:")
        print(out)

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
        status, out = run_command(["git", "commit", "-m", "Update final Python ESA LSTM model"], cwd=repo_folder)
        if status != 0:
            raise RuntimeError(f"Git commit failed:\n{out}")

    status, out = run_command(["git", "push", "-u", "origin", BRANCH_NAME], cwd=repo_folder)
    if status != 0:
        raise RuntimeError(f"Git push failed using SSH.\n{out}")
    print("Files pushed successfully to GitHub.")


if __name__ == "__main__":
    main()
