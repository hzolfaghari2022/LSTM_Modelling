#!/usr/bin/env python3
"""
ESA COMSOL Single Actuator Data Visualization and LSTM Modeling in Python

Purpose
-------
1. Read the COMSOL spreadsheet for one ESA actuator.
2. Plot each signal with physically meaningful axes.
3. Train a first LSTM dynamic model for one actuator.
4. Predict displacement and coil force from current history.
5. Save cleaned data, figures, model, prediction results, and accuracy metrics.
6. Optionally push the project outputs to GitHub using SSH.

Important
---------
This is for one actuator only.
The LSTM learns:

    current history  ->  displacement response and coil force response

Expected Excel file in the same folder:
    ESA-COMSOL_Data_05_22_2026.xlsx

GitHub repository:
    git@github.com:hzolfaghari2022/LSTM_Modelling.git
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# User settings
# ============================================================

EXCEL_FILE = "ESA-COMSOL_Data_05_22_2026.xlsx"
FIG_FOLDER = "figures"

TRAIN_LSTM = True
TRAIN_RATIO = 0.70
NUM_HIDDEN_UNITS = 64
MAX_EPOCHS = 600
INITIAL_LEARN_RATE = 1e-3
GRADIENT_THRESHOLD = 1.0

DO_GITHUB_PUSH = True
REPO_SSH = "git@github.com:hzolfaghari2022/LSTM_Modelling.git"
REPO_NAME = "LSTM_Modelling"
BRANCH_NAME = "main"
TARGET_FOLDER_NAME = "ESA_Single_Actuator_LSTM_Model_Python"

RANDOM_SEED = 42

# Figure display settings
# True: each generated figure is opened on screen after it is saved.
# False: figures are only saved to the figures folder.
SHOW_FIGURES = True

# Seconds to briefly pause after opening each figure.
# Increase this if you want more time to visually check each figure while the code runs.
FIGURE_PAUSE_SECONDS = 0.8

# False keeps all figure windows open until the end of the script.
# True opens each figure briefly, saves it, and then closes it automatically.
CLOSE_FIGURES_AFTER_SHOW = False


# ============================================================
# Utility functions
# ============================================================

def run_command(command: list[str], cwd: Path | None = None) -> Tuple[int, str]:
    """Run a shell command and return status and combined output."""
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
    """Remove missing rows, sort by time, and average duplicate time stamps."""
    df = df.dropna().copy()
    df = df.sort_values("Time_s")
    grouped = df.groupby("Time_s", as_index=False)[value_columns].mean()
    return grouped


def save_figure(fig: plt.Figure, fig_folder: Path, base_name: str) -> None:
    """Save figure as PNG, PDF, and editable SVG."""
    fig_folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_folder / f"{base_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_folder / f"{base_name}.pdf", bbox_inches="tight")
    fig.savefig(fig_folder / f"{base_name}.svg", bbox_inches="tight")


def finalize_figure(fig: plt.Figure, fig_folder: Path, base_name: str) -> None:
    """Save a figure and optionally open it on screen."""
    save_figure(fig, fig_folder, base_name)

    if SHOW_FIGURES:
        try:
            plt.show(block=False)
            plt.pause(FIGURE_PAUSE_SECONDS)
        except Exception as exc:
            print(f"Warning: could not display figure {base_name}: {exc}")

    if CLOSE_FIGURES_AFTER_SHOW or not SHOW_FIGURES:
        plt.close(fig)


def make_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    fig_folder: Path,
    file_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, linewidth=1.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finalize_figure(fig, fig_folder, file_name)


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
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def write_metrics_table_tex(file_path: Path, metrics: dict[str, float]) -> None:
    """Write a compact LaTeX table for Overleaf."""
    with file_path.open("w", encoding="utf-8") as fid:
        fid.write("\\begin{table}[H]\n")
        fid.write("\\centering\n")
        fid.write("\\caption{Prediction performance of the single actuator LSTM model.}\n")
        fid.write("\\label{tab:lstm_prediction_metrics_python}\n")
        fid.write("\\begin{tabular}{l c}\n")
        fid.write("\\hline\n")
        fid.write("Metric & Value \\\\\n")
        fid.write("\\hline\n")
        fid.write(f"Displacement RMSE, training & {metrics['rmse_disp_train']:.6g} mm \\\\\n")
        fid.write(f"Displacement RMSE, testing & {metrics['rmse_disp_test']:.6g} mm \\\\\n")
        fid.write(f"Displacement MAE, training & {metrics['mae_disp_train']:.6g} mm \\\\\n")
        fid.write(f"Displacement MAE, testing & {metrics['mae_disp_test']:.6g} mm \\\\\n")
        fid.write(f"Displacement $R^2$, testing & {metrics['r2_disp_test']:.6f} \\\\\n")
        fid.write(f"Coil force RMSE, training & {metrics['rmse_force_train']:.6g} N \\\\\n")
        fid.write(f"Coil force RMSE, testing & {metrics['rmse_force_test']:.6g} N \\\\\n")
        fid.write(f"Coil force MAE, training & {metrics['mae_force_train']:.6g} N \\\\\n")
        fid.write(f"Coil force MAE, testing & {metrics['mae_force_test']:.6g} N \\\\\n")
        fid.write(f"Coil force $R^2$, testing & {metrics['r2_force_test']:.6f} \\\\\n")
        fid.write("\\hline\n")
        fid.write("\\end{tabular}\n")
        fid.write("\\end{table}\n")


def write_figure_inventory(fig_folder: Path) -> None:
    """Create a text list and LaTeX include file for Overleaf."""
    png_files = sorted(fig_folder.glob("Fig*.png"))

    list_file = fig_folder / "ESA_LSTM_Report_Figure_List.txt"
    tex_file = fig_folder / "ESA_LSTM_Report_Figure_Includes.tex"

    with list_file.open("w", encoding="utf-8") as fid:
        fid.write("Report ready figures saved for Overleaf\n")
        fid.write("=======================================\n\n")
        for k, file in enumerate(png_files, start=1):
            fid.write(f"{k:02d}. {file.name}\n")

    with tex_file.open("w", encoding="utf-8") as fid:
        fid.write("% Auto generated LaTeX figure include file.\n")
        fid.write("% Copy selected blocks into your Overleaf report.\n\n")
        for file in png_files:
            label = file.stem.lower().replace("_", ":")
            caption = file.stem.replace("_", " ")
            fid.write("\\begin{figure}[H]\n")
            fid.write("    \\centering\n")
            fid.write(f"    \\includegraphics[width=0.92\\textwidth]{{figures/{file.name}}}\n")
            fid.write(f"    \\caption{{{caption}.}}\n")
            fid.write(f"    \\label{{fig:{label}}}\n")
            fid.write("\\end{figure}\n\n")


# ============================================================
# LSTM model
# ============================================================

class SingleActuatorLSTM(nn.Module):
    """Sequence to sequence LSTM for one actuator."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, output_size: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        z = self.fc1(lstm_out)
        z = self.relu(z)
        y = self.fc2(z)
        return y


# ============================================================
# Main workflow
# ============================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Use the folder where this Python file is saved, not the folder where
    # PowerShell is currently open. This prevents FileNotFoundError when the
    # script is called from another directory.
    script_dir = Path(__file__).resolve().parent
    working_folder = script_dir
    excel_path = working_folder / EXCEL_FILE
    fig_folder = working_folder / FIG_FOLDER
    fig_folder.mkdir(parents=True, exist_ok=True)

    if not excel_path.is_file():
        raise FileNotFoundError(
            f"The Excel file was not found: {excel_path}\n"
            f"Put {EXCEL_FILE} in the same folder as this Python file:\n"
            f"{script_dir}"
        )

    # ------------------------------------------------------------
    # Read raw data
    # ------------------------------------------------------------
    # MATLAB readtable ranges:
    # Displacement: A5:B565
    # Force:        A2:C519
    # Current:      A2:B519
    D = pd.read_excel(excel_path, sheet_name="Displacement", skiprows=4, usecols="A:B")
    F = pd.read_excel(excel_path, sheet_name="Force", skiprows=1, usecols="A:C")
    C = pd.read_excel(excel_path, sheet_name="Current", skiprows=1, usecols="A:B")

    D.columns = ["Time_s", "Displacement_mm"]
    F.columns = ["Time_s", "CoilForce_N", "WeightLoad_N"]
    C.columns = ["Time_s", "Current_A"]

    # Convert to numeric and drop non numeric rows
    for df in (D, F, C):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------
    # Clean duplicate time stamps
    # ------------------------------------------------------------
    D = clean_time_table(D, ["Displacement_mm"])
    F = clean_time_table(F, ["CoilForce_N", "WeightLoad_N"])
    C = clean_time_table(C, ["Current_A"])

    # ------------------------------------------------------------
    # Synchronize signals on force time grid
    # ------------------------------------------------------------
    t = F["Time_s"].to_numpy()
    current_on_grid = np.interp(t, C["Time_s"].to_numpy(), C["Current_A"].to_numpy())
    disp_on_grid = np.interp(t, D["Time_s"].to_numpy(), D["Displacement_mm"].to_numpy())
    coil_force = F["CoilForce_N"].to_numpy()
    weight_load = F["WeightLoad_N"].to_numpy()
    net_force = coil_force - weight_load

    dt_vec = np.diff(t)
    dt_mean = float(np.mean(dt_vec))

    print("\n====================================================")
    print("ESA single actuator dataset summary")
    print("====================================================")
    print(f"Number of synchronized samples: {len(t)}")
    print(f"Time range: {np.min(t):.6f} s to {np.max(t):.6f} s")
    print(f"Mean sampling time: {dt_mean:.6f} s")
    print(f"Mean sampling time: {dt_mean * 1000:.3f} ms")
    print(f"Current range: {np.min(current_on_grid):.6f} A to {np.max(current_on_grid):.6f} A")
    print(f"Displacement range: {np.min(disp_on_grid):.6f} mm to {np.max(disp_on_grid):.6f} mm")
    print(f"Coil force range: {np.min(coil_force):.6f} N to {np.max(coil_force):.6f} N")
    print("====================================================")

    # ------------------------------------------------------------
    # Basic data visualization figures
    # ------------------------------------------------------------
    make_line_plot(
        C["Time_s"].to_numpy(),
        C["Current_A"].to_numpy(),
        "Time (s)",
        "Coil current (A)",
        "Input Current Profile for Single ESA Actuator",
        fig_folder,
        "Fig01_Current_Time",
    )

    make_line_plot(
        F["Time_s"].to_numpy(),
        F["CoilForce_N"].to_numpy(),
        "Time (s)",
        "Coil force (N)",
        "Electromagnetic Coil Force Response",
        fig_folder,
        "Fig02_CoilForce_Time",
    )

    make_line_plot(
        F["Time_s"].to_numpy(),
        F["WeightLoad_N"].to_numpy(),
        "Time (s)",
        "Weight/load force (N)",
        "Weight/Load Force",
        fig_folder,
        "Fig03_Weight_Time",
    )

    make_line_plot(
        t,
        net_force,
        "Time (s)",
        "Net force estimate (N)",
        "Net Force Estimate: Coil Force Minus Weight/Load",
        fig_folder,
        "Fig04_NetForce_Time",
    )

    make_line_plot(
        D["Time_s"].to_numpy(),
        D["Displacement_mm"].to_numpy(),
        "Time (s)",
        "Z displacement (mm)",
        "Z Displacement Response of Single ESA Actuator",
        fig_folder,
        "Fig05_Displacement_Time",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, current_on_grid, label="Current (A)", linewidth=1.5)
    ax.plot(t, coil_force, label="Coil force (N)", linewidth=1.5)
    ax.plot(t, disp_on_grid, label="Displacement (mm)", linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal value with original units")
    ax.set_title("Synchronized Single Actuator Signals")
    ax.legend()
    finalize_figure(fig, fig_folder, "Fig06_SynchronizedSignals_Time")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(1000 * t, current_on_grid, label="Current (A)", linewidth=1.5)
    ax.plot(1000 * t, coil_force, label="Coil force (N)", linewidth=1.5)
    ax.plot(1000 * t, disp_on_grid, label="Displacement (mm)", linewidth=1.5)
    ax.set_xlim(0, 30)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Signal value with original units")
    ax.set_title("Trigger and Early Transient Region, 0 to 30 ms")
    ax.legend()
    finalize_figure(fig, fig_folder, "Fig07_TriggerZoom_0_30ms")

    make_line_plot(
        current_on_grid,
        coil_force,
        "Coil current (A)",
        "Coil force (N)",
        "Coil Force Versus Input Current",
        fig_folder,
        "Fig08_Force_Current",
    )

    make_line_plot(
        current_on_grid,
        disp_on_grid,
        "Coil current (A)",
        "Z displacement (mm)",
        "Displacement Versus Input Current",
        fig_folder,
        "Fig09_Displacement_Current",
    )

    make_line_plot(
        disp_on_grid,
        coil_force,
        "Z displacement (mm)",
        "Coil force (N)",
        "Force Versus Displacement",
        fig_folder,
        "Fig10_Force_Displacement",
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(t, current_on_grid, linewidth=1.8)
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("Report Summary of Single Actuator Dataset")
    axes[1].plot(t, coil_force, linewidth=1.8, label="Coil force")
    axes[1].plot(t, weight_load, "--", linewidth=1.4, label="Weight/load")
    axes[1].set_ylabel("Force (N)")
    axes[1].legend()
    axes[2].plot(t, disp_on_grid, linewidth=1.8)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Displacement (mm)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    finalize_figure(fig, fig_folder, "Fig15_Report_Data_Summary")

    # ------------------------------------------------------------
    # Save cleaned data
    # ------------------------------------------------------------
    cleaned_data = pd.DataFrame({
        "Time_s": t,
        "Current_A": current_on_grid,
        "CoilForce_N": coil_force,
        "WeightLoad_N": weight_load,
        "NetForce_N": net_force,
        "Displacement_mm": disp_on_grid,
    })
    cleaned_data.to_csv(working_folder / "ESA_cleaned_synchronized_data.csv", index=False)

    print("Data visualization completed successfully. Figures were saved in the figures folder.")

    if not TRAIN_LSTM:
        write_figure_inventory(fig_folder)
        if SHOW_FIGURES and not CLOSE_FIGURES_AFTER_SHOW:
            print("All figures were saved and opened. Close the figure windows to finish the script.")
            plt.show(block=True)
        return

    # ------------------------------------------------------------
    # LSTM dynamic model for one actuator
    # ------------------------------------------------------------
    print("\n====================================================")
    print("Training single actuator LSTM model")
    print("====================================================")

    # Input sequence: current, shape [T, 1]
    # Output sequence: displacement and coil force, shape [T, 2]
    Xraw = current_on_grid.reshape(-1, 1)
    Yraw = np.column_stack([disp_on_grid, coil_force])

    num_samples = len(t)
    n_train = max(10, int(np.floor(TRAIN_RATIO * num_samples)))
    n_train = min(n_train, num_samples - 5)

    train_slice = slice(0, n_train)
    test_slice = slice(n_train, num_samples)

    X_train_raw = Xraw[train_slice]
    Y_train_raw = Yraw[train_slice]
    X_test_raw = Xraw[test_slice]
    Y_test_raw = Yraw[test_slice]

    # Normalize using training data only
    mu_x = X_train_raw.mean(axis=0)
    sig_x = X_train_raw.std(axis=0)
    sig_x[sig_x == 0] = 1.0

    mu_y = Y_train_raw.mean(axis=0)
    sig_y = Y_train_raw.std(axis=0)
    sig_y[sig_y == 0] = 1.0

    X_train = normalize(X_train_raw, mu_x, sig_x)
    Y_train = normalize(Y_train_raw, mu_y, sig_y)
    X_test = normalize(X_test_raw, mu_x, sig_x)
    Y_test = normalize(Y_test_raw, mu_y, sig_y)
    X_full = normalize(Xraw, mu_x, sig_x)

    # Report figure: normalized input and outputs
    Y_full_norm = normalize(Yraw, mu_y, sig_y)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(t, X_full[:, 0], linewidth=1.5)
    axes[0].axvline(t[n_train], color="k", linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("Normalized current")
    axes[0].set_title("Normalized LSTM Input and Outputs")

    axes[1].plot(t, Y_full_norm[:, 0], linewidth=1.5)
    axes[1].axvline(t[n_train], color="k", linestyle=":", linewidth=1.2)
    axes[1].set_ylabel("Normalized displacement")

    axes[2].plot(t, Y_full_norm[:, 1], linewidth=1.5)
    axes[2].axvline(t[n_train], color="k", linestyle=":", linewidth=1.2)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Normalized force")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    finalize_figure(fig, fig_folder, "Fig16_LSTM_Normalized_Sequences")

    # Report figure: workflow
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.5", facecolor="0.95", edgecolor="black")
    ax.text(0.12, 0.58, "Input sequence\nCurrent history\nI(1), I(2), ..., I(k)",
            ha="center", va="center", fontsize=12, fontweight="bold", bbox=box)
    ax.text(0.50, 0.58, "LSTM dynamic memory\nLearns time dependent\nactuator behavior",
            ha="center", va="center", fontsize=12, fontweight="bold", bbox=box)
    ax.text(0.88, 0.58, "Output sequence\nDisplacement z(k)\nCoil force F(k)",
            ha="center", va="center", fontsize=12, fontweight="bold", bbox=box)
    ax.annotate("", xy=(0.37, 0.58), xytext=(0.24, 0.58), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.76, 0.58), xytext=(0.62, 0.58), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(
        0.50,
        0.17,
        f"Architecture: sequence input + LSTM ({NUM_HIDDEN_UNITS} hidden units) + fully connected layers + regression output",
        ha="center",
        va="center",
        fontsize=11,
    )
    finalize_figure(fig, fig_folder, "Fig17_LSTM_Model_Workflow")

    # PyTorch tensors, batch size is 1 because this is one time series experiment
    X_train_tensor = torch.tensor(X_train[None, :, :], dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train[None, :, :], dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test[None, :, :], dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test[None, :, :], dtype=torch.float32)
    X_full_tensor = torch.tensor(X_full[None, :, :], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleActuatorLSTM(
        input_size=1,
        hidden_size=NUM_HIDDEN_UNITS,
        output_size=2,
    ).to(device)

    X_train_tensor = X_train_tensor.to(device)
    Y_train_tensor = Y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    Y_test_tensor = Y_test_tensor.to(device)
    X_full_tensor = X_full_tensor.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARN_RATE)
    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        y_train_pred = model(X_train_tensor)
        train_loss = loss_fn(y_train_pred, Y_train_tensor)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_THRESHOLD)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test_tensor)
            val_loss = loss_fn(y_test_pred, Y_test_tensor)

        train_rmse = float(torch.sqrt(train_loss).detach().cpu())
        val_rmse = float(torch.sqrt(val_loss).detach().cpu())

        history.append({
            "Epoch": epoch,
            "Iteration": epoch,
            "TrainingLoss": float(train_loss.detach().cpu()),
            "ValidationLoss": float(val_loss.detach().cpu()),
            "TrainingRMSE": train_rmse,
            "ValidationRMSE": val_rmse,
            "BaseLearnRate": INITIAL_LEARN_RATE,
        })

        if epoch == 1 or epoch % 25 == 0 or epoch == MAX_EPOCHS:
            print(
                f"Epoch {epoch:4d} | "
                f"Training RMSE {train_rmse:.6f} | "
                f"Validation RMSE {val_rmse:.6f} | "
                f"Training Loss {float(train_loss.detach().cpu()):.6e} | "
                f"Validation Loss {float(val_loss.detach().cpu()):.6e}"
            )

    training_history = pd.DataFrame(history)
    training_history.to_csv(working_folder / "ESA_single_actuator_LSTM_training_history.csv", index=False)

    # Full prediction
    model.eval()
    with torch.no_grad():
        y_pred_full_norm = model(X_full_tensor).detach().cpu().numpy()[0]

    y_pred_full = denormalize(y_pred_full_norm, mu_y, sig_y)

    pred_displacement_mm = y_pred_full[:, 0]
    pred_coil_force_n = y_pred_full[:, 1]

    true_displacement_mm = disp_on_grid
    true_coil_force_n = coil_force

    displacement_error_mm = true_displacement_mm - pred_displacement_mm
    coil_force_error_n = true_coil_force_n - pred_coil_force_n

    train_mask = np.zeros(num_samples, dtype=bool)
    test_mask = np.zeros(num_samples, dtype=bool)
    train_mask[:n_train] = True
    test_mask[n_train:] = True

    metrics = {
        "rmse_disp_train": rmse(displacement_error_mm[train_mask]),
        "rmse_disp_test": rmse(displacement_error_mm[test_mask]),
        "rmse_force_train": rmse(coil_force_error_n[train_mask]),
        "rmse_force_test": rmse(coil_force_error_n[test_mask]),
        "mae_disp_train": mae(displacement_error_mm[train_mask]),
        "mae_disp_test": mae(displacement_error_mm[test_mask]),
        "mae_force_train": mae(coil_force_error_n[train_mask]),
        "mae_force_test": mae(coil_force_error_n[test_mask]),
        "r2_disp_test": r2_score(true_displacement_mm[test_mask], pred_displacement_mm[test_mask]),
        "r2_force_test": r2_score(true_coil_force_n[test_mask], pred_coil_force_n[test_mask]),
    }

    print("\nLSTM metrics for single actuator model:")
    print(f"Displacement RMSE train: {metrics['rmse_disp_train']:.6g} mm")
    print(f"Displacement RMSE test : {metrics['rmse_disp_test']:.6g} mm")
    print(f"Coil force RMSE train  : {metrics['rmse_force_train']:.6g} N")
    print(f"Coil force RMSE test   : {metrics['rmse_force_test']:.6g} N")
    print(f"Displacement R2 test   : {metrics['r2_disp_test']:.6g}")
    print(f"Coil force R2 test     : {metrics['r2_force_test']:.6g}")

    # Training history figures
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(training_history["Iteration"], training_history["TrainingRMSE"], linewidth=1.8, label="Training RMSE")
    ax.plot(training_history["Iteration"], training_history["ValidationRMSE"], "--", linewidth=1.8, label="Validation RMSE")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.set_title("LSTM Training and Validation RMSE History")
    ax.legend()
    finalize_figure(fig, fig_folder, "Fig20_LSTM_Training_RMSE_History")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(training_history["Iteration"], training_history["TrainingLoss"], linewidth=1.8, label="Training loss")
    ax.plot(training_history["Iteration"], training_history["ValidationLoss"], "--", linewidth=1.8, label="Validation loss")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("LSTM Training and Validation Loss History")
    ax.legend()
    finalize_figure(fig, fig_folder, "Fig21_LSTM_Training_Loss_History")

    # Prediction figures
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, true_displacement_mm, linewidth=1.8, label="COMSOL displacement")
    ax.plot(t, pred_displacement_mm, "--", linewidth=1.8, label="LSTM prediction")
    ax.axvline(t[n_train], color="k", linestyle=":", linewidth=1.2, label="Train/test split")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z displacement (mm)")
    ax.set_title("Single Actuator LSTM Model: Displacement Prediction")
    ax.legend()
    finalize_figure(fig, fig_folder, "Fig11_LSTM_Displacement_Prediction")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, true_coil_force_n, linewidth=1.8, label="COMSOL coil force")
    ax.plot(t, pred_coil_force_n, "--", linewidth=1.8, label="LSTM prediction")
    ax.axvline(t[n_train], color="k", linestyle=":", linewidth=1.2, label="Train/test split")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coil force (N)")
    ax.set_title("Single Actuator LSTM Model: Coil Force Prediction")
    ax.legend()
    finalize_figure(fig, fig_folder, "Fig12_LSTM_Force_Prediction")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, displacement_error_mm, linewidth=1.6)
    axes[0].axvline(t[n_train], color="k", linestyle=":", linewidth=1.2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Error (mm)")
    axes[0].set_title("Displacement Prediction Error: COMSOL minus LSTM")

    axes[1].plot(t, coil_force_error_n, linewidth=1.6)
    axes[1].axvline(t[n_train], color="k", linestyle=":", linewidth=1.2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (N)")
    axes[1].set_title("Coil Force Prediction Error: COMSOL minus LSTM")
    finalize_figure(fig, fig_folder, "Fig13_LSTM_Prediction_Errors")

    zoom_mask = t <= 0.03
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(1000 * t[zoom_mask], true_displacement_mm[zoom_mask], linewidth=1.7, label="COMSOL displacement")
    ax1.plot(1000 * t[zoom_mask], pred_displacement_mm[zoom_mask], "--", linewidth=1.7, label="LSTM displacement")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Displacement (mm)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(1000 * t[zoom_mask], current_on_grid[zoom_mask], ":", linewidth=1.7, label="Current input")
    ax2.set_ylabel("Current (A)")
    ax1.set_title("Early Transient: Current Trigger and LSTM Displacement Prediction")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    finalize_figure(fig, fig_folder, "Fig14_LSTM_TriggerZoom")

    # Compact accuracy figures
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, true_displacement_mm, linewidth=1.8, label="COMSOL")
    axes[0].plot(t, pred_displacement_mm, "--", linewidth=1.8, label="LSTM")
    axes[0].axvline(t[n_train], color="k", linestyle=":", linewidth=1.2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title(
        f"Displacement Prediction, Test RMSE = {metrics['rmse_disp_test']:.4g} mm, "
        f"R2 = {metrics['r2_disp_test']:.4f}"
    )
    axes[0].legend()

    axes[1].plot(t, true_coil_force_n, linewidth=1.8, label="COMSOL")
    axes[1].plot(t, pred_coil_force_n, "--", linewidth=1.8, label="LSTM")
    axes[1].axvline(t[n_train], color="k", linestyle=":", linewidth=1.2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Coil force (N)")
    axes[1].set_title(
        f"Force Prediction, Test RMSE = {metrics['rmse_force_test']:.4g} N, "
        f"R2 = {metrics['r2_force_test']:.4f}"
    )
    axes[1].legend()
    finalize_figure(fig, fig_folder, "Fig22_LSTM_Accuracy_Time_Response")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(true_displacement_mm[test_mask], pred_displacement_mm[test_mask], s=30)
    min_d = min(true_displacement_mm[test_mask].min(), pred_displacement_mm[test_mask].min())
    max_d = max(true_displacement_mm[test_mask].max(), pred_displacement_mm[test_mask].max())
    axes[0].plot([min_d, max_d], [min_d, max_d], "k--", linewidth=1.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("COMSOL displacement (mm)")
    axes[0].set_ylabel("LSTM displacement (mm)")
    axes[0].set_title(f"Displacement parity, R2 = {metrics['r2_disp_test']:.4f}")

    axes[1].scatter(true_coil_force_n[test_mask], pred_coil_force_n[test_mask], s=30)
    min_f = min(true_coil_force_n[test_mask].min(), pred_coil_force_n[test_mask].min())
    max_f = max(true_coil_force_n[test_mask].max(), pred_coil_force_n[test_mask].max())
    axes[1].plot([min_f, max_f], [min_f, max_f], "k--", linewidth=1.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("COMSOL coil force (N)")
    axes[1].set_ylabel("LSTM coil force (N)")
    axes[1].set_title(f"Force parity, R2 = {metrics['r2_force_test']:.4f}")
    finalize_figure(fig, fig_folder, "Fig23_LSTM_Accuracy_Parity_Plots")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(["RMSE", "MAE"], [metrics["rmse_disp_test"], metrics["mae_disp_test"]])
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Displacement error (mm)")
    axes[0].set_title("Displacement Test Error")

    axes[1].bar(["RMSE", "MAE"], [metrics["rmse_force_test"], metrics["mae_force_test"]])
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_ylabel("Force error (N)")
    axes[1].set_title("Force Test Error")
    finalize_figure(fig, fig_folder, "Fig24_LSTM_Test_Accuracy_Metrics")

    # Save results
    lstm_results = pd.DataFrame({
        "Time_s": t,
        "Current_A": current_on_grid,
        "TrueDisplacement_mm": true_displacement_mm,
        "PredictedDisplacement_mm": pred_displacement_mm,
        "DisplacementError_mm": displacement_error_mm,
        "TrueCoilForce_N": true_coil_force_n,
        "PredictedCoilForce_N": pred_coil_force_n,
        "CoilForceError_N": coil_force_error_n,
        "IsTrainingSample": train_mask,
        "IsTestingSample": test_mask,
    })
    lstm_results.to_csv(working_folder / "ESA_single_actuator_LSTM_results.csv", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(working_folder / "ESA_single_actuator_LSTM_metrics.csv", index=False)
    write_metrics_table_tex(working_folder / "ESA_single_actuator_LSTM_metrics_table.tex", metrics)

    model_package = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": 1,
            "hidden_size": NUM_HIDDEN_UNITS,
            "output_size": 2,
        },
        "normalization": {
            "mu_x": mu_x,
            "sig_x": sig_x,
            "mu_y": mu_y,
            "sig_y": sig_y,
            "input_name": "Current_A",
            "output_names": ["Displacement_mm", "CoilForce_N"],
        },
        "metrics": metrics,
    }
    torch.save(model_package, working_folder / "ESA_single_actuator_LSTM_model.pt")

    with (working_folder / "ESA_single_actuator_LSTM_summary.txt").open("w", encoding="utf-8") as fid:
        fid.write("Single Actuator ESA LSTM Model Summary\n")
        fid.write("=====================================\n\n")
        fid.write("Input: current history\n")
        fid.write("Outputs: displacement and coil force\n")
        fid.write(f"Training ratio: {TRAIN_RATIO:.2f}\n")
        fid.write(f"Hidden units: {NUM_HIDDEN_UNITS}\n")
        fid.write(f"Max epochs: {MAX_EPOCHS}\n\n")
        for key, value in metrics.items():
            fid.write(f"{key}: {value:.8g}\n")

    write_figure_inventory(fig_folder)

    print("Single actuator LSTM model training completed successfully.")
    print("LSTM model saved as ESA_single_actuator_LSTM_model.pt")
    print("LSTM prediction results saved as ESA_single_actuator_LSTM_results.csv")

    if SHOW_FIGURES and not CLOSE_FIGURES_AFTER_SHOW:
        print("All figures were saved and opened. Close the figure windows to continue to GitHub push or finish the script.")
        plt.show(block=True)

    if DO_GITHUB_PUSH:
        github_push(working_folder)


# ============================================================
# GitHub SSH push
# ============================================================

def github_push(source_folder: Path) -> None:
    """Push current project outputs to GitHub using SSH."""
    print("====================================================")
    print("Preparing GitHub SSH push for LSTM_Modelling repository")
    print("====================================================")

    github_root = Path(tempfile.gettempdir()) / "Python_GitHub_Repos"
    repo_folder = github_root / REPO_NAME
    target_folder = repo_folder / TARGET_FOLDER_NAME

    github_root.mkdir(parents=True, exist_ok=True)

    if not repo_folder.exists():
        print("Local repository does not exist. Cloning with SSH...")
        status, out = run_command(["git", "clone", REPO_SSH, str(repo_folder)])
        if status != 0:
            raise RuntimeError(
                "Git clone failed using SSH.\n"
                "Test SSH with: ssh -T git@github.com\n\n"
                f"Git message:\n{out}"
            )
    else:
        print("Local repository already exists.")

    status, out = run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_folder)
    if status != 0:
        raise RuntimeError(f"The local folder is not a valid Git repository:\n{out}")

    status, out = run_command(["git", "remote", "get-url", "origin"], cwd=repo_folder)
    if status == 0 and "hzolfaghari2022/LSTM_Modelling" not in out:
        raise RuntimeError(
            "This local folder is connected to a different GitHub repository.\n"
            f"Actual remote:\n{out}"
        )

    run_command(["git", "remote", "set-url", "origin", REPO_SSH], cwd=repo_folder)
    run_command(["git", "branch", "-M", BRANCH_NAME], cwd=repo_folder)

    print("Pulling latest changes from GitHub...")
    status, out = run_command(["git", "pull", "origin", BRANCH_NAME, "--rebase"], cwd=repo_folder)
    if status != 0:
        print("Warning: git pull had an issue:")
        print(out)
        print("Continuing. This can happen if the repository is empty or newly created.")

    print("Copying current project files into the repository...")
    target_folder.mkdir(parents=True, exist_ok=True)

    patterns = ["*.py", "*.xlsx", "*.csv", "*.pt", "*.txt", "*.tex", "*.pdf", "*.png", "*.jpg", "*.svg"]
    for pattern in patterns:
        for file in source_folder.glob(pattern):
            if file.is_file():
                shutil.copy2(file, target_folder / file.name)

    source_fig_folder = source_folder / FIG_FOLDER
    target_fig_folder = target_folder / FIG_FOLDER
    if source_fig_folder.is_dir():
        if target_fig_folder.exists():
            shutil.rmtree(target_fig_folder)
        shutil.copytree(source_fig_folder, target_fig_folder)

    gitignore = repo_folder / ".gitignore"
    ignore_lines = [
        "# Python cache",
        "__pycache__/",
        "*.pyc",
        "",
        "# MATLAB Drive system files",
        ".MATLABDriveTag",
        "**/.MATLABDriveTag",
        "",
        "# Operating system files",
        ".DS_Store",
        "Thumbs.db",
    ]
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    with gitignore.open("a", encoding="utf-8") as fid:
        for line in ignore_lines:
            if line == "" or line not in existing:
                fid.write(line + "\n")

    run_command(["git", "config", "user.name", "Hussein Zolfaghari"], cwd=repo_folder)
    run_command(["git", "config", "user.email", "h.zolfaghari2015@gmail.com"], cwd=repo_folder)

    print("Committing and pushing files to GitHub using SSH...")
    status, out = run_command(["git", "add", "."], cwd=repo_folder)
    if status != 0:
        raise RuntimeError(f"Git add failed:\n{out}")

    status, _ = run_command(["git", "diff", "--cached", "--quiet"], cwd=repo_folder)
    if status == 0:
        print("No new changes to commit. Repository is already up to date.")
    else:
        commit_message = "Update Python single actuator ESA LSTM model"
        status, out = run_command(["git", "commit", "-m", commit_message], cwd=repo_folder)
        if status != 0:
            raise RuntimeError(f"Git commit failed:\n{out}")
        print(f"Commit completed successfully: {commit_message}")

    status, out = run_command(["git", "push", "-u", "origin", BRANCH_NAME], cwd=repo_folder)
    if status != 0:
        raise RuntimeError(
            "Git push failed using SSH.\n"
            "If the message says Permission denied publickey, test SSH using:\n"
            "ssh -T git@github.com\n\n"
            f"Git message:\n{out}"
        )

    print(f"Files pushed successfully to GitHub {BRANCH_NAME} branch.")
    print("====================================================")
    print("GitHub SSH push completed for LSTM_Modelling repository.")
    print("====================================================")


if __name__ == "__main__":
    main()
