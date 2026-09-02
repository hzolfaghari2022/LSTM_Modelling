"""Primary entry point: causal one-step LSTM system identification.

Run this file for the measured-versus-predicted results requested in the
study. Pure-test records remain whole and are not used for fitting,
normalisation, early stopping, or model selection.

The separate autonomous_simulation.py answers the harder free-running
question and writes to different output folders. Keeping the two entry
points separate prevents an autonomous result from being mistaken for the
high-accuracy one-step result.
"""

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Required by PyTorch deterministic CUDA operations. This must be set before
# importing torch; setting it later produces repeated cuBLAS warnings.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from config import CPU_THREADS, SEED, TARGET_SAMPLE_RATE_HZ
from ablation_settings import SELECTED_FEATURE_COUNT
from data_utils import prepare_data
from one_step_lstm import run_one_step_study


torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(CPU_THREADS)

HERE = Path(__file__).resolve().parent

# Fail before a long training run if Windows cannot create the eventual
# prediction filenames. The old nested delivery reached exactly 260
# characters and failed only after training had completed.
if os.name == "nt":
    path_probe = (
        HERE
        / "ResultsData"
        / "one_step_predictions"
        / ("representative_prediction_filename_" + "x" * 70 + ".csv")
    )
    if len(str(path_probe)) >= 240:
        raise RuntimeError(
            "This project path is too long for reliable Windows output. "
            "Move the V15_Ablation folder closer to the repository root and "
            "avoid nesting it inside another V15_Ablation folder.\n"
            f"Current project folder: {HERE}"
        )

RESULTS_FOLDER = Path(
    os.environ.get("DLSTM_RESULTS_FOLDER", str(HERE / "ResultsData"))
).resolve()
FIGURES_FOLDER = Path(
    os.environ.get("DLSTM_FIGURES_FOLDER", str(HERE / "FiguresResults"))
).resolve()
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

# A new run must not leave the old autonomous Figures 01-13 beside the
# one-step report. Only generated PNGs in this dedicated folder are cleared.
for old_figure in FIGURES_FOLDER.glob("*.png"):
    old_figure.unlink()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Training device: {device}, {torch.cuda.get_device_name(0)}")
else:
    print("Training device: cpu")

print("\nLoading every measured record ...", flush=True)
data = prepare_data(HERE)
records = data["records"]

data["inventory_table"].to_csv(
    RESULTS_FOLDER / "record_inventory.csv", index=False
)
data["split_table"].to_csv(RESULTS_FOLDER / "data_split.csv", index=False)

print(f"Development workbook: {data['workbook'].name}")
if data["independent_test_workbook"] is not None:
    print(
        "Optional independent-test workbook: "
        f"{data['independent_test_workbook'].name}"
    )
print(f"Unique measured records: {len(records)}")
print(f"Common sample rate: {TARGET_SAMPLE_RATE_HZ:.0f} Hz")
print(f"Training windows: {len(data['training_pairs']):,}")
print(f"Validation windows: {len(data['validation_pairs']):,}")
print(f"Internal-test windows: {len(data['internal_test_pairs']):,}")
print("Untouched pure-test records from Total_Data.xlsx:")
for name in data["built_in_pure_test_names"]:
    print(f"  {name}")
if data["independent_test_names"]:
    print("Untouched independent-test records from Test_idpd.xlsx:")
    for name in data["independent_test_names"]:
        print(f"  {name}")

print(
    "\nPrediction mode: ONE STEP measured feedback. At sample k the model "
    "may use measured displacement and force only through k-1."
)
print(
    "This is intentionally different from autonomous/free-running "
    "simulation. Run autonomous_simulation.py only for that separate test."
)

metrics = run_one_step_study(
    data,
    records,
    device,
    RESULTS_FOLDER,
    FIGURES_FOLDER,
)

pure = metrics[metrics["Kind"] == "one_step_pure_test"]
failed = pure[~pure["Pass95"]]
independent = pure[
    pure["Evaluation"].isin(data["independent_test_names"])
]
print("\n" + "=" * 88)
if failed.empty:
    print("VERIFIED: every untouched pure-test channel passes its criterion.")
else:
    print("WARNING: these untouched pure-test channels did not pass:")
    print(
        failed[["Evaluation", "Output", "RMSE", "Fit_percent"]].to_string(
            index=False
        )
    )
if not independent.empty:
    print("\nINDEPENDENT TEST RESULTS (Test_idpd.xlsx):")
    print(
        independent[
            ["Evaluation", "Output", "RMSE", "Fit_percent", "Pass95"]
        ].to_string(index=False)
    )
print("=" * 88)
print("Numerical results:", RESULTS_FOLDER)
print("Measured-versus-predicted figures:", FIGURES_FOLDER)

# Commit and push this completed run to GitHub.
# Set DLSTM_SKIP_GITHUB_PUSH=1 only for a deliberately local run.
if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") != "1":
    try:
        from datetime import datetime
        from github_push import GitPushError, push_simulation

        run_time = datetime.now().astimezone()

        # Guarantees a genuine file change for every completed simulation.
        (HERE / "last_github_update.txt").write_text(
            (
                f"Latest completed {SELECTED_FEATURE_COUNT}-feature "
                "ablation simulation: "
                f"{run_time.isoformat(timespec='microseconds')}\n"
            ),
            encoding="utf-8",
        )

        push_simulation(
            HERE,
            commit_message=(
                f"Update V15 {SELECTED_FEATURE_COUNT}-feature ablation "
                "results - "
                f"{run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            ),
        )

    except (GitPushError, ImportError) as error:
        print("\nIMPORTANT: the simulation completed, but GitHub push failed:")
        print(error)
        print("Run 'python push_now.py' after correcting the Git configuration.")
        raise SystemExit(1) from error

print("\nDone.")
