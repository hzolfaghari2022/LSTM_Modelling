"""Run an honest leave-one-record-out evaluation for the requested pure set.

Each fold trains a fresh model that cannot access its target record.  Unlike
the default simultaneous holdout, related measured records remain available;
this estimates the accuracy of the final production model more realistically
without ever training on the trajectory being scored.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data_utils import prepare_data


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "UnseenRecordResults"


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def main():
    # A normal preparation pass discovers the default pure-test set from the
    # workbook.  Its arrays are discarded; every fold performs a fresh fit.
    targets = prepare_data(HERE)["pure_test_names"]
    if not targets:
        raise RuntimeError("No pure-test records were selected.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_number, target in enumerate(targets, start=1):
        fold = OUTPUT / safe_name(target)
        results = fold / "ResultsData"
        figures = fold / "FiguresResults"
        environment = os.environ.copy()
        environment.update(
            {
                "DLSTM_PURE_TEST_RECORD": target,
                "DLSTM_RESULTS_FOLDER": str(results),
                "DLSTM_FIGURES_FOLDER": str(figures),
                "DLSTM_SKIP_GITHUB_PUSH": "1",
            }
        )
        print(f"\nFold {fold_number}/{len(targets)}: unseen {target}", flush=True)
        subprocess.run(
            [sys.executable, str(HERE / "main.py")],
            cwd=HERE,
            env=environment,
            check=True,
        )
        metrics = pd.read_csv(results / "metrics.csv")
        selected = metrics[
            (metrics["Kind"] == "pure_test")
            & (metrics["Evaluation"] == target)
        ].copy()
        selected.insert(0, "Fold", fold_number)
        rows.append(selected)

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(OUTPUT / "leave_one_record_out_metrics.csv", index=False)
    print("\nLeave-one-record-out summary:")
    print(
        summary[
            ["Evaluation", "Output", "RMSE", "MAE", "R2", "Fit_percent"]
        ].to_string(index=False)
    )
    print("\nMeasured-versus-predicted figures and fold models are in:", OUTPUT)


if __name__ == "__main__":
    main()
