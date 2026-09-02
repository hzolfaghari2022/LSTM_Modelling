"""Aggregate completed V15 ablation pure-test metrics and push them."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from github_push import GitPushError, push_simulation


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Comparison"
PARAMETERS = {count: 30017 + 192 * count for count in range(1, 14)}


def main() -> None:
    metric_frames = []
    for count in range(13, 0, -1):
        path = (
            HERE
            / f"F{count:02d}"
            / "ResultsData"
            / "one_step_metrics.csv"
        )
        if not path.exists():
            print(f"Skipping {count} features; results do not exist yet.")
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["Kind"] == "one_step_pure_test"].copy()
        frame.insert(0, "Feature_count", count)
        frame.insert(1, "Trainable_parameters", PARAMETERS[count])
        metric_frames.append(frame)

    if not metric_frames:
        raise FileNotFoundError(
            "No completed one_step_metrics.csv files were found. Run at "
            "least one ablation model first."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics.to_csv(OUTPUT / "all_pure_test_metrics.csv", index=False)

    summary_rows = []
    for (count, output), group in all_metrics.groupby(
        ["Feature_count", "Output"], sort=False
    ):
        finite_fit = pd.to_numeric(group["Fit_percent"], errors="coerce")
        finite_fit = finite_fit[np.isfinite(finite_fit)]
        summary_rows.append(
            {
                "Feature_count": int(count),
                "Trainable_parameters": PARAMETERS[int(count)],
                "Output": output,
                "Pure_test_channels": len(group),
                "Passing_channels": int(
                    group["Pass95"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"true", "1"})
                    .sum()
                ),
                "Mean_RMSE": float(group["RMSE"].mean()),
                "Maximum_RMSE": float(group["RMSE"].max()),
                "Mean_fit_percent": (
                    float(finite_fit.mean()) if len(finite_fit) else np.nan
                ),
                "Worst_fit_percent": (
                    float(finite_fit.min()) if len(finite_fit) else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["Feature_count", "Output"], ascending=[False, True]
    )
    summary.to_csv(OUTPUT / "feature_ablation_summary.csv", index=False)

    displacement = summary[summary["Output"] == "Displacement"].sort_values(
        "Feature_count"
    )
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    axes[0].plot(
        displacement["Feature_count"],
        displacement["Mean_RMSE"],
        "o-",
        color="#0072B2",
        linewidth=2,
    )
    axes[0].set_ylabel("Mean pure-test RMSE (mm)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        displacement["Feature_count"],
        displacement["Worst_fit_percent"],
        "o-",
        color="#D55E00",
        linewidth=2,
    )
    axes[1].axhline(95.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of retained features")
    axes[1].set_ylabel("Worst pure-test fit (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(displacement["Feature_count"])
    figure.suptitle("V15 one-step displacement feature ablation")
    figure.tight_layout()
    figure.savefig(OUTPUT / "feature_ablation_comparison.png", dpi=200)
    plt.close(figure)

    print(f"\nComparison results: {OUTPUT}")
    if os.environ.get("DLSTM_SKIP_GITHUB_PUSH", "0") == "1":
        print("GitHub push skipped because DLSTM_SKIP_GITHUB_PUSH=1.")
        return

    run_time = datetime.now().astimezone()
    (OUTPUT / "last_github_update.txt").write_text(
        f"Latest completed ablation comparison: {run_time.isoformat()}\n",
        encoding="utf-8",
    )
    try:
        push_simulation(
            HERE,
            commit_message=(
                "Update V15 feature-ablation comparison - "
                f"{run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            ),
        )
    except GitPushError as error:
        print("\nComparison was saved, but its GitHub push failed:")
        print(error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
