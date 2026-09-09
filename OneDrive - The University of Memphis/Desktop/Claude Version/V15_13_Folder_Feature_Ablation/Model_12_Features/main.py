"""Run the cumulative 12-feature V15 ablation model."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
from case_runner import run_case

if __name__ == "__main__":
    run_case(12, HERE)
