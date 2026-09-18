"""Run the one-feature V15 model using coil current only."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
from case_runner import run_case

if __name__ == "__main__":
    run_case(1, HERE)
