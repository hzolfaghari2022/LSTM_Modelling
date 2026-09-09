"""Run the complete 13-feature V15 one-step model."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
from case_runner import run_case

if __name__ == "__main__":
    run_case(13, HERE)
