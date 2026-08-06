"""
PUSH THE CURRENT FINAL FOLDER WITHOUT RUNNING THE SIMULATION AGAIN

Run:
    python push_now.py

This file does not import or execute main.py.
"""

from pathlib import Path

from github_push import GitPushError, push_simulation


HERE = Path(__file__).resolve().parent


try:
    push_simulation(
        HERE,
        commit_message="Push Final LSTM simulation code and results",
    )
except GitPushError as error:
    print("\nGitHub push failed:")
    print(error)
    raise SystemExit(1) from error
