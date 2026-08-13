"""Push saved code and results without rerunning the simulation."""

from pathlib import Path

from github_push import GitPushError, push_simulation


HERE = Path(__file__).resolve().parent


# Call the reusable helper for this folder only.
try:
    push_simulation(
        HERE,
        commit_message="Push Final LSTM simulation code and results",
    )
# Stop with a readable message when Git cannot complete the push.
except GitPushError as error:
    print("\nGitHub push failed:")
    print(error)
    raise SystemExit(1) from error
