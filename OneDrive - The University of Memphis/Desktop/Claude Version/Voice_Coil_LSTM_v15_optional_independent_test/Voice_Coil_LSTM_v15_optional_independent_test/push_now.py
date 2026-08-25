"""Push the current project and saved results without retraining the model."""

from pathlib import Path

from github_push import GitPushError, push_simulation


HERE = Path(__file__).resolve().parent

try:
    push_simulation(
        HERE,
        commit_message="Push voice-coil LSTM code and saved results",
    )
except GitPushError as error:
    print("\nGitHub push failed:")
    print(error)
    raise SystemExit(1) from error
