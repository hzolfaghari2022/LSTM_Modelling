"""Push the complete 13-folder ablation project and saved results without retraining."""

from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "SharedCode"))

from github_push import GitPushError, push_simulation

try:
    now = datetime.now().astimezone()
    (HERE / "last_github_update.txt").write_text(
        f"Latest manual ablation push: {now.isoformat(timespec='microseconds')}\n",
        encoding="utf-8",
    )
    push_simulation(
        HERE,
        commit_message=(
            "Push V15 13-folder feature-ablation code and results - "
            f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        ),
    )
except GitPushError as error:
    print("\nGitHub push failed:")
    print(error)
    raise SystemExit(1) from error
