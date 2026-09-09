"""Commit and push this simulation folder to its configured GitHub repository."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
import subprocess


EXPECTED_REPOSITORY = os.environ.get(
    "DLSTM_GITHUB_REPOSITORY", "hzolfaghari2022/LSTM_Modelling"
).strip()
REMOTE_NAME = os.environ.get("DLSTM_GITHUB_REMOTE", "origin").strip() or "origin"
MAX_FILE_SIZE_BYTES = 95 * 1024 * 1024


class GitPushError(RuntimeError):
    """Raised when a safe, verified GitHub push cannot be completed."""


def _run(arguments, folder, check=True):
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=folder,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise GitPushError(
            "Git was not found. Install Git for Windows and ensure the "
            "'git' command works in the VS Code terminal."
        ) from error

    if check and result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise GitPushError(
            f"Command failed: git {' '.join(arguments)}\n\n{details}"
        )
    return result


def _repository_slug(remote_url):
    value = remote_url.strip().replace("\\", "/")
    if value.startswith("git@github.com:"):
        value = value.split(":", 1)[1]
    value = re.sub(
        r"^(?:ssh://git@|https?://|git://)github\.com/",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.removesuffix(".git").strip("/")


def _inside(path, scope):
    path = path.replace("\\", "/")
    scope = scope.replace("\\", "/").rstrip("/")
    return scope == "." or path == scope or path.startswith(scope + "/")


def _lines(result):
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _check_file_sizes(project_folder):
    oversized = []
    for path in project_folder.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            oversized.append(path)
    if oversized:
        details = "\n".join(
            f"  {path.relative_to(project_folder)} "
            f"({path.stat().st_size / 1024**2:.1f} MB)"
            for path in oversized
        )
        raise GitPushError(
            "These files exceed GitHub's normal file-size limit:\n"
            f"{details}\nConfigure Git LFS or remove them before pushing."
        )


def _changed_project_paths(repository_root, project_scope):
    """Return explicit changed paths so unrelated files are never staged."""
    result = _run(
        [
            "ls-files",
            "--modified",
            "--deleted",
            "--others",
            "--exclude-standard",
            "--",
            project_scope,
        ],
        repository_root,
    )
    return list(dict.fromkeys(_lines(result)))


def _stage_explicit_paths(repository_root, paths):
    for start in range(0, len(paths), 100):
        _run(["add", "--", *paths[start:start + 100]], repository_root)


def push_simulation(project_folder, commit_message=None):
    """Push only this project folder and verify the remote commit."""
    project_folder = Path(project_folder).resolve()
    _check_file_sizes(project_folder)

    root_result = _run(
        ["rev-parse", "--show-toplevel"], project_folder, check=False
    )
    if root_result.returncode != 0:
        raise GitPushError(
            "This project folder is not inside a cloned Git repository. "
            "Place it inside your clone of "
            f"https://github.com/{EXPECTED_REPOSITORY}."
        )
    repository_root = Path(root_result.stdout.strip()).resolve()
    try:
        relative_project = project_folder.relative_to(repository_root)
    except ValueError as error:
        raise GitPushError("The project is outside the detected repository.") from error
    project_scope = "." if relative_project == Path(".") else relative_project.as_posix()

    remote_result = _run(
        ["remote", "get-url", REMOTE_NAME], repository_root, check=False
    )
    if remote_result.returncode != 0:
        raise GitPushError(f"Git remote '{REMOTE_NAME}' is not configured.")
    remote_url = remote_result.stdout.strip()
    actual_repository = _repository_slug(remote_url)
    if actual_repository.lower() != EXPECTED_REPOSITORY.lower():
        raise GitPushError(
            "The Git remote points to a different repository.\n"
            f"Expected: {EXPECTED_REPOSITORY}\nDetected: {actual_repository}"
        )

    branch = _run(["branch", "--show-current"], repository_root).stdout.strip()
    if not branch:
        raise GitPushError("Git is in detached-HEAD state; check out a branch first.")

    staged_before = _lines(
        _run(["diff", "--cached", "--name-only"], repository_root)
    )
    unrelated = [path for path in staged_before if not _inside(path, project_scope)]
    if unrelated:
        raise GitPushError(
            "Unrelated files are already staged and were not touched:\n  "
            + "\n  ".join(unrelated)
        )

    changed = _changed_project_paths(repository_root, project_scope)
    if changed:
        _stage_explicit_paths(repository_root, changed)

    staged_project = _lines(
        _run(
            ["diff", "--cached", "--name-only", "--", project_scope],
            repository_root,
        )
    )
    if staged_project:
        if commit_message is None:
            timestamp = datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
            commit_message = f"Update {project_folder.name} results - {timestamp}"
        _run(["commit", "-m", commit_message], repository_root)
        print(f"Created Git commit with {len(staged_project)} project file(s).")
    else:
        print("No new project changes require a Git commit.")

    _run(["push", "--set-upstream", REMOTE_NAME, branch], repository_root)
    local_sha = _run(["rev-parse", "HEAD"], repository_root).stdout.strip()
    remote_line = _run(
        ["ls-remote", REMOTE_NAME, f"refs/heads/{branch}"], repository_root
    ).stdout.strip()
    if not remote_line or remote_line.split()[0] != local_sha:
        raise GitPushError("GitHub push returned, but the remote commit did not verify.")

    print("GitHub push completed and verified.")
    print("Repository:", EXPECTED_REPOSITORY)
    print("Branch:", branch)
    print("Commit:", local_sha[:12])
    return True
