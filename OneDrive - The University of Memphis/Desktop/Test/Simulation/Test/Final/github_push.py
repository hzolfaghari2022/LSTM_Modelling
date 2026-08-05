"""
REUSABLE GITHUB PUSH HELPER

This file contains only Git/GitHub-related code. It does not change the
simulation, model, training, data processing, metrics, or figures.

Expected setup:
    The simulation folder must be located inside a cloned Git repository.

Example:
    LSTM_Modelling/
        FARNN_COMSOL/
            main.py
            github_push.py
            ...

Use from main.py:
    from github_push import GitPushError, push_simulation

    try:
        push_simulation(HERE)
    except GitPushError as error:
        print(error)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess


# GitHub rejects a single normal Git file larger than 100 MB.
# A slightly smaller limit gives a clear warning before GitHub rejects it.
MAX_FILE_SIZE_BYTES = 95 * 1024 * 1024


class GitPushError(RuntimeError):
    """Raised when the automatic Git operation cannot be completed."""


def _run_git(
    arguments: list[str],
    working_directory: Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one Git command and return its completed process."""

    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=working_directory,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise GitPushError(
            "Git was not found. Install Git for Windows and make sure "
            "'git' works in PowerShell."
        ) from error

    if check and result.returncode != 0:
        command = "git " + " ".join(arguments)
        details = (
            result.stderr.strip()
            or result.stdout.strip()
            or "Git returned an unknown error."
        )

        raise GitPushError(
            f"Command failed:\n{command}\n\n{details}"
        )

    return result


def _find_repository_root(
    project_folder: Path,
) -> Path:
    """Find the root of the cloned repository containing the project."""

    result = _run_git(
        ["rev-parse", "--show-toplevel"],
        project_folder,
        check=False,
    )

    if result.returncode != 0:
        raise GitPushError(
            "This simulation folder is not inside a cloned Git "
            "repository.\n\n"
            "Clone the repository first, then place the simulation "
            "folder inside the cloned LSTM_Modelling folder."
        )

    return Path(result.stdout.strip()).resolve()


def _check_for_very_large_files(
    project_folder: Path,
) -> None:
    """Stop before GitHub rejects a file larger than the normal limit."""

    excluded_directories = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
    }

    large_files: list[tuple[Path, int]] = []

    for file_path in project_folder.rglob("*"):
        if not file_path.is_file():
            continue

        if any(
            part in excluded_directories
            for part in file_path.parts
        ):
            continue

        size = file_path.stat().st_size

        if size > MAX_FILE_SIZE_BYTES:
            large_files.append(
                (
                    file_path.relative_to(project_folder),
                    size,
                )
            )

    if large_files:
        formatted = "\n".join(
            f"  - {path} "
            f"({size / (1024 ** 2):.1f} MB)"
            for path, size in large_files
        )

        raise GitPushError(
            "The following files are too large for a normal GitHub "
            "push:\n"
            f"{formatted}\n\n"
            "Remove them, reduce them, or configure Git LFS."
        )


def _path_is_inside_scope(
    repository_path: str,
    project_scope: str,
) -> bool:
    """Return True when a repository file belongs to this simulation."""

    if project_scope == ".":
        return True

    normalized_path = repository_path.replace("\\", "/")
    normalized_scope = project_scope.rstrip("/")

    return (
        normalized_path == normalized_scope
        or normalized_path.startswith(
            normalized_scope + "/"
        )
    )


def push_simulation(
    project_folder: str | Path,
    *,
    commit_message: str | None = None,
    remote_name: str = "origin",
) -> bool:
    """Commit and push only the current simulation folder.

    Parameters
    ----------
    project_folder:
        Folder containing main.py. Usually pass HERE from main.py.

    commit_message:
        Optional custom Git commit message. When omitted, a timestamped
        simulation-update message is generated automatically.

    remote_name:
        Git remote to push. The normal cloned-repository value is origin.

    Returns
    -------
    bool
        True when a new commit was pushed.
        False when there were no changes to commit.
    """

    project_folder = Path(
        project_folder
    ).resolve()

    if not project_folder.exists():
        raise GitPushError(
            f"Project folder does not exist: {project_folder}"
        )

    _check_for_very_large_files(
        project_folder
    )

    repository_root = _find_repository_root(
        project_folder
    )

    try:
        relative_project = project_folder.relative_to(
            repository_root
        )
    except ValueError as error:
        raise GitPushError(
            "The simulation folder is not inside the detected "
            "repository."
        ) from error

    project_scope = (
        "."
        if relative_project == Path(".")
        else relative_project.as_posix()
    )

    # Make sure the cloned repository has a usable remote.
    remote_result = _run_git(
        ["remote", "get-url", remote_name],
        repository_root,
        check=False,
    )

    if remote_result.returncode != 0:
        raise GitPushError(
            f"The Git remote '{remote_name}' does not exist.\n"
            "Clone the repository using GitHub CLI or SSH instead of "
            "creating an unrelated local folder."
        )

    # Protect unrelated staged files. The helper pushes only this
    # simulation folder, not other work in the repository.
    staged_before_result = _run_git(
        ["diff", "--cached", "--name-only"],
        repository_root,
    )

    staged_before = [
        line.strip()
        for line in staged_before_result.stdout.splitlines()
        if line.strip()
    ]

    unrelated_staged = [
        path
        for path in staged_before
        if not _path_is_inside_scope(
            path,
            project_scope,
        )
    ]

    if unrelated_staged:
        formatted = "\n".join(
            f"  - {path}"
            for path in unrelated_staged
        )

        raise GitPushError(
            "Git already has staged files outside this simulation:\n"
            f"{formatted}\n\n"
            "Commit or unstage those files first. They were not changed "
            "by this helper."
        )

    # Stage only this simulation directory.
    _run_git(
        [
            "add",
            "--all",
            "--",
            project_scope,
        ],
        repository_root,
    )

    staged_result = _run_git(
        [
            "diff",
            "--cached",
            "--quiet",
            "--",
            project_scope,
        ],
        repository_root,
        check=False,
    )

    if staged_result.returncode == 0:
        print(
            "\nGitHub: no changed simulation files to commit."
        )
        return False

    if staged_result.returncode != 1:
        raise GitPushError(
            "Git could not check the staged simulation changes."
        )

    if commit_message is None:
        timestamp = datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )

        commit_message = (
            f"Update {project_folder.name} simulation - "
            f"{timestamp}"
        )

    _run_git(
        [
            "commit",
            "-m",
            commit_message,
        ],
        repository_root,
    )

    branch_result = _run_git(
        ["branch", "--show-current"],
        repository_root,
    )

    branch = branch_result.stdout.strip()

    if not branch:
        raise GitPushError(
            "Git is in a detached-HEAD state. Check out the main branch "
            "before running the simulation."
        )

    # Rebase the new local commit on any changes already pushed from
    # another computer. This keeps the history clean.
    remote_branch_result = _run_git(
        [
            "ls-remote",
            "--exit-code",
            "--heads",
            remote_name,
            branch,
        ],
        repository_root,
        check=False,
    )

    if remote_branch_result.returncode == 0:
        _run_git(
            [
                "pull",
                "--rebase",
                remote_name,
                branch,
            ],
            repository_root,
        )

    _run_git(
        [
            "push",
            "--set-upstream",
            remote_name,
            branch,
        ],
        repository_root,
    )

    commit_sha = _run_git(
        ["rev-parse", "--short", "HEAD"],
        repository_root,
    ).stdout.strip()

    remote_url = remote_result.stdout.strip()

    print("\nGitHub push completed.")
    print("Repository:", remote_url)
    print("Branch:", branch)
    print("Commit:", commit_sha)
    print("Simulation folder:", project_scope)

    return True
