# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Safely commit and push one completed simulation folder."""

# Import selected names from __future__ instead of importing its complete namespace.
from __future__ import annotations

# Import selected names from datetime instead of importing its complete namespace.
from datetime import datetime
# Import selected names from pathlib instead of importing its complete namespace.
from pathlib import Path
# Import regular expressions for recognizing and cleaning GitHub remote URLs.
import re
# Import subprocess support for running Git commands or the separate plotting script.
import subprocess


# Evaluate `"hzolfaghari2022/LSTM_Modelling"` and store the result in `EXPECTED_REPOSITORY` for the following steps.
EXPECTED_REPOSITORY = "hzolfaghari2022/LSTM_Modelling"

# Repository target and normal GitHub file-size limit
MAX_FILE_SIZE_BYTES = 95 * 1024 * 1024


# Define the GitPushError class used below.
class GitPushError(RuntimeError):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Raised when the automatic Git operation cannot be completed."""


# Run Git commands and report readable failures
def _run_git(
    # Use the expression `arguments: list[str]` as the next part of the surrounding Python statement.
    arguments: list[str],
    # Use the expression `working_directory: Path` as the next part of the surrounding Python statement.
    working_directory: Path,
    # Multiply the previous quantity by this factor to form the current weighted term.
    *,
    # Use the expression `check: bool = True` as the next part of the surrounding Python statement.
    check: bool = True,
# Begin the indented block controlled by this statement.
) -> subprocess.CompletedProcess[str]:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Run one Git command and return the completed process."""

    # Attempt the following operation so an expected failure can be handled cleanly.
    try:
        # Store the completed subprocess result, including return code and captured text.
        result = subprocess.run(
            # Begin the grouped expression or collection continued on the following lines.
            ["git", *arguments],
            # Run the external command from this explicitly selected working directory.
            cwd=working_directory,
            # Decode subprocess output to Python strings instead of raw bytes.
            text=True,
            # Capture the command's standard output and error so this script can inspect and report them.
            capture_output=True,
            # Do not let subprocess raise automatically; this code checks the return code and produces its own message.
            check=False,
        # Close the current function call, tuple, or grouped expression.
        )
    # Handle the stated exception instead of ending with an unprocessed traceback.
    except FileNotFoundError as error:
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Git was not found. Install Git for Windows and make sure "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "'git' works in PowerShell."
        # Use the expression `) from error` as the next part of the surrounding Python statement.
        ) from error

    # Evaluate this condition and run the following indented block only when it is true.
    if check and result.returncode != 0:
        # Build the exact external command that will be executed.
        command = "git " + " ".join(arguments)
        # Combine available command output into a useful diagnostic message.
        details = (
            # Call `result.stderr.strip`; the following indented continuation lines provide its arguments.
            result.stderr.strip()
            # Continue the current Boolean condition with this additional test.
            or result.stdout.strip()
            # Continue the current Boolean condition with this additional test.
            or "Git returned an unknown error."
        # Close the current function call, tuple, or grouped expression.
        )

        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Command failed:\n{command}\n\n{details}"
        # Close the current function call, tuple, or grouped expression.
        )

    # Return this value to the code that called the current function.
    return result


# Locate and verify the local repository and its remote
def _find_repository_root(
    # Use the expression `project_folder: Path` as the next part of the surrounding Python statement.
    project_folder: Path,
# Begin the indented block controlled by this statement.
) -> Path:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Find the root of the cloned repository containing the project."""

    # Store the completed subprocess result, including return code and captured text.
    result = _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        ["rev-parse", "--show-toplevel"],
        # Pass `project_folder` as the next value required by the surrounding call or collection.
        project_folder,
        # Do not let subprocess raise automatically; this code checks the return code and produces its own message.
        check=False,
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate this condition and run the following indented block only when it is true.
    if result.returncode != 0:
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "This simulation folder is not inside a cloned Git "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "repository.\n\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Clone hzolfaghari2022/LSTM_Modelling first, then place "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "the Final folder anywhere inside that cloned repository."
        # Close the current function call, tuple, or grouped expression.
        )

    # Return this value to the code that called the current function.
    return Path(result.stdout.strip()).resolve()


# Define the _repository_slug_from_remote function; its indented lines form the function body.
def _repository_slug_from_remote(
    # Use the expression `remote_url: str` as the next part of the surrounding Python statement.
    remote_url: str,
# Begin the indented block controlled by this statement.
) -> str:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Convert HTTPS or SSH GitHub URLs to owner/repository form."""

    # Normalize URL or path text so equivalent GitHub repository forms compare consistently.
    normalized = remote_url.strip().replace("\\", "/")

    # git@github.com:owner/repository.git
    if normalized.startswith("git@github.com:"):
        # Normalize URL or path text so equivalent GitHub repository forms compare consistently.
        normalized = normalized.split(":", 1)[1]

    # ssh://git@github.com/owner/repository.git
    normalized = re.sub(
        # Use the expression `r"^ssh://git@github\.com/"` as the next part of the surrounding Python statement.
        r"^ssh://git@github\.com/",
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        "",
        # Pass `normalized` as the next value required by the surrounding call or collection.
        normalized,
        # Pass `re.IGNORECASE` as the `flags` argument of the surrounding function call.
        flags=re.IGNORECASE,
    # Close the current function call, tuple, or grouped expression.
    )

    # https://github.com/owner/repository.git
    normalized = re.sub(
        # Use the expression `r"^https?://github\.com/"` as the next part of the surrounding Python statement.
        r"^https?://github\.com/",
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        "",
        # Pass `normalized` as the next value required by the surrounding call or collection.
        normalized,
        # Pass `re.IGNORECASE` as the `flags` argument of the surrounding function call.
        flags=re.IGNORECASE,
    # Close the current function call, tuple, or grouped expression.
    )

    # git://github.com/owner/repository.git
    normalized = re.sub(
        # Use the expression `r"^git://github\.com/"` as the next part of the surrounding Python statement.
        r"^git://github\.com/",
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        "",
        # Pass `normalized` as the next value required by the surrounding call or collection.
        normalized,
        # Pass `re.IGNORECASE` as the `flags` argument of the surrounding function call.
        flags=re.IGNORECASE,
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate this condition and run the following indented block only when it is true.
    if normalized.endswith(".git"):
        # Normalize URL or path text so equivalent GitHub repository forms compare consistently.
        normalized = normalized[:-4]

    # Return this value to the code that called the current function.
    return normalized.strip("/")


# Define the _check_repository_target function; its indented lines form the function body.
def _check_repository_target(
    # Use the expression `remote_url: str` as the next part of the surrounding Python statement.
    remote_url: str,
# Begin the indented block controlled by this statement.
) -> None:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Prevent an accidental push to a different repository."""

    # Extract the repository owner/name from the configured remote URL.
    actual_repository = _repository_slug_from_remote(
        # Pass `remote_url` as the next value required by the surrounding call or collection.
        remote_url
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate this condition and run the following indented block only when it is true.
    if actual_repository.lower() != EXPECTED_REPOSITORY.lower():
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "The local Git remote does not point to the expected "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "repository.\n\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Expected: {EXPECTED_REPOSITORY}\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Detected: {actual_repository or remote_url}\n\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Correct the origin remote before pushing."
        # Close the current function call, tuple, or grouped expression.
        )


# Check the simulation folder before staging it
def _check_for_very_large_files(
    # Use the expression `project_folder: Path` as the next part of the surrounding Python statement.
    project_folder: Path,
# Begin the indented block controlled by this statement.
) -> None:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Stop before GitHub rejects a file larger than the normal limit."""

    # List generated or environment folders that should not be committed.
    excluded_directories = {
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        ".git",
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        "__pycache__",
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        ".venv",
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        "venv",
    # Close the current dictionary.
    }

    # Use the expression `large_files: list[tuple[Path, int]] = []` as the next part of the surrounding Python statement.
    large_files: list[tuple[Path, int]] = []

    # Repeat the following indented block once for each item in this iterable.
    for file_path in project_folder.rglob("*"):
        # Evaluate this condition and run the following indented block only when it is true.
        if not file_path.is_file():
            # Skip the remaining statements in this loop iteration and continue with the next item.
            continue

        # Evaluate this condition and run the following indented block only when it is true.
        if any(
            # Use the expression `part in excluded_directories` as the next part of the surrounding Python statement.
            part in excluded_directories
            # Repeat the following indented block once for each item in this iterable.
            for part in file_path.parts
        # Begin the indented block controlled by this statement.
        ):
            # Skip the remaining statements in this loop iteration and continue with the next item.
            continue

        # Read this file's byte size to enforce the GitHub upload limit.
        size = file_path.stat().st_size

        # Evaluate this condition and run the following indented block only when it is true.
        if size > MAX_FILE_SIZE_BYTES:
            # Call `large_files.append`; the following indented continuation lines provide its arguments.
            large_files.append(
                # Begin the grouped expression or collection continued on the following lines.
                (
                    # Call `file_path.relative_to`; the following indented continuation lines provide its arguments.
                    file_path.relative_to(project_folder),
                    # Pass `size` as the next value required by the surrounding call or collection.
                    size,
                # Close the current function call, tuple, or grouped expression.
                )
            # Close the current function call, tuple, or grouped expression.
            )

    # Evaluate this condition and run the following indented block only when it is true.
    if large_files:
        # Format the collected paths or messages for readable terminal output.
        formatted = "\n".join(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"  - {path} ({size / (1024 ** 2):.1f} MB)"
            # Repeat the following indented block once for each item in this iterable.
            for path, size in large_files
        # Close the current function call, tuple, or grouped expression.
        )

        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "The following files are too large for a normal GitHub "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "push:\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"{formatted}\n\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Remove them, reduce them, or configure Git LFS."
        # Close the current function call, tuple, or grouped expression.
        )


# Define the _path_is_inside_scope function; its indented lines form the function body.
def _path_is_inside_scope(
    # Use the expression `repository_path: str` as the next part of the surrounding Python statement.
    repository_path: str,
    # Use the expression `project_scope: str` as the next part of the surrounding Python statement.
    project_scope: str,
# Begin the indented block controlled by this statement.
) -> bool:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Return True when a repository file belongs to this simulation."""

    # Evaluate this condition and run the following indented block only when it is true.
    if project_scope == ".":
        # Return this value to the code that called the current function.
        return True

    # Convert path separators and redundant syntax to a consistent comparison form.
    normalized_path = repository_path.replace("\\", "/")
    # Normalize the project subfolder used to limit Git staging.
    normalized_scope = project_scope.rstrip("/")

    # Return this value to the code that called the current function.
    return (
        # Convert path separators and redundant syntax to a consistent comparison form.
        normalized_path == normalized_scope
        # Continue the current Boolean condition with this additional test.
        or normalized_path.startswith(
            # Use the expression `normalized_scope + "/"` as the next part of the surrounding Python statement.
            normalized_scope + "/"
        # Close the current function call, tuple, or grouped expression.
        )
    # Close the current function call, tuple, or grouped expression.
    )


# Read branch information and verify the final remote commit
def _current_branch(
    # Use the expression `repository_root: Path` as the next part of the surrounding Python statement.
    repository_root: Path,
# Begin the indented block controlled by this statement.
) -> str:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Return the checked-out branch and reject detached HEAD."""

    # Store the currently checked-out Git branch name.
    branch = _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        ["branch", "--show-current"],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
    # Use the expression `).stdout.strip()` as the next part of the surrounding Python statement.
    ).stdout.strip()

    # Evaluate this condition and run the following indented block only when it is true.
    if not branch:
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Git is in a detached-HEAD state. Check out the main branch "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "before running the simulation."
        # Close the current function call, tuple, or grouped expression.
        )

    # Return this value to the code that called the current function.
    return branch


# Define the _remote_branch_exists function; its indented lines form the function body.
def _remote_branch_exists(
    # Use the expression `repository_root: Path` as the next part of the surrounding Python statement.
    repository_root: Path,
    # Use the expression `remote_name: str` as the next part of the surrounding Python statement.
    remote_name: str,
    # Use the expression `branch: str` as the next part of the surrounding Python statement.
    branch: str,
# Begin the indented block controlled by this statement.
) -> bool:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Return whether the named branch already exists on the remote."""

    # Store the completed subprocess result, including return code and captured text.
    result = _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        [
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "ls-remote",
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "--exit-code",
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "--heads",
            # Pass `remote_name` as the next value required by the surrounding call or collection.
            remote_name,
            # Pass `branch` as the next value required by the surrounding call or collection.
            branch,
        # Close the current list or index expression.
        ],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
        # Do not let subprocess raise automatically; this code checks the return code and produces its own message.
        check=False,
    # Close the current function call, tuple, or grouped expression.
    )

    # Return this value to the code that called the current function.
    return result.returncode == 0


# Define the _verify_remote_commit function; its indented lines form the function body.
def _verify_remote_commit(
    # Use the expression `repository_root: Path` as the next part of the surrounding Python statement.
    repository_root: Path,
    # Use the expression `remote_name: str` as the next part of the surrounding Python statement.
    remote_name: str,
    # Use the expression `branch: str` as the next part of the surrounding Python statement.
    branch: str,
# Begin the indented block controlled by this statement.
) -> str:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Verify that local HEAD and the remote branch contain the same SHA."""

    # Read the local commit identifier for push verification.
    local_sha = _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        ["rev-parse", "HEAD"],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
    # Use the expression `).stdout.strip()` as the next part of the surrounding Python statement.
    ).stdout.strip()

    # Store the Git query result used to inspect the remote branch.
    remote_result = _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        [
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "ls-remote",
            # Pass `remote_name` as the next value required by the surrounding call or collection.
            remote_name,
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"refs/heads/{branch}",
        # Close the current list or index expression.
        ],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
    # Close the current function call, tuple, or grouped expression.
    )

    # Select the remote-reference line returned by Git.
    remote_line = remote_result.stdout.strip()

    # Evaluate this condition and run the following indented block only when it is true.
    if not remote_line:
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "The push command returned successfully, but the remote "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "branch could not be verified."
        # Close the current function call, tuple, or grouped expression.
        )

    # Extract the remote commit identifier for comparison with the local commit.
    remote_sha = remote_line.split()[0]

    # Evaluate this condition and run the following indented block only when it is true.
    if local_sha != remote_sha:
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Remote verification failed.\n\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Local HEAD:  {local_sha}\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Remote HEAD: {remote_sha}"
        # Close the current function call, tuple, or grouped expression.
        )

    # Return this value to the code that called the current function.
    return local_sha


# Stage, commit, rebase, push, and verify one simulation folder
def push_simulation(
    # Use the expression `project_folder: str | Path` as the next part of the surrounding Python statement.
    project_folder: str | Path,
    # Multiply the previous quantity by this factor to form the current weighted term.
    *,
    # Use the expression `commit_message: str | None = None` as the next part of the surrounding Python statement.
    commit_message: str | None = None,
    # Use the expression `remote_name: str = "origin"` as the next part of the surrounding Python statement.
    remote_name: str = "origin",
# Begin the indented block controlled by this statement.
) -> bool:
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Commit, push, and verify only the current simulation folder.

    A previous interrupted run may already have created a local commit.
    Therefore, this function still performs the pull/push/verification
    steps when there are no new files to commit.
    """

    # Resolve the project folder that this helper is allowed to stage and push.
    project_folder = Path(
        # Pass `project_folder` as the next value required by the surrounding call or collection.
        project_folder
    # Use the expression `).resolve()` as the next part of the surrounding Python statement.
    ).resolve()

    # Evaluate this condition and run the following indented block only when it is true.
    if not project_folder.exists():
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Project folder does not exist: {project_folder}"
        # Close the current function call, tuple, or grouped expression.
        )

    # Call `_check_for_very_large_files`; the following indented continuation lines provide its arguments.
    _check_for_very_large_files(
        # Pass `project_folder` as the next value required by the surrounding call or collection.
        project_folder
    # Close the current function call, tuple, or grouped expression.
    )

    # Store the root folder of the detected local Git repository.
    repository_root = _find_repository_root(
        # Pass `project_folder` as the next value required by the surrounding call or collection.
        project_folder
    # Close the current function call, tuple, or grouped expression.
    )

    # Attempt the following operation so an expected failure can be handled cleanly.
    try:
        # Express the selected project path relative to the repository root.
        relative_project = project_folder.relative_to(
            # Pass `repository_root` as the next value required by the surrounding call or collection.
            repository_root
        # Close the current function call, tuple, or grouped expression.
        )
    # Handle the stated exception instead of ending with an unprocessed traceback.
    except ValueError as error:
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "The simulation folder is not inside the detected "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "repository."
        # Use the expression `) from error` as the next part of the surrounding Python statement.
        ) from error

    # Store the normalized relative subfolder used to prevent unrelated staging.
    project_scope = (
        # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
        "."
        # Evaluate this condition and run the following indented block only when it is true.
        if relative_project == Path(".")
        # Use the expression `else relative_project.as_posix()` as the next part of the surrounding Python statement.
        else relative_project.as_posix()
    # Close the current function call, tuple, or grouped expression.
    )

    # Store the Git query result used to inspect the remote branch.
    remote_result = _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        ["remote", "get-url", remote_name],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
        # Do not let subprocess raise automatically; this code checks the return code and produces its own message.
        check=False,
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate this condition and run the following indented block only when it is true.
    if remote_result.returncode != 0:
        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"The Git remote '{remote_name}' does not exist.\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Clone the repository using GitHub CLI or SSH instead of "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "creating an unrelated local folder."
        # Close the current function call, tuple, or grouped expression.
        )

    # Store the configured Git remote URL for repository verification.
    remote_url = remote_result.stdout.strip()
    # Call `_check_repository_target`; the following indented continuation lines provide its arguments.
    _check_repository_target(remote_url)

    # Store the currently checked-out Git branch name.
    branch = _current_branch(
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root
    # Close the current function call, tuple, or grouped expression.
    )

    # Print this progress or result message in the terminal.
    print("\nGitHub target verified.")
    # Print this progress or result message in the terminal.
    print("Repository:", EXPECTED_REPOSITORY)
    # Print this progress or result message in the terminal.
    print("Remote:", remote_url)
    # Print this progress or result message in the terminal.
    print("Branch:", branch)
    # Print this progress or result message in the terminal.
    print("Local repository:", repository_root)
    # Print this progress or result message in the terminal.
    print("Simulation folder:", project_scope)

    # Do not mix unrelated staged work into this simulation commit.
    staged_before_result = _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        ["diff", "--cached", "--name-only"],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
    # Close the current function call, tuple, or grouped expression.
    )

    # Record files already staged before this helper runs so unrelated work is preserved.
    staged_before = [
        # Call `line.strip`; the following indented continuation lines provide its arguments.
        line.strip()
        # Repeat the following indented block once for each item in this iterable.
        for line in staged_before_result.stdout.splitlines()
        # Evaluate this condition and run the following indented block only when it is true.
        if line.strip()
    # Close the current list or index expression.
    ]

    # Identify pre-existing staged files outside this project scope.
    unrelated_staged = [
        # Pass `path` as the next value required by the surrounding call or collection.
        path
        # Repeat the following indented block once for each item in this iterable.
        for path in staged_before
        # Evaluate this condition and run the following indented block only when it is true.
        if not _path_is_inside_scope(
            # Pass `path` as the next value required by the surrounding call or collection.
            path,
            # Pass `project_scope` as the next value required by the surrounding call or collection.
            project_scope,
        # Close the current function call, tuple, or grouped expression.
        )
    # Close the current list or index expression.
    ]

    # Evaluate this condition and run the following indented block only when it is true.
    if unrelated_staged:
        # Format the collected paths or messages for readable terminal output.
        formatted = "\n".join(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"  - {path}"
            # Repeat the following indented block once for each item in this iterable.
            for path in unrelated_staged
        # Close the current function call, tuple, or grouped expression.
        )

        # Stop this operation and report the stated error condition.
        raise GitPushError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Git already has staged files outside this simulation:\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"{formatted}\n\n"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Commit or unstage those files first. They were not changed "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "by this helper."
        # Close the current function call, tuple, or grouped expression.
        )

    # Stage only this simulation folder.
    _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        [
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "add",
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "--all",
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "--",
            # Pass `project_scope` as the next value required by the surrounding call or collection.
            project_scope,
        # Close the current list or index expression.
        ],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
    # Close the current function call, tuple, or grouped expression.
    )

    # Query Git for the names staged after adding the selected project.
    staged_names_result = _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        [
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "diff",
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "--cached",
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "--name-only",
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "--",
            # Pass `project_scope` as the next value required by the surrounding call or collection.
            project_scope,
        # Close the current list or index expression.
        ],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
    # Close the current function call, tuple, or grouped expression.
    )

    # Parse the staged filenames returned by Git.
    staged_names = [
        # Call `line.strip`; the following indented continuation lines provide its arguments.
        line.strip()
        # Repeat the following indented block once for each item in this iterable.
        for line in staged_names_result.stdout.splitlines()
        # Evaluate this condition and run the following indented block only when it is true.
        if line.strip()
    # Close the current list or index expression.
    ]

    # Evaluate this condition and run the following indented block only when it is true.
    if staged_names:
        # Print this progress or result message in the terminal.
        print(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Staged {len(staged_names)} changed simulation file(s)."
        # Close the current function call, tuple, or grouped expression.
        )

        # Evaluate this condition and run the following indented block only when it is true.
        if commit_message is None:
            # Create a human-readable timestamp for the default commit message.
            timestamp = datetime.now().astimezone().strftime(
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                "%Y-%m-%d %H:%M:%S %Z"
            # Close the current function call, tuple, or grouped expression.
            )

            # Store the Git commit message supplied by the caller or generated from the timestamp.
            commit_message = (
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                f"Update {project_folder.name} simulation - "
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                f"{timestamp}"
            # Close the current function call, tuple, or grouped expression.
            )

        # Call `_run_git`; the following indented continuation lines provide its arguments.
        _run_git(
            # Begin the grouped expression or collection continued on the following lines.
            [
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                "commit",
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                "-m",
                # Pass `commit_message` as the next value required by the surrounding call or collection.
                commit_message,
            # Close the current list or index expression.
            ],
            # Pass `repository_root` as the next value required by the surrounding call or collection.
            repository_root,
        # Close the current function call, tuple, or grouped expression.
        )

        # Print this progress or result message in the terminal.
        print("Created a new local simulation commit.")
    # Run the following block when the preceding condition was false.
    else:
        # Print this progress or result message in the terminal.
        print(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "No new simulation files require a commit."
        # Close the current function call, tuple, or grouped expression.
        )
        # Print this progress or result message in the terminal.
        print(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Continuing to push and verify any earlier local commit."
        # Close the current function call, tuple, or grouped expression.
        )

    # Protect tracked, unstaged work elsewhere in the repository.
    # Git restores that work after the rebase.
    if _remote_branch_exists(
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
        # Pass `remote_name` as the next value required by the surrounding call or collection.
        remote_name,
        # Pass `branch` as the next value required by the surrounding call or collection.
        branch,
    # Begin the indented block controlled by this statement.
    ):
        # Call `_run_git`; the following indented continuation lines provide its arguments.
        _run_git(
            # Begin the grouped expression or collection continued on the following lines.
            [
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                "pull",
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                "--rebase",
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                "--autostash",
                # Pass `remote_name` as the next value required by the surrounding call or collection.
                remote_name,
                # Pass `branch` as the next value required by the surrounding call or collection.
                branch,
            # Close the current list or index expression.
            ],
            # Pass `repository_root` as the next value required by the surrounding call or collection.
            repository_root,
        # Close the current function call, tuple, or grouped expression.
        )

    # Call `_run_git`; the following indented continuation lines provide its arguments.
    _run_git(
        # Begin the grouped expression or collection continued on the following lines.
        [
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "push",
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "--set-upstream",
            # Pass `remote_name` as the next value required by the surrounding call or collection.
            remote_name,
            # Pass `branch` as the next value required by the surrounding call or collection.
            branch,
        # Close the current list or index expression.
        ],
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
    # Close the current function call, tuple, or grouped expression.
    )

    # Read the remote commit after pushing to confirm that it matches the local commit.
    verified_sha = _verify_remote_commit(
        # Pass `repository_root` as the next value required by the surrounding call or collection.
        repository_root,
        # Pass `remote_name` as the next value required by the surrounding call or collection.
        remote_name,
        # Pass `branch` as the next value required by the surrounding call or collection.
        branch,
    # Close the current function call, tuple, or grouped expression.
    )

    # Print this progress or result message in the terminal.
    print("\nGitHub push completed and remotely verified.")
    # Print this progress or result message in the terminal.
    print("Repository:", EXPECTED_REPOSITORY)
    # Print this progress or result message in the terminal.
    print("Branch:", branch)
    # Print this progress or result message in the terminal.
    print("Verified commit:", verified_sha[:12])
    # Print this progress or result message in the terminal.
    print("Simulation folder:", project_scope)

    # Return this value to the code that called the current function.
    return True
