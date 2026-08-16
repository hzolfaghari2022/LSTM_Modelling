from __future__ import annotations


from datetime import datetime

from pathlib import Path

import re

import subprocess


EXPECTED_REPOSITORY = "hzolfaghari2022/LSTM_Modelling"


MAX_FILE_SIZE_BYTES = 95 * 1024 * 1024


class GitPushError(RuntimeError):

    pass


def _run_git(

    arguments: list[str],

    working_directory: Path,

    *,

    check: bool = True,

) -> subprocess.CompletedProcess[str]:


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


    result = _run_git(

        ["rev-parse", "--show-toplevel"],

        project_folder,

        check=False,

    )


    if result.returncode != 0:

        raise GitPushError(

            "This simulation folder is not inside a cloned Git "

            "repository.\n\n"

            "Clone hzolfaghari2022/LSTM_Modelling first, then place "

            "the Final folder anywhere inside that cloned repository."

        )


    return Path(result.stdout.strip()).resolve()


def _repository_slug_from_remote(

    remote_url: str,

) -> str:


    normalized = remote_url.strip().replace("\\", "/")


    if normalized.startswith("git@github.com:"):

        normalized = normalized.split(":", 1)[1]


    normalized = re.sub(

        r"^ssh://git@github\.com/",

        "",

        normalized,

        flags=re.IGNORECASE,

    )


    normalized = re.sub(

        r"^https?://github\.com/",

        "",

        normalized,

        flags=re.IGNORECASE,

    )


    normalized = re.sub(

        r"^git://github\.com/",

        "",

        normalized,

        flags=re.IGNORECASE,

    )


    if normalized.endswith(".git"):

        normalized = normalized[:-4]


    return normalized.strip("/")


def _check_repository_target(

    remote_url: str,

) -> None:


    actual_repository = _repository_slug_from_remote(

        remote_url

    )


    if actual_repository.lower() != EXPECTED_REPOSITORY.lower():

        raise GitPushError(

            "The local Git remote does not point to the expected "

            "repository.\n\n"

            f"Expected: {EXPECTED_REPOSITORY}\n"

            f"Detected: {actual_repository or remote_url}\n\n"

            "Correct the origin remote before pushing."

        )


def _check_for_very_large_files(

    project_folder: Path,

) -> None:


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

            f"  - {path} ({size / (1024 ** 2):.1f} MB)"

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


def _current_branch(

    repository_root: Path,

) -> str:


    branch = _run_git(

        ["branch", "--show-current"],

        repository_root,

    ).stdout.strip()


    if not branch:

        raise GitPushError(

            "Git is in a detached-HEAD state. Check out the main branch "

            "before running the simulation."

        )


    return branch


def _remote_branch_exists(

    repository_root: Path,

    remote_name: str,

    branch: str,

) -> bool:


    result = _run_git(

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


    return result.returncode == 0


def _verify_remote_commit(

    repository_root: Path,

    remote_name: str,

    branch: str,

) -> str:


    local_sha = _run_git(

        ["rev-parse", "HEAD"],

        repository_root,

    ).stdout.strip()


    remote_result = _run_git(

        [

            "ls-remote",

            remote_name,

            f"refs/heads/{branch}",

        ],

        repository_root,

    )


    remote_line = remote_result.stdout.strip()


    if not remote_line:

        raise GitPushError(

            "The push command returned successfully, but the remote "

            "branch could not be verified."

        )


    remote_sha = remote_line.split()[0]


    if local_sha != remote_sha:

        raise GitPushError(

            "Remote verification failed.\n\n"

            f"Local HEAD:  {local_sha}\n"

            f"Remote HEAD: {remote_sha}"

        )


    return local_sha


def push_simulation(

    project_folder: str | Path,

    *,

    commit_message: str | None = None,

    remote_name: str = "origin",

) -> bool:


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


    remote_url = remote_result.stdout.strip()

    _check_repository_target(remote_url)


    branch = _current_branch(

        repository_root

    )


    print("\nGitHub target verified.")

    print("Repository:", EXPECTED_REPOSITORY)

    print("Remote:", remote_url)

    print("Branch:", branch)

    print("Local repository:", repository_root)

    print("Simulation folder:", project_scope)


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


    _run_git(

        [

            "add",

            "--all",

            "--",

            project_scope,

        ],

        repository_root,

    )


    staged_names_result = _run_git(

        [

            "diff",

            "--cached",

            "--name-only",

            "--",

            project_scope,

        ],

        repository_root,

    )


    staged_names = [

        line.strip()

        for line in staged_names_result.stdout.splitlines()

        if line.strip()

    ]


    if staged_names:

        print(

            f"Staged {len(staged_names)} changed simulation file(s)."

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


        print("Created a new local simulation commit.")

    else:

        print(

            "No new simulation files require a commit."

        )

        print(

            "Continuing to push and verify any earlier local commit."

        )


    if _remote_branch_exists(

        repository_root,

        remote_name,

        branch,

    ):

        _run_git(

            [

                "pull",

                "--rebase",

                "--autostash",

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


    verified_sha = _verify_remote_commit(

        repository_root,

        remote_name,

        branch,

    )


    print("\nGitHub push completed and remotely verified.")

    print("Repository:", EXPECTED_REPOSITORY)

    print("Branch:", branch)

    print("Verified commit:", verified_sha[:12])

    print("Simulation folder:", project_scope)


    return True
