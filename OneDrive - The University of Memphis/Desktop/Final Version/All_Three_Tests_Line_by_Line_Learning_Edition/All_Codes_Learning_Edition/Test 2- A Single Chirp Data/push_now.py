# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Push saved code and results without rerunning the simulation."""

# Import selected names from pathlib instead of importing its complete namespace.
from pathlib import Path

# Import selected names from github_push instead of importing its complete namespace.
from github_push import GitPushError, push_simulation


# Store the folder containing the current script so every path is relative to the project.
HERE = Path(__file__).resolve().parent


# Use the expression `try:` as the next part of the surrounding Python statement.
try:
    # Call `push_simulation`; the following indented continuation lines provide its arguments.
    push_simulation(
        # Pass `HERE` as the next value required by the surrounding call or collection.
        HERE,
        # Store the Git commit message supplied by the caller or generated from the timestamp.
        commit_message="Push Final LSTM simulation code and results",
    # Close the current function call, tuple, or grouped expression.
    )
# Stop with a readable message when Git cannot complete the push.
except GitPushError as error:
    # Print this progress or result message in the terminal.
    print("\nGitHub push failed:")
    # Print this progress or result message in the terminal.
    print(error)
    # Stop this operation and report the stated error condition.
    raise SystemExit(1) from error
