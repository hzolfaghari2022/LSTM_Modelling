"""Entry point copied into F01 through F13."""

from pathlib import Path
import re

from incremental_case import run_case


HERE = Path(__file__).resolve().parent
match = re.fullmatch(r"F(\d{2})", HERE.name)
if match is None:
    raise RuntimeError("Case folder must be named F01, F02, ..., F13.")

run_case(int(match.group(1)), HERE)

