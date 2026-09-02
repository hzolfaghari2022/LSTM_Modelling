"""Check that the shared V15 workbook can be read before training."""

from __future__ import annotations

from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
MODEL_FOLDER = HERE / "F13"
sys.path.insert(0, str(MODEL_FOLDER))

from workbook_scan import scan_workbook  # noqa: E402


def main() -> None:
    workbook = MODEL_FOLDER / "Total_Data.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")

    print(f"Checking: {workbook}")
    cache = {}
    started = time.perf_counter()
    records = scan_workbook(workbook, sheet_cache=cache)
    elapsed = time.perf_counter() - started
    print(f"\nWorkbook check passed: {len(records)} experiments discovered.")
    print(f"Worksheets parsed once and cached: {len(cache)}")
    print(f"Loading time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
