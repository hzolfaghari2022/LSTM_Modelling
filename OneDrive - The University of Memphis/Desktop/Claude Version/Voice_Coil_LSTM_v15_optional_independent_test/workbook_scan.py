"""
Automatic discovery of the records held in the COMSOL workbook.

The first version of this project hardcoded a table of sheet names and column
offsets. That turned out to be fragile: the workbook is regenerated from
COMSOL from time to time, and between two exports the sheet order changed,
a record was dropped and another was duplicated. A pipeline keyed on
"Sheet4" then either crashes or, much worse, silently trains on the wrong
configuration.

Every sheet in the workbook already describes itself. A metadata block sits
above the data with the coil mass, the load mass and one description per
case, and the column titles row marks where each case begins. This module
reads that block and derives the record list from the file itself, so
renaming or reordering sheets cannot change what the pipeline trains on.

Layout that is recognised:

    row 0            "Coil Mass=1.427 gram, Load Mass=3.813 gram ..."
    row 15           one description per case, at the case start column
    row 16           "Time (s)" "Displacement(mm)" "Coil Current (A)"
                     "Lorentz force (N)", repeated for each case
    row 17 onwards   numeric data
"""

import re
import unicodedata

import numpy as np
import pandas as pd


METADATA_ROWS = 18

TIME_COLUMN_PATTERN = re.compile(r"^\s*time\b", re.IGNORECASE)


def _clean(value):
    """Normalise a spreadsheet cell to a plain comparable string."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", text).strip()


def _find_number(pattern, text, default=None):
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return default
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return default


def classify_family(description):
    """
    Map a case description onto an excitation family.

    The families drive the held out test selection, so the order of these
    tests matters. "DC offset plus delayed chirp" has to be recognised as a
    chirp before the plain "DC offset" test can claim it.
    """
    text = _clean(description).lower().replace("_", " ")

    if "no current" in text or "zero current" in text:
        return "zero_input"
    if "chirp" in text:
        return "dc_plus_chirp" if "dc offset" in text else "chirp"
    if "sine only" in text:
        return "sine"
    if "dc offset" in text and "sine" in text:
        return "dc_plus_sine"
    if "dc offset" in text:
        return "step"
    if "sine" in text:
        return "sine"
    return "unknown"


FAMILY_LABEL = {
    "zero_input": "ZeroInput",
    "step": "Step",
    "sine": "Sine",
    "dc_plus_sine": "DCSine",
    "chirp": "Chirp",
    "dc_plus_chirp": "DCChirp",
    "unknown": "Case",
}


def scan_workbook(workbook):
    """
    Return one entry per case found in the workbook.

    Each entry carries the sheet and column offset needed to read it, the
    load mass, the excitation family and a name derived from the physics
    rather than from the sheet name.
    """
    excel = pd.ExcelFile(workbook, engine="openpyxl")
    discovered = []

    for sheet_name in excel.sheet_names:
        header = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
            nrows=METADATA_ROWS,
            engine="openpyxl",
        )
        if header.empty:
            continue

        metadata_text = " ".join(
            _clean(value)
            for value in header.to_numpy().ravel()
            if isinstance(value, str)
        )

        coil_mass = _find_number(r"coil\s*mass\s*=\s*([\d.]+)", metadata_text)
        load_mass = _find_number(r"load\s*mass\s*=\s*([\d.]+)", metadata_text)
        if load_mass is None:
            continue

        # A case begins wherever the column title row says "Time".
        title_row = header.iloc[16] if len(header) > 16 else header.iloc[-1]
        start_columns = [
            column
            for column in range(header.shape[1])
            if TIME_COLUMN_PATTERN.match(_clean(title_row.iloc[column]))
        ]
        if not start_columns:
            continue

        description_row = header.iloc[15] if len(header) > 15 else header.iloc[0]
        sheet_description = ""
        for column in range(header.shape[1]):
            text = _clean(description_row.iloc[column])
            if text:
                sheet_description = text
                break

        for start in start_columns:
            description = _clean(description_row.iloc[start]) or sheet_description
            discovered.append(
                {
                    "sheet": sheet_name,
                    "first_column": int(start),
                    "coil_mass_g": coil_mass,
                    "load_mass_g": float(load_mass),
                    "family": classify_family(description),
                    "description": description,
                    "dc_offset_a": _find_number(
                        r"dc[_ ]?offset\s*=\s*([\d.]+)", description
                    ),
                    "amplitude_a": _find_number(
                        r"amplitude\s*=?\s*([\d.]+)", description
                    ),
                    "stop_frequency_hz": _find_number(
                        r"stop\s*freq\w*\s*=\s*([\d.]+)", description
                    ),
                }
            )

    if not discovered:
        raise ValueError(
            f"No recognisable COMSOL cases were found in {workbook}. "
            "Each sheet needs a metadata block with a load mass and a "
            "column title row containing 'Time (s)'."
        )

    return discovered


def name_records(discovered, measurements):
    """
    Give every record a stable, physically meaningful name.

    Load masses are ranked so the lightest becomes Load1. A record keeps the
    same name when the workbook is regenerated with the sheets in a different
    order, which is the whole point of doing this from content.

    measurements supplies the DC level and duration measured from the data,
    used to separate records that share a family and a mass.
    """
    masses = sorted({entry["load_mass_g"] for entry in discovered})
    rank = {mass: position + 1 for position, mass in enumerate(masses)}

    names = []
    for entry, measured in zip(discovered, measurements):
        parts = [f"Load{rank[entry['load_mass_g']]}", FAMILY_LABEL[entry["family"]]]

        if entry["family"] in ("dc_plus_chirp", "chirp", "dc_plus_sine", "step"):
            level = entry["dc_offset_a"]
            if level is None:
                level = measured["dc_level_a"]
            if level is not None and abs(level) > 1e-6:
                parts.append(f"{round(level * 1000):d}mA")

        if entry["family"] in ("sine",) and entry["amplitude_a"]:
            parts.append(f"{round(entry['amplitude_a'] * 1000):d}mA")

        if entry["stop_frequency_hz"]:
            parts.append(f"{round(entry['stop_frequency_hz']):d}Hz")

        parts.append(f"{measured['duration_s']:g}s")
        names.append("_".join(parts))

    # Duplicate names mean duplicate records. Keep them distinguishable so
    # that the duplicate report downstream can point at a specific one.
    seen = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}_copy{seen[name] - 1}")

    return unique


def find_duplicate_records(records, tolerance=1e-9):
    """
    Report records whose data is identical.

    A duplicated export is not harmless. It doubles that record's weight in
    the training set and, if one copy lands in training while the other lands
    in a test split, it turns the test into a memorisation check that will
    look excellent for entirely the wrong reason.
    """
    groups = {}
    for record in records:
        signature = (
            record["real_samples"],
            round(float(np.sum(record["current"], dtype=np.float64)), 9),
            round(float(np.sum(record["outputs"][:, 0], dtype=np.float64)), 9),
            round(float(np.sum(record["outputs"][:, 1], dtype=np.float64)), 9),
        )
        groups.setdefault(signature, []).append(record)

    duplicates = []
    for group in groups.values():
        if len(group) < 2:
            continue
        reference = group[0]
        matched = [reference]
        for candidate in group[1:]:
            if candidate["outputs"].shape != reference["outputs"].shape:
                continue
            same = np.allclose(
                candidate["outputs"], reference["outputs"], atol=tolerance
            ) and np.allclose(
                candidate["current"], reference["current"], atol=tolerance
            )
            if same:
                matched.append(candidate)
        if len(matched) > 1:
            duplicates.append([record["name"] for record in matched])

    return duplicates
