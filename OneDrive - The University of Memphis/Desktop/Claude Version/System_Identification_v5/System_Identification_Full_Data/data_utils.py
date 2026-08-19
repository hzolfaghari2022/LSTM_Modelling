"""
Data loading, resampling, feature construction and splitting.

The whole pipeline is record based. A record is one continuous COMSOL
simulation with its own load mass, its own sample rate and its own
excitation. Records are discovered from the workbook itself by
workbook_scan.py, so no case in the file is ignored and nothing depends on
sheet names staying put between COMSOL exports.

Strict rule enforced here: the records chosen as pure tests never contribute
to the normalisation statistics, to the quasi static baseline fit, or to any
training or validation window.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    BLOCK_SIZE,
    DROP_DUPLICATE_RECORDS,
    PURE_TEST_RULES,
    COIL_MASS_GRAM,
    CONFIGURATION_FEATURE_INDICES,
    DC_ESTIMATOR_TIME_CONSTANT,
    FEATURE_NAMES,
    HEADER_ROW,
    PURE_TEST_STRIDE,
    REFERENCE_LOAD_MASS_GRAM,
    REFERENCE_TOTAL_MASS_GRAM,
    SCALED_TIME_CLIP_SECONDS,
    SEQUENCE_LENGTH,
    SPLIT_PATTERN,
    STARTUP_TIME_CONSTANT,
    TARGET_SAMPLE_RATE_HZ,
    TARGET_TIME_STEP,
    TEST_STRIDE,
    TRAIN_STRIDE,
    USE_GREY_BOX,
    USE_STATIC_BASELINE,
    VALIDATION_STRIDE,
    WORKBOOK_PATTERN,
)
import grey_box
from workbook_scan import find_duplicate_records, name_records, scan_workbook


STRIDE_BY_ROLE = {
    "training": TRAIN_STRIDE,
    "validation": VALIDATION_STRIDE,
    "test": TEST_STRIDE,
}


# ----------------------------------------------------------------------
# Workbook access
# ----------------------------------------------------------------------

def _candidate_workbooks(folder: Path):
    """Every place the workbook is reasonably allowed to live, in priority order."""
    folder = Path(folder).resolve()
    searched = [
        folder,                 # beside main.py, the normal case
        folder.parent,          # one level up, e.g. when the zip added a wrapper folder
        folder.parent.parent,   # two levels up, same reason
        Path.cwd().resolve(),   # wherever python was launched from
    ]

    ordered = []
    for place in searched:
        if place not in ordered:
            ordered.append(place)
    return ordered


def find_workbook(folder: Path) -> Path:
    """
    Locate the data workbook.

    The explicit path in the DLSTM_WORKBOOK environment variable wins if it is
    set. Otherwise the folder holding main.py is searched first, then the two
    folders above it, which covers the common case of an archive that unpacked
    with an extra wrapper folder around the code.

    Excel lock files (the ~$ prefixed ones left behind while a workbook is
    open) are ignored, and so are the temporary copies OneDrive leaves during
    a sync conflict, because reading either of those produces a confusing
    failure much further downstream.
    """
    override = os.environ.get("DLSTM_WORKBOOK", "").strip().strip('"')
    if override:
        chosen = Path(override).expanduser()
        if not chosen.is_file():
            raise FileNotFoundError(
                f"DLSTM_WORKBOOK points at {chosen}, which is not a file."
            )
        return chosen

    searched = _candidate_workbooks(folder)

    for place in searched:
        matches = [
            candidate
            for candidate in sorted(place.glob(WORKBOOK_PATTERN))
            if not candidate.name.startswith("~$")
            and candidate.is_file()
        ]
        if matches:
            chosen = max(matches, key=lambda file: file.stat().st_mtime)
            if len(matches) > 1:
                print(
                    f"More than one workbook matches {WORKBOOK_PATTERN} in "
                    f"{place}. Using the most recently modified one:"
                )
                for candidate in matches:
                    marker = "  <-- using" if candidate == chosen else ""
                    print(f"    {candidate.name}{marker}")
                print(
                    "  Set DLSTM_WORKBOOK to an explicit path to choose a "
                    "different one."
                )
            return chosen

    # Nothing found. Say exactly where the search looked and what is actually
    # sitting there, because "file not found" on its own is rarely enough.
    lines = [
        f"No workbook matching {WORKBOOK_PATTERN} was found.",
        "",
        "Searched these folders, in order:",
    ]
    for place in searched:
        lines.append(f"  {place}")

    nearby = sorted(
        {
            candidate.name
            for place in searched
            for candidate in place.glob("*.xls*")
            if not candidate.name.startswith("~$")
        }
    )
    if nearby:
        lines.append("")
        lines.append("Spreadsheets that are present but do not match the pattern:")
        for name in nearby:
            lines.append(f"  {name}")
        lines.append("")
        lines.append(
            "Rename the one you want to Total_Data.xlsx, or set the "
            "DLSTM_WORKBOOK environment variable to its full path."
        )
    else:
        lines.append("")
        lines.append(
            "Copy your Total_Data.xlsx into the folder that holds main.py. "
            "It is deliberately not shipped with the code, so that a stale "
            "copy can never silently override the workbook you meant to use."
        )

    raise FileNotFoundError("\n".join(lines))


def read_sheet_frame(workbook: Path, sheet_name: str, cache: dict):
    """
    Read one worksheet once and keep it in a cache.

    The sheet is read with no header at all. Text rows above the data turn
    into missing values and are dropped later, which makes the reader immune
    to a metadata block that grows or shrinks by a row between exports.
    """
    if sheet_name not in cache:
        cache[sheet_name] = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
            engine="openpyxl",
        )
    return cache[sheet_name]


def resample_to_uniform_grid(time, columns, target_rate=TARGET_SAMPLE_RATE_HZ):
    """
    Place a set of signals on a uniform time grid.

    The COMSOL exports are dense compared with the dynamics of interest
    (the fastest content in the data is around 200 Hz while the slowest
    export rate is 500 Hz), so straight interpolation onto the common grid
    introduces no visible error and, unlike polyphase resampling, it leaves
    no ringing at the ends of a step or of a settled displacement.
    """
    start = float(time[0])
    stop = float(time[-1])
    number_of_samples = int(np.floor((stop - start) * target_rate)) + 1
    uniform_time = start + np.arange(number_of_samples) / target_rate
    resampled = [
        np.interp(uniform_time, time, column).astype(np.float32)
        for column in columns
    ]
    return uniform_time.astype(np.float32), resampled


def load_record(workbook: Path, specification: dict, cache: dict) -> dict:
    """Load one case from the workbook and return it on the common grid."""
    frame = read_sheet_frame(workbook, specification["sheet"], cache)

    first = specification["first_column"]
    if first + 4 > frame.shape[1]:
        raise ValueError(
            f"Sheet '{specification['sheet']}' has {frame.shape[1]} columns, "
            f"but a case was expected at columns {first} to {first + 3}. "
            "The workbook layout does not match what was discovered in it."
        )

    block = frame.iloc[:, first:first + 4].copy()
    block.columns = ["time", "displacement", "current", "force"]

    block = block.apply(pd.to_numeric, errors="coerce").dropna()
    block = (
        block.sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )

    if len(block) < 8:
        raise ValueError(
            f"Record {specification['name']} contains too few usable rows."
        )

    raw_time = block["time"].to_numpy(np.float64)
    native_rate = 1.0 / float(np.median(np.diff(raw_time)))

    time, (displacement, current, force) = resample_to_uniform_grid(
        raw_time,
        [
            block["displacement"].to_numpy(np.float64),
            block["current"].to_numpy(np.float64),
            block["force"].to_numpy(np.float64),
        ],
    )

    outputs = np.column_stack([displacement, force]).astype(np.float32)

    # Prepend the rest state that the simulation started from.
    #
    # A window of SEQUENCE_LENGTH samples is needed before the network can
    # produce its first output. Without this padding the first 0.2 s of every
    # record would be unavailable, which for the 0.8 s transient records is
    # the entire step response. Every COMSOL case starts from rest with the
    # coil current at its initial value, so repeating that state backwards in
    # time is exact rather than an approximation.
    pad = SEQUENCE_LENGTH - 1
    if pad > 0:
        pad_time = time[0] - np.arange(pad, 0, -1, dtype=np.float32) / (
            TARGET_SAMPLE_RATE_HZ
        )
        time = np.concatenate([pad_time.astype(np.float32), time])
        current = np.concatenate(
            [np.full(pad, current[0], dtype=np.float32), current]
        )
        outputs = np.concatenate([np.repeat(outputs[:1], pad, axis=0), outputs])

    total_mass = COIL_MASS_GRAM + specification["load_mass_g"]

    record = dict(specification)
    record.update(
        {
            "time": time,
            "current": current,
            "outputs": outputs,
            "total_mass_g": total_mass,
            "mass_ratio": total_mass / REFERENCE_TOTAL_MASS_GRAM,
            "native_rate_hz": native_rate,
            "pad": pad,
            "duration_s": float(time[-1] - time[pad]),
            "samples": int(len(time)),
            "real_samples": int(len(time)) - pad,
            "is_pure_test": False,
        }
    )
    return record


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------

def causal_dc_estimate(current, time_step=TARGET_TIME_STEP,
                       time_constant=DC_ESTIMATOR_TIME_CONSTANT):
    """
    First order causal low pass of the coil current.

    A causal estimator is used instead of the mean of the whole record so
    that the feature can also be produced online, and so that a record can
    never see its own future.
    """
    alpha = time_step / (time_constant + time_step)
    output = np.empty_like(current, dtype=np.float32)
    state = 0.0
    for index in range(len(current)):
        state += alpha * (float(current[index]) - state)
        output[index] = state
    return output


def build_features(time, current, mass_ratio, start_time=None, model_state=None):
    """Assemble the input channels described in config.FEATURE_NAMES."""
    current = np.asarray(current, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    if start_time is None:
        start_time = float(time[0])

    inverse_mass_ratio = 1.0 / mass_ratio
    frequency_ratio = float(np.sqrt(inverse_mass_ratio))

    current_rate = np.diff(current, prepend=current[0]).astype(np.float32)
    current_dc = causal_dc_estimate(current)

    # The padded samples sit at negative time. Clamping the elapsed time at
    # zero keeps the startup channels well defined across the padding.
    elapsed = np.maximum(time - start_time, 0.0)
    scaled_time = np.clip(
        elapsed * frequency_ratio,
        0.0,
        SCALED_TIME_CLIP_SECONDS,
    ).astype(np.float32)
    startup = np.exp(-elapsed / STARTUP_TIME_CONSTANT).astype(np.float32)

    ones = np.ones_like(current, dtype=np.float32)

    if model_state is None:
        model_displacement = np.zeros_like(current)
        model_velocity = np.zeros_like(current)
        model_force = np.zeros_like(current)
    else:
        model_displacement, model_velocity, model_force = model_state

    features = np.column_stack(
        [
            current,
            current_rate,
            current_dc,
            current * inverse_mass_ratio,
            current_dc * inverse_mass_ratio,
            ones * mass_ratio,
            ones * inverse_mass_ratio,
            ones * frequency_ratio,
            scaled_time,
            startup,
            model_displacement,
            model_velocity,
            model_force,
        ]
    ).astype(np.float32)

    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("Feature matrix does not match FEATURE_NAMES.")

    return features


def baseline_design_matrices(record, features):
    """
    Regressors of the quasi static physical baseline.

    At equilibrium the actuator obeys  k x = Bl i - m g, so the resting
    position is linear in the current and linear in the moving mass:

        x_static = (Bl / k) i_dc - (g / k) m + constant

    Only two distinct load masses appear in the training pool, so no term
    beyond first order in the mass ratio can be identified. An interaction
    term such as i_dc times the mass ratio fits the training records just as
    well but extrapolates to a negative actuator gain at the heaviest load,
    which is physically impossible. It is therefore deliberately excluded.

    The force is essentially Bl times the instantaneous current. The real Bl
    varies slowly with coil position, and that part is left to the network.
    """
    current = features[:, 0]
    current_dc = features[:, 2]
    mass_ratio = features[:, 5]
    ones = np.ones_like(current)

    displacement_terms = np.column_stack(
        [current_dc, mass_ratio, ones]
    ).astype(np.float64)

    force_terms = np.column_stack([current, ones]).astype(np.float64)

    return displacement_terms, force_terms


def weighted_least_squares(design, target, weights):
    """Least squares with per sample weights."""
    root = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(
        design * root, target * np.sqrt(weights), rcond=None
    )
    return solution


# ----------------------------------------------------------------------
# Block splitting
# ----------------------------------------------------------------------

MINIMUM_BLOCK_SAMPLES = 40


def make_blocks(number_of_samples, first_sample=0):
    """
    Cut a development record into contiguous role blocks.

    Blocks are laid out over the real part of the record only. Windows never
    cross a block boundary, so a training sample can never leak into a
    validation or internal test sample. A block only has to be long enough to
    contain a few target samples, because the input history for its first
    target is allowed to reach back into the padding or into the block before
    it, and that history is input data rather than a label.
    """
    blocks = []
    for block_number, start in enumerate(
        range(first_sample, number_of_samples, BLOCK_SIZE)
    ):
        stop = min(start + BLOCK_SIZE, number_of_samples)
        role = (
            SPLIT_PATTERN[block_number % len(SPLIT_PATTERN)]
            if (stop - start) >= MINIMUM_BLOCK_SAMPLES
            else "unused_short_remainder"
        )
        blocks.append((block_number, start, stop, role))
    return blocks


def single_block(number_of_samples, role, first_sample=0):
    """Treat a whole record as one block, used for the pure test records."""
    return [(0, first_sample, number_of_samples, role)]


# ----------------------------------------------------------------------
# Window dataset
# ----------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Lazily sliced sequence windows.

    Windows are produced on demand from the per record feature arrays. This
    keeps memory flat even though the full data set holds close to eighty
    thousand samples across nineteen records.
    """

    def __init__(self, records, index_pairs, sequence_length=SEQUENCE_LENGTH):
        self.features = [record["features_normalised"] for record in records]
        self.targets = [record["targets_normalised"] for record in records]
        self.index_pairs = np.asarray(index_pairs, dtype=np.int64)
        self.sequence_length = sequence_length
        self.record_of_window = self.index_pairs[:, 0]

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, item):
        record_index, target_index = self.index_pairs[item]
        window = self.features[record_index][
            target_index - self.sequence_length + 1: target_index + 1
        ]
        target = self.targets[record_index][target_index]
        return torch.from_numpy(np.ascontiguousarray(window)), torch.from_numpy(
            np.ascontiguousarray(target)
        )


def collect_window_indices(records, requested_role):
    """Return (record index, target index) pairs for one role."""
    pairs = []
    for record_index, record in enumerate(records):
        for _, start, stop, role in record["blocks"]:
            if role != requested_role:
                continue
            for target_index in range(
                start, stop, STRIDE_BY_ROLE.get(requested_role, 1)
            ):
                if target_index < SEQUENCE_LENGTH - 1:
                    continue
                pairs.append((record_index, target_index))
    return pairs


def record_window_indices(records, record_index, stride=PURE_TEST_STRIDE):
    """
    Sweep one complete record.

    Thanks to the rest state padding this starts at the very first simulated
    sample, so the transient at the beginning of a step or a gravity release
    is part of the evaluation instead of being skipped.
    """
    record = records[record_index]
    first = max(record.get("pad", SEQUENCE_LENGTH - 1), SEQUENCE_LENGTH - 1)
    total = record["samples"]
    return [
        (record_index, target_index)
        for target_index in range(first, total, stride)
    ]


# ----------------------------------------------------------------------
# Pure test selection
# ----------------------------------------------------------------------

def select_pure_test_records(records):
    """
    Apply config.PURE_TEST_RULES to whatever the workbook turned out to hold.

    Selecting by rule rather than by name means the held out set stays
    physically the same set when the workbook is regenerated with the sheets
    in a different order.
    """
    masses = sorted({record["load_mass_g"] for record in records})
    heaviest = masses[-1]

    reference = min(masses, key=lambda mass: abs(mass - REFERENCE_LOAD_MASS_GRAM))

    def at_reference(family):
        return [
            record
            for record in records
            if record["family"] == family and record["load_mass_g"] == reference
        ]

    selected = []

    for rule in PURE_TEST_RULES:
        if rule == "heaviest_load":
            if len(masses) > 1:
                selected += [
                    record["name"]
                    for record in records
                    if record["load_mass_g"] == heaviest
                ]

        elif rule == "strongest_dc_chirp":
            chirps = [
                record
                for record in records
                if record["family"] in ("dc_plus_chirp", "chirp")
                and record["load_mass_g"] == reference
            ]
            if len(chirps) > 1:
                strongest = max(
                    chirps, key=lambda record: float(np.median(record["current"]))
                )
                selected.append(strongest["name"])

        elif rule == "reference_step":
            candidates = at_reference("step")
            if candidates:
                selected.append(candidates[0]["name"])

        elif rule == "reference_zero_input":
            candidates = at_reference("zero_input")
            if candidates:
                selected.append(candidates[0]["name"])

        else:
            raise ValueError(f"Unknown pure test rule: {rule}")

    ordered = []
    for name in selected:
        if name not in ordered:
            ordered.append(name)
    return ordered


# ----------------------------------------------------------------------
# Top level preparation
# ----------------------------------------------------------------------

def prepare_data(folder: Path):
    """Load every record, fit the training statistics and build the splits."""
    workbook = find_workbook(folder)
    cache = {}

    discovered = scan_workbook(workbook)
    records = [load_record(workbook, entry, cache) for entry in discovered]

    measurements = [
        {
            "dc_level_a": float(np.median(record["current"])),
            "duration_s": round(record["duration_s"], 4),
        }
        for record in records
    ]
    for record, name in zip(records, name_records(discovered, measurements)):
        record["name"] = name

    # ---- drop duplicated exports -------------------------------------
    duplicate_groups = find_duplicate_records(records)
    dropped = []
    if DROP_DUPLICATE_RECORDS:
        for group in duplicate_groups:
            for name in group[1:]:
                dropped.append(name)
        records = [record for record in records if record["name"] not in dropped]

    # ---- choose the pure test records by rule ------------------------
    pure_test_names = select_pure_test_records(records)
    for record in records:
        record["is_pure_test"] = record["name"] in pure_test_names

    remaining = [record for record in records if not record["is_pure_test"]]
    if not remaining:
        raise RuntimeError(
            "Every discovered record was selected as a pure test. "
            "Check PURE_TEST_RULES against the contents of the workbook."
        )

    index_of = {record["name"]: position for position, record in enumerate(records)}
    development_names = [record["name"] for record in remaining]

    # Roles. Development records are cut into blocks, pure test records are
    # kept whole and marked so that they can never be selected by role.
    for record in records:
        first = record["pad"]
        if record["is_pure_test"]:
            record["blocks"] = single_block(
                record["samples"], "pure_test", first_sample=first
            )
        else:
            record["blocks"] = make_blocks(record["samples"], first_sample=first)

    # ---- identify the physical model on training samples only --------
    # This has to happen before the features are built, because the state of
    # the physical model is itself fed to the network as three input channels.
    development_records = [record for record in records if not record["is_pure_test"]]
    training_masks = []
    for record in development_records:
        mask = np.zeros(record["samples"], dtype=bool)
        for _, start, stop, role in record["blocks"]:
            if role == "training":
                mask[start:stop] = True
        training_masks.append(mask)

    if USE_GREY_BOX:
        physical_model = grey_box.identify(
            development_records, training_masks, TARGET_TIME_STEP
        )
        for record in records:
            displacement, velocity, force = grey_box.simulate(
                record, physical_model, TARGET_TIME_STEP
            )
            record["model_state"] = (displacement, velocity, force)
    else:
        physical_model = None
        for record in records:
            zeros = np.zeros(record["samples"], dtype=np.float32)
            record["model_state"] = (zeros, zeros.copy(), zeros.copy())

    # Raw features for every record. The startup channels are referenced to
    # the first real sample, not to the first padded sample.
    for record in records:
        record["features"] = build_features(
            record["time"],
            record["current"],
            record["mass_ratio"],
            start_time=float(record["time"][record["pad"]]),
            model_state=record["model_state"],
        )

    # ---- statistics fitted on training samples only ----------------
    training_feature_rows = []
    training_target_rows = []
    training_weight_rows = []
    for record in records:
        if record["is_pure_test"]:
            continue
        record_rows = sum(
            stop - start
            for _, start, stop, role in record["blocks"]
            if role == "training"
        )
        if record_rows == 0:
            continue
        for _, start, stop, role in record["blocks"]:
            if role != "training":
                continue
            training_feature_rows.append(record["features"][start:stop])
            training_target_rows.append(record["outputs"][start:stop])
            training_weight_rows.append(
                np.full(stop - start, 1.0 / record_rows, dtype=np.float64)
            )

    if not training_feature_rows:
        raise RuntimeError("No training samples were produced.")

    training_features = np.concatenate(training_feature_rows).astype(np.float64)
    training_targets = np.concatenate(training_target_rows).astype(np.float64)
    training_weights = np.concatenate(training_weight_rows)

    # ---- quasi static baseline, fitted on training samples only -----
    # The fit is weighted so that every record counts equally. Without that
    # weighting the twenty second chirp would set the baseline on its own and
    # the short transient records would be fitted badly.
    if USE_STATIC_BASELINE:
        training_displacement_terms = np.column_stack(
            [
                training_features[:, 2],
                training_features[:, 5],
                np.ones(len(training_features)),
            ]
        )
        training_force_terms = np.column_stack(
            [training_features[:, 0], np.ones(len(training_features))]
        )
        displacement_coefficients = weighted_least_squares(
            training_displacement_terms, training_targets[:, 0], training_weights
        )
        force_coefficients = weighted_least_squares(
            training_force_terms, training_targets[:, 1], training_weights
        )
    else:
        displacement_coefficients = np.zeros(3)
        force_coefficients = np.zeros(2)

    for record in records:
        if USE_GREY_BOX:
            # The baseline is the simulated response of the physical model.
            # The network only has to supply what Newton's second law with a
            # single spring curve and one damping constant cannot explain.
            record["baseline"] = np.column_stack(
                [record["model_state"][0], record["model_state"][2]]
            ).astype(np.float32)
        else:
            displacement_terms, force_terms = baseline_design_matrices(
                record, record["features"]
            )
            record["baseline"] = np.column_stack(
                [
                    displacement_terms @ displacement_coefficients,
                    force_terms @ force_coefficients,
                ]
            ).astype(np.float32)
        record["residuals"] = (record["outputs"] - record["baseline"]).astype(
            np.float32
        )

    # ---- normalisation constants, again training samples only -------
    input_mean = training_features.mean(axis=0).astype(np.float32)
    input_std = (training_features.std(axis=0) + 1e-8).astype(np.float32)

    training_residual_rows = []
    for record in records:
        if record["is_pure_test"]:
            continue
        for _, start, stop, role in record["blocks"]:
            if role != "training":
                continue
            training_residual_rows.append(record["residuals"][start:stop])
    training_residuals = np.concatenate(training_residual_rows)

    target_mean = training_residuals.mean(axis=0).astype(np.float32)
    target_std = (training_residuals.std(axis=0) + 1e-8).astype(np.float32)

    for record in records:
        record["features_normalised"] = (
            (record["features"] - input_mean) / input_std
        ).astype(np.float32)
        record["targets_normalised"] = (
            (record["residuals"] - target_mean) / target_std
        ).astype(np.float32)

    # ---- window index lists -----------------------------------------
    training_pairs = collect_window_indices(records, "training")
    validation_pairs = collect_window_indices(records, "validation")
    internal_test_pairs = collect_window_indices(records, "test")

    if not training_pairs:
        raise RuntimeError("No training windows were produced.")
    if not validation_pairs:
        raise RuntimeError("No validation windows were produced.")
    if not internal_test_pairs:
        raise RuntimeError("No internal test windows were produced.")

    pure_test_pairs = {
        name: record_window_indices(records, index_of[name])
        for name in pure_test_names
    }

    # ---- split bookkeeping table -------------------------------------
    split_rows = []
    for record in records:
        for block_number, start, stop, role in record["blocks"]:
            split_rows.append(
                {
                    "record": record["name"],
                    "sheet": record["sheet"],
                    "first_column": record["first_column"],
                    "load_mass_g": record["load_mass_g"],
                    "total_mass_g": record["total_mass_g"],
                    "mass_ratio": record["mass_ratio"],
                    "family": record["family"],
                    "block": block_number,
                    "role": role,
                    "start_sample": start,
                    "stop_sample": stop,
                    "start_time_s": float(record["time"][start]),
                    "end_time_s": float(record["time"][stop - 1]),
                }
            )

    inventory_rows = [
        {
            "record": record["name"],
            "sheet": record["sheet"],
            "first_column": record["first_column"],
            "family": record["family"],
            "load_mass_g": record["load_mass_g"],
            "total_mass_g": record["total_mass_g"],
            "mass_ratio": round(record["mass_ratio"], 5),
            "native_rate_hz": round(record["native_rate_hz"], 2),
            "duration_s": round(record["duration_s"], 4),
            "samples_at_target_rate": record["real_samples"],
            "usage": "pure_test" if record["is_pure_test"] else "development",
            "description": record["description"],
        }
        for record in records
    ]

    return {
        "workbook": workbook,
        "records": records,
        "physical_model": physical_model,
        "pure_test_names": pure_test_names,
        "duplicate_groups": duplicate_groups,
        "dropped_duplicates": dropped,
        "index_of": index_of,
        "development_names": development_names,
        "split_table": pd.DataFrame(split_rows),
        "inventory_table": pd.DataFrame(inventory_rows),
        "training_pairs": training_pairs,
        "validation_pairs": validation_pairs,
        "internal_test_pairs": internal_test_pairs,
        "pure_test_pairs": pure_test_pairs,
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "displacement_baseline_coefficients": displacement_coefficients,
        "force_baseline_coefficients": force_coefficients,
        "configuration_feature_indices": CONFIGURATION_FEATURE_INDICES,
    }


def reconstruct_outputs(data, records, pairs, normalised_prediction):
    """
    Turn normalised residual predictions back into physical units and pick
    up the matching measurements and time stamps.
    """
    target_mean = data["target_mean"]
    target_std = data["target_std"]

    pairs = np.asarray(pairs, dtype=np.int64)
    residual = normalised_prediction * target_std + target_mean

    baseline = np.stack(
        [records[record_index]["baseline"][target_index]
         for record_index, target_index in pairs]
    )
    measured = np.stack(
        [records[record_index]["outputs"][target_index]
         for record_index, target_index in pairs]
    )
    time = np.asarray(
        [records[record_index]["time"][target_index]
         for record_index, target_index in pairs],
        dtype=np.float32,
    )

    predicted = (residual + baseline).astype(np.float32)
    return time, measured.astype(np.float32), predicted, baseline.astype(np.float32)


# ----------------------------------------------------------------------
# Synthetic probe signals
# ----------------------------------------------------------------------

def make_synthetic_record(name, load_mass_g, current, duration_seconds):
    """
    Build a record that has an input but no COMSOL ground truth.

    Used only for the closing physical sanity check of the frozen model.
    """
    samples = int(round(duration_seconds * TARGET_SAMPLE_RATE_HZ)) + 1
    current = np.asarray(current, dtype=np.float32)
    if len(current) != samples:
        raise ValueError("Synthetic current length does not match the duration.")

    pad = SEQUENCE_LENGTH - 1
    time = ((np.arange(samples + pad) - pad) * TARGET_TIME_STEP).astype(np.float32)
    current = np.concatenate([np.full(pad, current[0], dtype=np.float32), current])
    samples = samples + pad

    total_mass = COIL_MASS_GRAM + load_mass_g
    mass_ratio = total_mass / REFERENCE_TOTAL_MASS_GRAM

    return {
        "name": name,
        "sheet": "synthetic",
        "first_column": -1,
        "load_mass_g": load_mass_g,
        "family": "synthetic",
        "description": "Synthetic probe signal, no COMSOL reference",
        "time": time,
        "current": current,
        "outputs": np.zeros((samples, 2), dtype=np.float32),
        "total_mass_g": total_mass,
        "mass_ratio": mass_ratio,
        "native_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "pad": pad,
        "duration_s": float(time[-1]),
        "samples": samples,
        "real_samples": samples - pad,
        "is_pure_test": True,
    }


def attach_synthetic_arrays(data, record):
    """Give a synthetic record the same feature and baseline treatment."""
    if data.get("physical_model") is not None:
        record["model_state"] = grey_box.simulate(
            record, data["physical_model"], TARGET_TIME_STEP
        )
    else:
        zeros = np.zeros(record["samples"], dtype=np.float32)
        record["model_state"] = (zeros, zeros.copy(), zeros.copy())

    record["features"] = build_features(
        record["time"],
        record["current"],
        record["mass_ratio"],
        start_time=float(record["time"][record["pad"]]),
        model_state=record["model_state"],
    )
    if data.get("physical_model") is not None:
        record["baseline"] = np.column_stack(
            [record["model_state"][0], record["model_state"][2]]
        ).astype(np.float32)
    else:
        displacement_terms, force_terms = baseline_design_matrices(
            record, record["features"]
        )
        record["baseline"] = np.column_stack(
            [
                displacement_terms @ data["displacement_baseline_coefficients"],
                force_terms @ data["force_baseline_coefficients"],
            ]
        ).astype(np.float32)
    record["residuals"] = np.zeros_like(record["baseline"])
    record["features_normalised"] = (
        (record["features"] - data["input_mean"]) / data["input_std"]
    ).astype(np.float32)
    record["targets_normalised"] = np.zeros_like(record["baseline"])
    record["blocks"] = single_block(
        record["samples"], "pure_test", first_sample=record["pad"]
    )
    return record
