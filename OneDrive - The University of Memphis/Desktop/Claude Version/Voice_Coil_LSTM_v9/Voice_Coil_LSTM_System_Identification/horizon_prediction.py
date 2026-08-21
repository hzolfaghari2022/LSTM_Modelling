"""
k step ahead prediction, alongside the free running simulation.

Why this exists.

The pipeline reports a *simulation*: the model is given the coil current and
nothing else, started from rest, and integrated to the end of the record. For a
lightly damped resonator that is the hardest possible test. The heaviest load
rings at 6.2 Hz, so over a 1.6 s record a natural frequency that is off by only
1 percent accumulates about 36 degrees of phase. The predicted curve then has
the right shape and the right envelope but sits beside the measurement instead
of on top of it, and RMSE punishes that heavily even though the dynamics are
close.

k step ahead prediction is the other standard way to report an identified
model. The integration is restarted from the measured state every k samples, so
the error shown is what the model accumulates over k samples rather than over
the whole record. Phase drift cannot build up.

Both numbers are meaningful and they answer different questions:

    simulation        can the model replace COMSOL, given only the current?
    k step ahead      are the identified dynamics locally correct?

Report both. A model that is excellent at k step ahead and poor at simulation
has correct local physics and a small parameter error somewhere; that is a very
different diagnosis from one that fails at both.
"""

import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

import config
import grey_box
from data_utils import prepare_data

MM_PER_M = 1000.0


def simulate_with_resets(record, parameters, time_step, horizon):
    """Integrate, restarting from the measured state every `horizon` samples."""
    (spring, derivative, force_poly,
     low, high, damping) = grey_box._compiled_parameters(parameters)

    current = np.asarray(record["current"], dtype=np.float64)
    pad = record["pad"]
    total = len(current)
    mass = record["total_mass_g"] / 1000.0
    weight = mass * grey_box.GRAVITY

    measured_mm = np.asarray(record["outputs"][:, 0], dtype=np.float64)
    # The reset velocity comes from differentiating the measured displacement,
    # so the smoothing window has to be short compared with the fastest content
    # in the record. A fixed 31 ms window spans more than one period of the
    # 37 Hz drive in the DC plus sine records and destroys the derivative,
    # which then poisons every restart.
    window = min(9, len(measured_mm) - (1 - len(measured_mm) % 2))
    velocity_mm = savgol_filter(measured_mm, window, 2, deriv=1, delta=time_step)

    displacement = np.zeros(total)
    force = np.zeros(total)

    def acceleration(position_m, speed_m, drive):
        position_mm = position_m * MM_PER_M
        clamped = low if position_mm < low else (high if position_mm > high else position_mm)
        inside = 0.0
        for coefficient in spring:
            inside = inside * clamped + coefficient
        slope = 0.0
        for coefficient in derivative:
            slope = slope * clamped + coefficient
        if slope < 0.0:
            slope = 0.0
        restoring = inside + (position_mm - clamped) * slope
        constant = 0.0
        for coefficient in reversed(force_poly):
            constant = constant * clamped + coefficient
        return (constant * drive - weight - damping * speed_m - restoring) / mass

    def coil_force(position_m, drive):
        position_mm = position_m * MM_PER_M
        clamped = low if position_mm < low else (high if position_mm > high else position_mm)
        constant = 0.0
        for coefficient in reversed(force_poly):
            constant = constant * clamped + coefficient
        return constant * drive

    position = 0.0
    speed = 0.0
    half = 0.5 * time_step

    for index in range(pad, total):
        if (index - pad) % horizon == 0:
            position = measured_mm[index] / MM_PER_M
            speed = velocity_mm[index] / MM_PER_M

        displacement[index] = position * MM_PER_M
        force[index] = coil_force(position, current[index])

        start_drive = current[index]
        end_drive = current[min(index + 1, total - 1)]
        mid_drive = 0.5 * (start_drive + end_drive)

        v1 = speed
        a1 = acceleration(position, speed, start_drive)
        v2 = speed + half * a1
        a2 = acceleration(position + half * v1, v2, mid_drive)
        v3 = speed + half * a2
        a3 = acceleration(position + half * v2, v3, mid_drive)
        v4 = speed + time_step * a3
        a4 = acceleration(position + time_step * v3, v4, end_drive)

        position += time_step / 6.0 * (v1 + 2 * v2 + 2 * v3 + v4)
        speed += time_step / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)

    return displacement, force


def score(measured, predicted):
    error = measured - predicted
    spread = measured - measured.mean()
    total = float(np.sum(spread ** 2))
    r2 = 1.0 - float(np.sum(error ** 2)) / total if total > 0 else np.nan
    return float(np.sqrt(np.mean(error ** 2))), r2


if __name__ == "__main__":
    data = prepare_data(Path(__file__).resolve().parent)
    step = config.TARGET_TIME_STEP
    parameters = data["physical_model"]

    horizons = [int(round(seconds / step)) for seconds in (0.02, 0.05, 0.10)]

    print()
    print(f"{'record':30s} {'free run':>18s} "
          f"{'20 ms':>16s} {'50 ms':>16s} {'100 ms':>16s}")
    print(f"{'':30s} {'RMSE      R2':>18s} "
          f"{'RMSE     R2':>16s} {'RMSE     R2':>16s} {'RMSE     R2':>16s}")

    for record in data["records"]:
        if not record["is_pure_test"]:
            continue
        pad = record["pad"]
        measured = np.asarray(record["outputs"][pad:, 0], dtype=np.float64)

        free = grey_box.simulate_fast(record, parameters, step)[pad:]
        line = f"{record['name']:30s} " + "%8.4f %8.4f  " % score(measured, free)

        for horizon in horizons:
            predicted, _ = simulate_with_resets(record, parameters, step, horizon)
            line += "%7.4f %7.4f " % score(measured, predicted[pad:])
        print(line)
