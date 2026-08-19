"""
Grey box physical model of the actuator.

Why this module exists
----------------------
The first version of this project asked the network to predict displacement
from the coil current alone, sitting on top of an algebraic quasi static
baseline. That failed badly on the held out records, and the zero current
records showed exactly why: the true response is a lightly damped 6 to 9 Hz
ring down set by the initial condition, while the coil current is identically
zero for the whole record. There is no information in the input from which any
network could reconstruct that oscillation. It learned an average decay
instead, which is the smooth curve seen in the earlier figures.

The oscillation is not mysterious, it is Newton's second law:

    m x'' = Bl(x) i - m g - c x' - Fs(x)

with x(0) = 0 and x'(0) = 0. Simulating that equation reproduces the ring
down exactly, and because the mass appears explicitly it also extrapolates to
a load mass that was never trained on, which a purely data driven model cannot
do.

What is identified, and how
---------------------------
Three unknowns are identified from training samples only.

  Fs(x)   the spring characteristic. It is strongly asymmetric: about
          19.6 N/m at -1.7 mm but only 6.6 N/m at +1.9 mm, where the force
          saturates. An odd polynomial cannot represent that, so a general
          polynomial in x is used, with linear extrapolation outside the
          range that the training records actually visited.
  c       viscous damping. It came out at 0.110 Ns/m and, usefully, is the
          same for all three load masses.
  Bl(x)   the force constant, which varies with coil position from about
          0.476 N/A near centre down to 0.375 N/A at +1.9 mm.

The identification is direct rather than iterative. Rearranging the equation
of motion,

    Fs(x) + c x'  =  F_measured - m g - m x''

Everything on the right is measured, and both unknowns on the left are linear
in their parameters, so a single linear least squares solve recovers the
spring polynomial and the damping together. No optimiser, no initial guess,
no convergence to worry about.

One numerical detail matters. The derivatives x' and x'' are taken with a
Savitzky Golay filter, and the window has to stay short. The sine records are
excited at 50 Hz, far above the 6 to 12 Hz resonance, and a 21 sample window
smooths that away completely: it inflated the force balance residual by a
factor of six and biased the damping estimate. Five samples is enough for
data this clean.
"""

import numpy as np
from scipy.signal import savgol_filter


GRAVITY = 9.81

# Displacement is carried in millimetres inside the polynomials. Fitting a
# high order polynomial against metre scale values (around 1e-3) produces a
# hopelessly ill conditioned Vandermonde matrix.
MM_PER_M = 1000.0

SPRING_DEGREE = 9
BL_DEGREE = 3

DERIVATIVE_WINDOW = 5
DERIVATIVE_ORDER = 3


def state_derivatives(displacement_m, time_step):
    """Velocity and acceleration of a clean simulated displacement signal."""
    window = min(DERIVATIVE_WINDOW, len(displacement_m) - (1 - len(displacement_m) % 2))
    if window < DERIVATIVE_ORDER + 2:
        velocity = np.gradient(displacement_m, time_step)
        return velocity, np.gradient(velocity, time_step)

    velocity = savgol_filter(
        displacement_m, window, DERIVATIVE_ORDER, deriv=1, delta=time_step
    )
    acceleration = savgol_filter(
        displacement_m, window, DERIVATIVE_ORDER, deriv=2, delta=time_step
    )
    return velocity, acceleration


def identify(records, training_masks, time_step):
    """
    Recover the spring curve, the damping and the force constant.

    records         every development record
    training_masks  boolean mask per record marking the samples the training
                    split is allowed to use. Validation and internal test
                    blocks are excluded here exactly as they are everywhere
                    else, so the physical model cannot quietly learn from
                    data the network is being scored on.
    """
    spring_rows = []
    spring_target = []
    spring_weight = []

    force_rows = []
    force_target = []
    force_weight = []

    for record, mask in zip(records, training_masks):
        if not np.any(mask):
            continue

        pad = record["pad"]
        displacement = record["outputs"][pad:, 0].astype(np.float64) / MM_PER_M
        measured_force = record["outputs"][pad:, 1].astype(np.float64)
        current = record["current"][pad:].astype(np.float64)
        mass = record["total_mass_g"] / 1000.0

        velocity, acceleration = state_derivatives(displacement, time_step)

        selected = mask[pad:]
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        weight = np.full(count, 1.0 / count)

        position_mm = displacement[selected] * MM_PER_M

        # Fs(x) + c x' = F - m g - m x''
        design = np.column_stack(
            [np.vander(position_mm, SPRING_DEGREE + 1), velocity[selected]]
        )
        target = (
            measured_force[selected]
            - mass * GRAVITY
            - mass * acceleration[selected]
        )
        spring_rows.append(design)
        spring_target.append(target)
        spring_weight.append(weight)

        # F = i * (b0 + b1 x + b2 x^2 + b3 x^3)
        force_rows.append(
            np.column_stack(
                [current[selected] * position_mm ** power
                 for power in range(BL_DEGREE + 1)]
            )
        )
        force_target.append(measured_force[selected])
        force_weight.append(weight)

    if not spring_rows:
        raise RuntimeError("No training samples were available to identify the "
                           "physical model.")

    def weighted_solve(rows, targets, weights):
        design = np.concatenate(rows)
        target = np.concatenate(targets)
        weight = np.concatenate(weights)
        root = np.sqrt(weight)[:, None]
        solution, *_ = np.linalg.lstsq(design * root, target * np.sqrt(weight),
                                       rcond=None)
        residual = target - design @ solution
        scatter = float(np.sqrt(np.average(residual ** 2, weights=weight)))
        return solution, scatter

    spring_solution, spring_scatter = weighted_solve(
        spring_rows, spring_target, spring_weight
    )
    force_solution, force_scatter = weighted_solve(
        force_rows, force_target, force_weight
    )

    spring_coefficients = spring_solution[:SPRING_DEGREE + 1]
    damping = float(spring_solution[-1])

    visited = np.concatenate(
        [
            record["outputs"][record["pad"]:, 0][mask[record["pad"]:]]
            for record, mask in zip(records, training_masks)
            if np.any(mask)
        ]
    ).astype(np.float64)

    return {
        "spring_coefficients": spring_coefficients,
        "spring_derivative": np.polyder(spring_coefficients),
        "damping": damping,
        "bl_coefficients": force_solution,
        "position_low_mm": float(visited.min()),
        "position_high_mm": float(visited.max()),
        "spring_scatter_n": spring_scatter,
        "force_scatter_n": force_scatter,
    }


def spring_force(position_mm, parameters):
    """
    Spring force in newtons.

    Outside the displacement range that the training records actually reached,
    the curve continues along its end tangent. A high order polynomial left to
    its own devices diverges violently once extrapolated, and the heaviest
    load does reach a little beyond the training range.
    """
    clamped = np.clip(
        position_mm, parameters["position_low_mm"], parameters["position_high_mm"]
    )
    inside = np.polyval(parameters["spring_coefficients"], clamped)

    # The measured characteristic genuinely turns over past about +1.9 mm:
    # the restoring force stops growing and begins to fall. That is real and
    # is kept inside the fitted range. Continuing a negative slope outside the
    # range would be a runaway though, because further travel would then meet
    # even less resistance, so the extrapolation slope is floored at zero.
    slope = np.maximum(np.polyval(parameters["spring_derivative"], clamped), 0.0)
    return inside + (position_mm - clamped) * slope


def force_constant(position_mm, parameters):
    """Position dependent force constant Bl(x) in N/A."""
    clamped = np.clip(
        position_mm, parameters["position_low_mm"], parameters["position_high_mm"]
    )
    coefficients = parameters["bl_coefficients"]
    value = np.zeros_like(clamped, dtype=np.float64)
    for power, coefficient in enumerate(coefficients):
        value = value + coefficient * clamped ** power
    return value


def simulate(record, parameters, time_step):
    """
    Integrate the equation of motion for one record with Runge Kutta 4.

    Returns displacement in millimetres, velocity in millimetres per second
    and the actuator force in newtons, on the same padded grid the record
    uses. The padded samples sit before the simulation started, so the model
    is simply held at rest there, which is exactly the state COMSOL began
    from.
    """
    current = record["current"].astype(np.float64)
    pad = record["pad"]
    total = len(current)
    mass = record["total_mass_g"] / 1000.0
    damping = parameters["damping"]

    displacement = np.zeros(total)
    velocity = np.zeros(total)
    force = np.zeros(total)

    def acceleration(position, speed, coil_current):
        position_mm = position * MM_PER_M
        actuator = force_constant(np.array(position_mm), parameters)[()] * coil_current
        return (
            actuator
            - mass * GRAVITY
            - damping * speed
            - spring_force(np.array(position_mm), parameters)[()]
        ) / mass

    position = 0.0
    speed = 0.0

    for index in range(pad):
        force[index] = (
            force_constant(np.array(0.0), parameters)[()] * current[index]
        )

    for index in range(pad, total):
        displacement[index] = position * MM_PER_M
        velocity[index] = speed * MM_PER_M
        force[index] = (
            force_constant(np.array(position * MM_PER_M), parameters)[()]
            * current[index]
        )

        if index == total - 1:
            break

        start_current = current[index]
        end_current = current[index + 1]
        mid_current = 0.5 * (start_current + end_current)

        a1 = acceleration(position, speed, start_current)
        a2 = acceleration(position + 0.5 * time_step * speed,
                          speed + 0.5 * time_step * a1, mid_current)
        a3 = acceleration(position + 0.5 * time_step * (speed + 0.5 * time_step * a1),
                          speed + 0.5 * time_step * a2, mid_current)
        a4 = acceleration(position + time_step * (speed + 0.5 * time_step * a2),
                          speed + time_step * a3, end_current)

        position = position + time_step * (speed + time_step / 6.0 * (a1 + a2 + a3))
        speed = speed + time_step / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)

        if not (np.isfinite(position) and np.isfinite(speed)):
            raise FloatingPointError(
                f"The physical model diverged while simulating "
                f"{record.get('name', 'a record')}. Check the identified "
                "spring curve."
            )

    return (
        displacement.astype(np.float32),
        velocity.astype(np.float32),
        force.astype(np.float32),
    )


def describe(parameters):
    """Printable summary of the identified physical model."""
    lines = [
        f"  damping c            = {parameters['damping']:.5f} Ns/m",
        f"  Bl at centre         = {parameters['bl_coefficients'][0]:.4f} N/A",
        f"  spring fitted over     {parameters['position_low_mm']:.3f} to "
        f"{parameters['position_high_mm']:.3f} mm, tangent continued outside",
        f"  force balance scatter= {parameters['spring_scatter_n']:.5f} N",
    ]
    for position in (-5.0, -2.0, 0.0, 2.0):
        stiffness = np.polyval(parameters["spring_derivative"], position) * MM_PER_M
        lines.append(
            f"  stiffness at {position:+5.1f} mm  = {stiffness:7.3f} N/m"
        )
    return "\n".join(lines)
