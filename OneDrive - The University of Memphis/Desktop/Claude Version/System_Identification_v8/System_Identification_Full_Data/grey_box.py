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


# ----------------------------------------------------------------------
# Simulation error refinement
# ----------------------------------------------------------------------

def _pack(parameters):
    return np.concatenate(
        [
            parameters["spring_coefficients"],
            parameters["bl_coefficients"],
            [parameters["damping"]],
        ]
    )


def _unpack(vector, template):
    spring_size = len(template["spring_coefficients"])
    force_size = len(template["bl_coefficients"])
    spring = vector[:spring_size]
    return {
        "spring_coefficients": spring,
        "spring_derivative": np.polyder(spring),
        "damping": float(vector[-1]),
        "bl_coefficients": vector[spring_size:spring_size + force_size],
        "position_low_mm": template["position_low_mm"],
        "position_high_mm": template["position_high_mm"],
        "spring_scatter_n": template.get("spring_scatter_n", 0.0),
        "force_scatter_n": template.get("force_scatter_n", 0.0),
    }


def refine_by_simulation(records, training_masks, parameters, time_step,
                         max_evaluations=200, max_samples_per_record=6000,
                         verbose=True):
    """
    Re-fit the physical parameters by minimising simulation error.

    identify() above solves the equation error problem: it differentiates the
    measured displacement twice and finds the coefficients that best satisfy
    Newton's second law sample by sample. That is one linear solve and it is
    fast, but it is also biased, because differentiating a sampled signal
    twice amplifies everything that is not smooth and the regressors then
    carry that noise.

    The bias is not academic. On a five parameter version of this model the
    equation error fit reproduced the training records to about 0.6 mm, while
    refitting the same five parameters against simulated trajectories brought
    that to about 0.16 mm, and the unseen heavy load improved in step.

    This function therefore takes the equation error answer as a starting
    point, shoots the model forward over the training records and adjusts the
    coefficients until the trajectories themselves line up. Long records are
    subsampled to keep the optimisation affordable; the parameters are
    physical constants, so a representative stretch identifies them as well as
    the whole record does.

    If the refinement fails to improve on the starting point, the starting
    point is returned unchanged.
    """
    from scipy.optimize import least_squares

    bundle = []
    for record, mask in zip(records, training_masks):
        if not np.any(mask):
            continue
        pad = record["pad"]
        measured = record["outputs"][pad:, 0].astype(np.float64)
        if len(measured) > max_samples_per_record:
            trimmed = dict(record)
            keep = pad + max_samples_per_record
            trimmed["current"] = record["current"][:keep]
            trimmed["outputs"] = record["outputs"][:keep]
            trimmed["samples"] = keep
            record = trimmed
            measured = measured[:max_samples_per_record]
        scale = float(np.std(measured)) + 1e-9
        bundle.append((record, measured, scale))

    if not bundle:
        return parameters

    def residual(vector):
        trial = _unpack(vector, parameters)
        pieces = []
        for record, measured, scale in bundle:
            simulated = simulate_fast(record, trial, time_step)
            error = simulated[record["pad"]:] - measured
            pieces.append(error / scale / np.sqrt(len(measured)))
        return np.concatenate(pieces)

    start = _pack(parameters)
    starting_cost = float(np.sum(residual(start) ** 2))

    if verbose:
        print("  refining the physical parameters against simulated "
              "trajectories")
        print(f"    starting simulation cost {starting_cost:.6f}")

    scale = np.where(np.abs(start) < 1e-12, 1.0, np.abs(start))
    try:
        solution = least_squares(
            residual,
            start,
            x_scale=scale,
            diff_step=1e-4,
            max_nfev=max_evaluations,
        )
    except (ValueError, FloatingPointError):
        if verbose:
            print("    refinement failed, keeping the equation error fit")
        return parameters

    refined_cost = float(np.sum(solution.fun ** 2))
    if verbose:
        print(f"    refined simulation cost  {refined_cost:.6f}")

    if not np.isfinite(refined_cost) or refined_cost >= starting_cost:
        if verbose:
            print("    no improvement, keeping the equation error fit")
        return parameters

    if verbose:
        print(f"    improvement factor {starting_cost / max(refined_cost, 1e-12):.2f}")
    return _unpack(solution.x, parameters)


# ----------------------------------------------------------------------
# Fast integrator
# ----------------------------------------------------------------------
# simulate() above evaluates spring_force and force_constant through numpy on
# one scalar at a time, inside a Python loop. Each of those calls allocates
# arrays and goes through np.polyval, which costs far more than the arithmetic
# itself. Refitting the model against simulated trajectories needs tens of
# thousands of these integrations, so the same recursion is written out here
# in plain floats with Horner's rule. The result is numerically identical and
# roughly an order of magnitude quicker.

def _compiled_parameters(parameters):
    spring = [float(value) for value in parameters["spring_coefficients"]]
    derivative = [float(value) for value in parameters["spring_derivative"]]
    force = [float(value) for value in parameters["bl_coefficients"]]
    return (
        tuple(spring),
        tuple(derivative),
        tuple(force),
        float(parameters["position_low_mm"]),
        float(parameters["position_high_mm"]),
        float(parameters["damping"]),
    )


def simulate_fast(record, parameters, time_step):
    """Displacement in millimetres, matching simulate() but without numpy overhead."""
    (spring, derivative, force_poly,
     low, high, damping) = _compiled_parameters(parameters)

    current = np.asarray(record["current"], dtype=np.float64)
    pad = record["pad"]
    total = len(current)
    mass = record["total_mass_g"] / 1000.0
    weight = mass * GRAVITY

    output = np.zeros(total)

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

    position = 0.0
    speed = 0.0
    half = 0.5 * time_step

    for index in range(pad, total):
        output[index] = position * MM_PER_M

        start_drive = current[index]
        end_drive = current[min(index + 1, total - 1)]
        mid_drive = 0.5 * (start_drive + end_drive)

        velocity_1 = speed
        acceleration_1 = acceleration(position, speed, start_drive)

        velocity_2 = speed + half * acceleration_1
        acceleration_2 = acceleration(position + half * velocity_1, velocity_2, mid_drive)

        velocity_3 = speed + half * acceleration_2
        acceleration_3 = acceleration(position + half * velocity_2, velocity_3, mid_drive)

        velocity_4 = speed + time_step * acceleration_3
        acceleration_4 = acceleration(
            position + time_step * velocity_3, velocity_4, end_drive
        )

        position += (
            time_step / 6.0
            * (velocity_1 + 2.0 * velocity_2 + 2.0 * velocity_3 + velocity_4)
        )
        speed += (
            time_step / 6.0
            * (acceleration_1 + 2.0 * acceleration_2
               + 2.0 * acceleration_3 + acceleration_4)
        )

    return output
