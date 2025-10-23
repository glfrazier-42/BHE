"""
Runtime diagnostics for simulation health checks.

This module provides functions to detect:
- Timestep instability (orbit decay/growth)
- Energy conservation violations
- Particle velocities exceeding c
- Numerical blow-up
"""

import numpy as np
from bhe import constants as const


def check_timestep_stability(state, params, history_length=100):
    """
    Check if timestep is appropriate by monitoring energy drift.

    For a conservative system, total energy should be constant.
    If energy is growing/shrinking significantly, the timestep is too large.

    Args:
        state: SimulationState
        params: SimulationParameters
        history_length: Number of recent timesteps to check

    Returns:
        dict with:
            - is_stable: bool
            - energy_drift_percent: float (% change per Gyr)
            - warnings: list of warning messages
    """
    warnings = []

    # Calculate current total energy
    E_total = calculate_total_energy(state, params)

    # TODO: Track energy history and check for drift
    # For now, just check for obvious problems

    # Check for NaN or Inf
    if not np.isfinite(E_total):
        warnings.append("CRITICAL: Total energy is NaN or Inf - numerical instability!")
        return {
            'is_stable': False,
            'energy_drift_percent': np.inf,
            'warnings': warnings
        }

    # Check particle velocities don't exceed c
    max_debris_v = 0.0
    if state.n_debris > 0:
        debris_v_mags = np.sqrt(np.sum(state.debris_velocities**2, axis=1))
        max_debris_v = np.max(debris_v_mags[~state.debris_accreted])

    max_bh_v = 0.0
    if state.n_bh > 0:
        bh_v_mags = np.sqrt(np.sum(state.bh_velocities**2, axis=1))
        max_bh_v = np.max(bh_v_mags)

    max_v = max(max_debris_v, max_bh_v)

    if max_v > const.c:
        warnings.append(f"WARNING: Particle velocity ({max_v/const.c:.3f}c) exceeds speed of light!")
        warnings.append("         Timestep is too large or integration is unstable.")
        return {
            'is_stable': False,
            'energy_drift_percent': np.nan,
            'warnings': warnings
        }

    if max_v > 0.99 * const.c:
        warnings.append(f"CAUTION: Particle velocity ({max_v/const.c:.3f}c) approaching speed of light.")

    return {
        'is_stable': True,
        'energy_drift_percent': 0.0,  # TODO: Calculate actual drift
        'warnings': warnings
    }


def calculate_total_energy(state, params):
    """
    Calculate total energy (kinetic + potential) of the system.

    E_total = Σ KE_i + Σ PE_ij

    Args:
        state: SimulationState
        params: SimulationParameters

    Returns:
        float: Total energy in Joules
    """
    from .physics import calculate_kinetic_energy, calculate_potential_energy

    E_kinetic = 0.0
    E_potential = 0.0

    # Debris kinetic energy
    for i in range(state.n_debris):
        if state.debris_accreted[i]:
            continue
        E_kinetic += calculate_kinetic_energy(
            state.debris_masses[i],
            state.debris_velocities[i]
        )

    # BH kinetic energy
    for i in range(state.n_bh):
        if state.bh_is_static[i]:
            continue
        E_kinetic += calculate_kinetic_energy(
            state.bh_masses[i],
            state.bh_velocities[i]
        )

    # Debris-debris potential energy
    for i in range(state.n_debris):
        if state.debris_accreted[i]:
            continue
        for j in range(i + 1, state.n_debris):
            if state.debris_accreted[j]:
                continue
            E_potential += calculate_potential_energy(
                state.debris_positions[i],
                state.debris_positions[j],
                state.debris_masses[i],
                state.debris_masses[j]
            )

    # Debris-BH potential energy
    for i in range(state.n_debris):
        if state.debris_accreted[i]:
            continue
        for j in range(state.n_bh):
            E_potential += calculate_potential_energy(
                state.debris_positions[i],
                state.bh_positions[j],
                state.debris_masses[i],
                state.bh_masses[j]
            )

    # BH-BH potential energy
    for i in range(state.n_bh):
        for j in range(i + 1, state.n_bh):
            E_potential += calculate_potential_energy(
                state.bh_positions[i],
                state.bh_positions[j],
                state.bh_masses[i],
                state.bh_masses[j]
            )

    return E_kinetic + E_potential


def estimate_courant_timestep(state, safety_factor=0.1):
    """
    Estimate maximum stable timestep using Courant condition.

    For explicit integrators, dt should satisfy:
    dt < safety_factor * (min distance / max velocity)

    Args:
        state: SimulationState
        safety_factor: Safety margin (typically 0.1 to 0.5)

    Returns:
        float: Recommended maximum timestep in seconds
    """
    min_distance = np.inf
    max_velocity = 0.0

    # Find minimum distance between any two active particles
    # Check debris-debris
    for i in range(state.n_debris):
        if state.debris_accreted[i]:
            continue
        for j in range(i + 1, state.n_debris):
            if state.debris_accreted[j]:
                continue
            r_vec = state.debris_positions[j] - state.debris_positions[i]
            r = np.sqrt(np.sum(r_vec**2))
            min_distance = min(min_distance, r)

    # Check debris-BH
    for i in range(state.n_debris):
        if state.debris_accreted[i]:
            continue
        for j in range(state.n_bh):
            r_vec = state.bh_positions[j] - state.debris_positions[i]
            r = np.sqrt(np.sum(r_vec**2))
            min_distance = min(min_distance, r)

    # Check BH-BH
    for i in range(state.n_bh):
        for j in range(i + 1, state.n_bh):
            r_vec = state.bh_positions[j] - state.bh_positions[i]
            r = np.sqrt(np.sum(r_vec**2))
            min_distance = min(min_distance, r)

    # Find maximum velocity
    if state.n_debris > 0:
        debris_v_mags = np.sqrt(np.sum(state.debris_velocities**2, axis=1))
        max_velocity = max(max_velocity, np.max(debris_v_mags[~state.debris_accreted]))

    if state.n_bh > 0:
        bh_v_mags = np.sqrt(np.sum(state.bh_velocities**2, axis=1))
        max_velocity = max(max_velocity, np.max(bh_v_mags))

    if max_velocity == 0.0 or min_distance == np.inf:
        return np.inf

    # Courant timestep
    dt_courant = safety_factor * min_distance / max_velocity

    return dt_courant


def check_timestep_against_courant(state, params):
    """
    Check if current timestep satisfies Courant condition.

    Returns:
        dict with:
            - satisfies_courant: bool
            - current_dt: float (seconds)
            - recommended_dt: float (seconds)
            - ratio: float (current / recommended)
            - warning: str or None
    """
    dt_courant = estimate_courant_timestep(state, safety_factor=0.1)
    dt_current = params.dt

    if dt_courant == np.inf:
        return {
            'satisfies_courant': True,
            'current_dt': dt_current,
            'recommended_dt': dt_courant,
            'ratio': 0.0,
            'warning': None
        }

    ratio = dt_current / dt_courant

    warning = None
    satisfies = ratio <= 1.0

    if ratio > 1.0:
        warning = (f"Timestep ({dt_current / 1.0e9:.6f} Gyr) is {ratio:.1f}x "
                   f"larger than Courant limit ({dt_courant / 1.0e9:.6f} Gyr). "
                   "Integration may be unstable!")
    elif ratio > 0.5:
        warning = (f"Timestep is {ratio*100:.0f}% of Courant limit. "
                   "Consider reducing timestep if orbits are unstable.")

    return {
        'satisfies_courant': satisfies,
        'current_dt': dt_current,
        'recommended_dt': dt_courant,
        'ratio': ratio,
        'warning': warning
    }
