"""
Initialization functions for black hole explosion simulation.

This module provides functions to set up initial conditions for the UNIFIED
PARTICLE SYSTEM where all particles (black holes and debris) are stored in
the same arrays and tracked via metadata.
"""

import numpy as np
from typing import Tuple

from bhe import constants as const
from bhe.config import SimulationParameters, RingConfig
from bhe.state import SimulationState, BLACK_HOLE, DEBRIS


def initialize_circular_orbit_ring(
    ring_config: RingConfig,
    start_index: int,
    state: SimulationState
) -> None:
    """
    Initialize ring black holes in circular orbit around origin.

    Places BHs evenly spaced around a circle in the xy-plane with
    velocities perpendicular to radial direction for circular orbit.

    Args:
        ring_config: Configuration for this ring
        start_index: Starting index in unified particle arrays for this ring
        state: SimulationState to populate (modified in place)

    Notes:
        - BHs placed at radius r in xy-plane (z=0)
        - Angular spacing: Δθ = 2π / N
        - Orbital velocity: v = v_orbital (perpendicular to radius)
        - For circular orbit: v = sqrt(G × M_central / r)
        - Actual velocity comes from ring_config.orbital_velocity
    """
    r = ring_config.radius
    n = ring_config.count
    v_orbital = ring_config.orbital_velocity

    for i in range(n):
        idx = start_index + i

        # Angular position (evenly spaced)
        theta = 2.0 * np.pi * i / n

        # Position: (r cos θ, r sin θ, 0)
        state.positions[idx, 0] = r * np.cos(theta)
        state.positions[idx, 1] = r * np.sin(theta)
        state.positions[idx, 2] = 0.0

        # Velocity: perpendicular to radial direction
        # v_hat = (-sin θ, cos θ, 0) for counterclockwise orbit
        state.velocities[idx, 0] = -v_orbital * np.sin(theta)
        state.velocities[idx, 1] = v_orbital * np.cos(theta)
        state.velocities[idx, 2] = 0.0

        # Core physics properties
        state.masses[idx] = ring_config.mass_per_bh
        state.accreted[idx] = False
        state.proper_times[idx] = 0.0

        # Metadata for tracking
        state.particle_type[idx] = BLACK_HOLE
        state.ring_id[idx] = ring_config.ring_id
        state.capture_radius[idx] = ring_config.capture_radius

        # Store initial conditions for analysis
        state.initial_speed[idx] = v_orbital
        state.initial_position[idx] = state.positions[idx].copy()


def sample_debris_positions_uniform_sphere(
    n_debris: int,
    r_min: float,
    r_max: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample debris positions uniformly over solid angle and radial range.

    Uses inverse transform sampling for uniform distribution:
    - Solid angle: uniform on unit sphere (Fibonacci sphere)
    - Radial: uniform in r (not r³, to get uniform over volume)

    Args:
        n_debris: Number of debris particles
        r_min: Minimum radius [ly]
        r_max: Maximum radius [ly]
        rng: NumPy random number generator

    Returns:
        positions: (n_debris, 3) array of positions [ly]

    Notes:
        - For uniform distribution over solid angle, use Fibonacci sphere
        - For radial distribution, we use uniform in r (linear)
        - Alternative: uniform in r³ for volume-uniform distribution
    """
    positions = np.zeros((n_debris, 3))

    # Fibonacci sphere for uniform angular distribution
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(n_debris):
        # Uniform solid angle (Fibonacci sphere)
        if n_debris == 1:
            # Special case: single particle at origin
            y = 0.0
        else:
            y = 1.0 - (i / float(n_debris - 1)) * 2.0
        radius_at_y = np.sqrt(1.0 - y * y)
        theta = phi * i

        # Unit direction vector
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        direction = np.array([x, y, z])

        # Uniform radial distribution in [r_min, r_max]
        r = rng.uniform(r_min, r_max)

        positions[i] = r * direction

    return positions


def sample_debris_velocities_uniform(
    n_debris: int,
    v_min: float,
    v_max: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample debris velocities with uniform magnitude and random direction.

    Velocity magnitudes uniformly sampled from [v_min, v_max].
    Directions uniformly sampled over unit sphere.

    Args:
        n_debris: Number of debris particles
        v_min: Minimum velocity magnitude [fraction of c]
        v_max: Maximum velocity magnitude [fraction of c]
        rng: NumPy random number generator

    Returns:
        velocities: (n_debris, 3) array of velocities [fraction of c]
    """
    velocities = np.zeros((n_debris, 3))

    # Fibonacci sphere for uniform angular distribution
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(n_debris):
        # Uniform direction (Fibonacci sphere)
        if n_debris == 1:
            # Special case: single particle with zero velocity
            y = 0.0
        else:
            y = 1.0 - (i / float(n_debris - 1)) * 2.0
        radius_at_y = np.sqrt(1.0 - y * y)
        theta = phi * i

        # Unit direction vector
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        direction = np.array([x, y, z])

        # Uniform velocity magnitude
        v = rng.uniform(v_min, v_max)

        velocities[i] = v * direction

    return velocities


def initialize_simulation(
    params: SimulationParameters,
    seed: int = 42
) -> SimulationState:
    """
    Initialize complete simulation state from parameters.

    Sets up unified particle system:
    1. Black hole rings (indices 0 to n_bh-1)
    2. Debris particles (indices n_bh to n_total-1)
    3. All metadata for tracking

    Args:
        params: Simulation parameters
        seed: Random seed for reproducibility

    Returns:
        state: Initialized SimulationState

    Notes:
        - Particle order: BHs first (by ring ID), then debris
        - All particles stored in unified arrays (positions, velocities, masses)
        - Metadata arrays track particle identity (particle_type, ring_id, etc.)
        - All proper times start at 0
        - Initial conditions stored for analysis (initial_speed, initial_position)
    """
    rng = np.random.default_rng(seed)

    # Calculate total particle count
    n_bh = params.total_bh_count
    n_debris = params.debris_count
    n_total = n_bh + n_debris

    # Create empty state with unified arrays
    state = SimulationState(n_total=n_total, M_central=params.M_central)

    # Initialize black hole rings (indices 0 to n_bh-1)
    # All rings are placed in circular orbits in the xy-plane (galactic disk structure)
    # Velocity is set according to ring_config.orbital_velocity (0 for static, Keplerian for orbiting)
    particle_index = 0
    for ring in params.rings:
        if ring.count > 0:
            initialize_circular_orbit_ring(ring, particle_index, state)
            particle_index += ring.count

    # Initialize debris particles (indices n_bh to n_total-1)
    debris_start = n_bh
    debris_end = n_total

    # Sample positions and velocities
    debris_positions = sample_debris_positions_uniform_sphere(
        n_debris,
        params.debris_r_min,
        params.debris_r_max,
        rng
    )

    debris_velocities = sample_debris_velocities_uniform(
        n_debris,
        params.debris_v_min,
        params.debris_v_max,
        rng
    )

    # Populate unified arrays for debris
    for i in range(n_debris):
        idx = debris_start + i

        # Core physics properties
        state.positions[idx] = debris_positions[i]
        state.velocities[idx] = debris_velocities[i]
        state.masses[idx] = params.debris_mass_per_particle
        state.accreted[idx] = False
        state.proper_times[idx] = 0.0

        # Metadata for tracking
        state.particle_type[idx] = DEBRIS
        state.ring_id[idx] = -1  # Debris not in any ring
        state.capture_radius[idx] = 0.0  # Debris don't capture

        # Store initial conditions for analysis
        v_magnitude = np.linalg.norm(debris_velocities[i])
        state.initial_speed[idx] = v_magnitude
        state.initial_position[idx] = debris_positions[i].copy()

    # Simulation starts at t=0
    state.time = 0.0
    state.timestep_count = 0

    return state


def validate_initial_conditions(state: SimulationState) -> dict:
    """
    Validate initial conditions and compute diagnostics.

    Checks:
    - Total momentum (should be ~0 for symmetric initial conditions)
    - Total angular momentum
    - Total energy (kinetic + potential)
    - Velocity distribution statistics

    Args:
        state: SimulationState to validate

    Returns:
        diagnostics: Dictionary with validation metrics
    """
    diagnostics = {}

    # Get masks for filtering
    active_mask = state.get_active_mask()
    bh_mask = state.get_black_hole_mask()
    debris_mask = state.get_debris_mask()

    # Total momentum (all active particles)
    active_momentum = np.sum(
        state.masses[active_mask, np.newaxis] * state.velocities[active_mask],
        axis=0
    )
    diagnostics['total_momentum'] = active_momentum
    diagnostics['total_momentum_magnitude'] = np.linalg.norm(active_momentum)

    # Total angular momentum (L = r × p)
    active_angular_momentum = np.sum(
        np.cross(
            state.positions[active_mask],
            state.masses[active_mask, np.newaxis] * state.velocities[active_mask]
        ),
        axis=0
    )
    diagnostics['total_angular_momentum'] = active_angular_momentum
    diagnostics['total_angular_momentum_magnitude'] = np.linalg.norm(active_angular_momentum)

    # Velocity statistics for debris
    if np.any(debris_mask):
        debris_v_magnitudes = np.linalg.norm(state.velocities[debris_mask], axis=1)
        diagnostics['debris_v_min'] = np.min(debris_v_magnitudes)
        diagnostics['debris_v_max'] = np.max(debris_v_magnitudes)
        diagnostics['debris_v_mean'] = np.mean(debris_v_magnitudes)
        diagnostics['debris_v_std'] = np.std(debris_v_magnitudes)

        # Position statistics for debris
        debris_r_magnitudes = np.linalg.norm(state.positions[debris_mask], axis=1)
        diagnostics['debris_r_min'] = np.min(debris_r_magnitudes)
        diagnostics['debris_r_max'] = np.max(debris_r_magnitudes)
        diagnostics['debris_r_mean'] = np.mean(debris_r_magnitudes)

    # Black hole statistics
    if np.any(bh_mask):
        bh_v_magnitudes = np.linalg.norm(state.velocities[bh_mask], axis=1)
        diagnostics['bh_v_max'] = np.max(bh_v_magnitudes)

        # Ring breakdown
        for ring_id in np.unique(state.ring_id[bh_mask]):
            ring_mask = state.get_ring_mask(ring_id)
            count = np.sum(ring_mask)
            diagnostics[f'ring_{ring_id}_count'] = count

            if count > 0:
                ring_r = np.linalg.norm(state.positions[ring_mask], axis=1)
                diagnostics[f'ring_{ring_id}_radius_mean'] = np.mean(ring_r)

    return diagnostics
