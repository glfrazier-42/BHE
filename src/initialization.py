"""
Initialization functions for black hole explosion simulation.

This module provides functions to set up initial conditions:
- Black hole ring positions and velocities
- Debris particle positions and velocities
- Initial state validation
"""

import numpy as np
from typing import Tuple

from . import constants as const
from .config import SimulationParameters, RingConfig
from .state import SimulationState


def initialize_ring0_circular_orbit(
    ring_config: RingConfig,
    start_index: int,
    state: SimulationState
) -> None:
    """
    Initialize Ring 0 black holes in circular orbit around origin.

    Places BHs evenly spaced around a circle in the xy-plane with
    velocities perpendicular to radial direction for circular orbit.

    Args:
        ring_config: Configuration for Ring 0
        start_index: Starting index in BH arrays for this ring
        state: SimulationState to populate (modified in place)

    Notes:
        - BHs placed at radius r in xy-plane (z=0)
        - Angular spacing: Δθ = 2π / N
        - Orbital velocity: v = v_orbital (perpendicular to radius)
        - For circular orbit: v = sqrt(G × M_central / r)
          But we use configured v_orbital instead
    """
    r = ring_config.radius
    n = ring_config.count
    v_orbital = ring_config.orbital_velocity

    for i in range(n):
        bh_idx = start_index + i

        # Angular position (evenly spaced)
        theta = 2.0 * np.pi * i / n

        # Position: (r cos θ, r sin θ, 0)
        state.bh_positions[bh_idx, 0] = r * np.cos(theta)
        state.bh_positions[bh_idx, 1] = r * np.sin(theta)
        state.bh_positions[bh_idx, 2] = 0.0

        # Velocity: perpendicular to radial direction
        # v_hat = (-sin θ, cos θ, 0) for counterclockwise orbit
        state.bh_velocities[bh_idx, 0] = -v_orbital * np.sin(theta)
        state.bh_velocities[bh_idx, 1] = v_orbital * np.cos(theta)
        state.bh_velocities[bh_idx, 2] = 0.0

        # Metadata
        state.bh_masses[bh_idx] = ring_config.mass_per_bh
        state.bh_ring_ids[bh_idx] = ring_config.ring_id
        state.bh_is_static[bh_idx] = ring_config.is_static
        state.bh_capture_radii[bh_idx] = ring_config.capture_radius


def initialize_static_ring_spiral(
    ring_config: RingConfig,
    start_index: int,
    state: SimulationState,
    n_arms: int = 2
) -> None:
    """
    Initialize static ring black holes in 2-arm spiral pattern.

    Places BHs in a logarithmic spiral pattern on a sphere of given radius.
    This creates a more uniform coverage than simple latitude bands.

    Args:
        ring_config: Configuration for this ring
        start_index: Starting index in BH arrays for this ring
        state: SimulationState to populate (modified in place)
        n_arms: Number of spiral arms (default: 2)

    Notes:
        - Uses Fibonacci sphere distribution for uniform coverage
        - BHs are static (zero velocity)
        - Positioned on sphere of radius r
    """
    r = ring_config.radius
    n = ring_config.count

    for i in range(n):
        bh_idx = start_index + i

        # Fibonacci sphere distribution for uniform coverage
        # https://arxiv.org/abs/0912.4540
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle ≈ 2.4 radians

        # Latitude: evenly spaced from -1 to 1
        y = 1.0 - (i / float(n - 1)) * 2.0  # y from 1 to -1
        radius_at_y = np.sqrt(1.0 - y * y)

        # Longitude: golden angle spacing
        theta = phi * i

        # Convert to Cartesian coordinates on unit sphere, then scale to r
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        state.bh_positions[bh_idx, 0] = r * x
        state.bh_positions[bh_idx, 1] = r * y
        state.bh_positions[bh_idx, 2] = r * z

        # Static: zero velocity
        state.bh_velocities[bh_idx, :] = 0.0

        # Metadata
        state.bh_masses[bh_idx] = ring_config.mass_per_bh
        state.bh_ring_ids[bh_idx] = ring_config.ring_id
        state.bh_is_static[bh_idx] = ring_config.is_static
        state.bh_capture_radii[bh_idx] = ring_config.capture_radius


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
        r_min: Minimum radius [m]
        r_max: Maximum radius [m]
        rng: NumPy random number generator

    Returns:
        positions: (n_debris, 3) array of positions [m]

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
        v_min: Minimum velocity magnitude [m/s]
        v_max: Maximum velocity magnitude [m/s]
        rng: NumPy random number generator

    Returns:
        velocities: (n_debris, 3) array of velocities [m/s]
    """
    velocities = np.zeros((n_debris, 3))

    # Fibonacci sphere for uniform angular distribution
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(n_debris):
        # Uniform direction (Fibonacci sphere)
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

    Sets up:
    1. Black hole rings (Ring 0 circular orbit, Rings 1-3 static spiral)
    2. Debris particles (positions and velocities)
    3. All metadata

    Args:
        params: Simulation parameters
        seed: Random seed for reproducibility

    Returns:
        state: Initialized SimulationState

    Notes:
        - Black holes are initialized first, ordered by ring ID
        - Debris positions: uniform over solid angle, r ∈ [r_min, r_max]
        - Debris velocities: uniform magnitude ∈ [v_min, v_max], random direction
        - All proper times start at 0
    """
    rng = np.random.default_rng(seed)

    # Create empty state
    state = SimulationState(
        n_bh=params.total_bh_count,
        n_debris=params.debris_count,
        M_central=params.M_central
    )

    # Initialize black hole rings
    bh_index = 0
    for ring in params.rings:
        if ring.ring_id == 0:
            # Ring 0: circular orbit
            initialize_ring0_circular_orbit(ring, bh_index, state)
        else:
            # Rings 1-3: static spiral
            initialize_static_ring_spiral(ring, bh_index, state)

        bh_index += ring.count

    # Initialize debris particles
    state.debris_positions[:] = sample_debris_positions_uniform_sphere(
        params.debris_count,
        params.debris_r_min,
        params.debris_r_max,
        rng
    )

    state.debris_velocities[:] = sample_debris_velocities_uniform(
        params.debris_count,
        params.debris_v_min,
        params.debris_v_max,
        rng
    )

    state.debris_masses[:] = params.debris_mass_per_particle
    state.debris_proper_times[:] = 0.0
    state.debris_accreted[:] = False
    state.debris_accreted_by[:] = -1

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

    # Total momentum (BHs + debris)
    bh_momentum = np.sum(state.bh_masses[:, np.newaxis] * state.bh_velocities, axis=0)
    debris_momentum = np.sum(
        state.debris_masses[:, np.newaxis] * state.debris_velocities, axis=0
    )
    total_momentum = bh_momentum + debris_momentum
    diagnostics['total_momentum'] = total_momentum
    diagnostics['total_momentum_magnitude'] = np.linalg.norm(total_momentum)

    # Total angular momentum (L = r × p)
    bh_angular_momentum = np.sum(
        np.cross(state.bh_positions, state.bh_masses[:, np.newaxis] * state.bh_velocities),
        axis=0
    )
    debris_angular_momentum = np.sum(
        np.cross(state.debris_positions, state.debris_masses[:, np.newaxis] * state.debris_velocities),
        axis=0
    )
    total_angular_momentum = bh_angular_momentum + debris_angular_momentum
    diagnostics['total_angular_momentum'] = total_angular_momentum
    diagnostics['total_angular_momentum_magnitude'] = np.linalg.norm(total_angular_momentum)

    # Velocity statistics
    debris_v_magnitudes = np.linalg.norm(state.debris_velocities, axis=1)
    diagnostics['debris_v_min'] = np.min(debris_v_magnitudes)
    diagnostics['debris_v_max'] = np.max(debris_v_magnitudes)
    diagnostics['debris_v_mean'] = np.mean(debris_v_magnitudes)
    diagnostics['debris_v_std'] = np.std(debris_v_magnitudes)

    # Position statistics
    debris_r_magnitudes = np.linalg.norm(state.debris_positions, axis=1)
    diagnostics['debris_r_min'] = np.min(debris_r_magnitudes)
    diagnostics['debris_r_max'] = np.max(debris_r_magnitudes)
    diagnostics['debris_r_mean'] = np.mean(debris_r_magnitudes)

    # Black hole statistics
    bh_v_magnitudes = np.linalg.norm(state.bh_velocities, axis=1)
    diagnostics['bh_v_max'] = np.max(bh_v_magnitudes)

    # Ring breakdown
    for ring_id in np.unique(state.bh_ring_ids):
        ring_mask = state.bh_ring_ids == ring_id
        count = np.sum(ring_mask)
        diagnostics[f'ring_{ring_id}_count'] = count

        if count > 0:
            ring_r = np.linalg.norm(state.bh_positions[ring_mask], axis=1)
            diagnostics[f'ring_{ring_id}_radius_mean'] = np.mean(ring_r)

    return diagnostics
