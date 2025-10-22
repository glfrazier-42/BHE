"""
Time evolution engine for black hole explosion simulation.

This module implements the core N-body simulation loop, including:
- Debris particle updates (positions, velocities, proper times)
- Dynamic black hole orbital evolution
- Accretion detection and momentum conservation
- Main simulation driver with progress tracking

All computational hot loops are Numba-compiled for performance.

INTEGRATION SCHEME:
Uses leapfrog (kick-drift-kick) integration, which is symplectic and
conserves energy for orbital systems. This is the standard method in
astrophysical N-body simulations (GADGET, GIZMO, REBOUND, etc.).

The algorithm:
  1. Half-step velocity kick: v_half = v + 0.5 * a * dt
  2. Full-step position drift: x_new = x + v_half * dt
  3. Recalculate acceleration at new position
  4. Final half-step velocity kick: v_new = v_half + 0.5 * a_new * dt
"""

import numpy as np
from numba import jit, prange
from tqdm import tqdm

from . import constants as const
from .physics import (
    calculate_acceleration_from_bhs,
    lorentz_factor,
    lorentz_factor_scalar
)
from .state import SimulationState
from .config import SimulationParameters


@jit(nopython=True)
def calculate_debris_acceleration(
    particle_idx: int,
    debris_pos: np.ndarray,
    debris_vel: np.ndarray,
    debris_masses: np.ndarray,
    debris_accreted: np.ndarray,
    bh_positions: np.ndarray,
    bh_masses: np.ndarray,
    bh_velocities: np.ndarray,
    bh_is_static: np.ndarray,
    use_relativistic: bool
) -> np.ndarray:
    """
    Calculate acceleration on a single debris particle from all forces.

    Args:
        particle_idx: Index of the particle to calculate acceleration for
        debris_pos: Debris positions (N_debris, 3) [meters]
        debris_vel: Debris velocities (N_debris, 3) [m/s]
        debris_masses: Debris masses (N_debris,) [kg]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        bh_positions: BH positions (N_bh, 3) [meters]
        bh_masses: BH masses (N_bh,) [kg]
        bh_velocities: BH velocities (N_bh, 3) [m/s]
        bh_is_static: BH static flags (N_bh,) [bool]
        use_relativistic: Whether to use relativistic mass

    Returns:
        Acceleration vector (3,) [m/s²]
    """
    i = particle_idx
    N_debris = len(debris_pos)

    # Calculate acceleration from all Ring black holes
    accel = calculate_acceleration_from_bhs(
        debris_pos[i],
        bh_positions,
        bh_masses,
        bh_velocities,
        bh_is_static,
        use_relativistic
    )

    # Add acceleration from all other debris particles (debris-debris gravity)
    for j in range(N_debris):
        if j == i or debris_accreted[j]:
            continue

        # Vector from particle i to particle j
        r_vec = debris_pos[j] - debris_pos[i]
        r_squared = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2

        # Minimum distance threshold to avoid singularities
        if r_squared < 1e20:  # (1e10 meters)^2
            continue

        r = np.sqrt(r_squared)

        # Effective mass (relativistic if requested)
        if use_relativistic:
            gamma = lorentz_factor(debris_vel[j])
            m_eff = gamma * debris_masses[j]
        else:
            m_eff = debris_masses[j]

        # Gravitational acceleration: a = G × M / r²
        accel_magnitude = const.G * m_eff / r_squared
        accel += accel_magnitude * (r_vec / r)

    return accel


@jit(nopython=True)
def update_all_particles(
    debris_pos: np.ndarray,
    debris_vel: np.ndarray,
    debris_masses: np.ndarray,
    debris_proper_times: np.ndarray,
    debris_accreted: np.ndarray,
    bh_positions: np.ndarray,
    bh_velocities: np.ndarray,
    bh_masses: np.ndarray,
    bh_ring_ids: np.ndarray,
    bh_is_static: np.ndarray,
    dt: float,
    use_relativistic: bool
) -> None:
    """
    Update ALL particles (debris + BHs) simultaneously using leapfrog integration.

    This ensures that both debris and BHs see the same state at each phase of
    the integration, which is critical for time-reversibility and energy conservation.

    Uses kick-drift-kick leapfrog integration:
    1. Calculate all accelerations at t=0
    2. Half-step velocity kick for all particles
    3. Full-step position drift for all particles
    4. Recalculate all accelerations at t=dt
    5. Final half-step velocity kick for all particles

    Args:
        debris_pos: Debris positions (N_debris, 3) [meters]
        debris_vel: Debris velocities (N_debris, 3) [m/s]
        debris_masses: Debris masses (N_debris,) [kg]
        debris_proper_times: Debris proper times (N_debris,) [seconds]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        bh_positions: BH positions (N_bh, 3) [meters]
        bh_velocities: BH velocities (N_bh, 3) [m/s]
        bh_masses: BH masses (N_bh,) [kg]
        bh_ring_ids: BH ring IDs (N_bh,) [int]
        bh_is_static: BH static flags (N_bh,) [bool]
        dt: Timestep [seconds]
        use_relativistic: Whether to use relativistic mass

    Returns:
        None (modifies arrays in place)
    """
    N_debris = len(debris_pos)
    N_bh = len(bh_positions)

    # ===== PHASE 1: Save initial state =====
    old_debris_pos = debris_pos.copy()
    old_debris_vel = debris_vel.copy()
    old_bh_pos = bh_positions.copy()
    old_bh_vel = bh_velocities.copy()

    # ===== PHASE 2: Calculate all accelerations at t=0 =====
    debris_accels = np.zeros_like(debris_pos)
    bh_accels = np.zeros_like(bh_positions)

    # Calculate debris accelerations
    for i in range(N_debris):
        if debris_accreted[i]:
            continue
        debris_accels[i] = calculate_debris_acceleration(
            i, old_debris_pos, old_debris_vel, debris_masses, debris_accreted,
            old_bh_pos, bh_masses, old_bh_vel, bh_is_static,
            use_relativistic
        )

    # Calculate BH accelerations
    for i in range(N_bh):
        if bh_is_static[i]:
            continue
        bh_accels[i] = calculate_bh_acceleration(
            i, old_bh_pos, old_bh_vel, bh_masses, bh_is_static,
            old_debris_pos, old_debris_vel, debris_masses, debris_accreted,
            use_relativistic
        )

    # ===== PHASE 3: Half-step velocity kick and position drift =====
    new_debris_pos = np.zeros_like(debris_pos)
    new_debris_vel = np.zeros_like(debris_vel)
    new_bh_pos = np.zeros_like(bh_positions)
    new_bh_vel = np.zeros_like(bh_velocities)

    # Update debris
    for i in range(N_debris):
        if debris_accreted[i]:
            new_debris_pos[i] = old_debris_pos[i]
            new_debris_vel[i] = old_debris_vel[i]
            continue

        # Half-step velocity kick
        vel_half = old_debris_vel[i] + 0.5 * debris_accels[i] * dt

        # Full-step position drift
        new_debris_pos[i] = old_debris_pos[i] + vel_half * dt
        new_debris_vel[i] = vel_half  # Store for phase 4

    # Update BHs
    for i in range(N_bh):
        if bh_is_static[i]:
            new_bh_pos[i] = old_bh_pos[i]
            new_bh_vel[i] = old_bh_vel[i]
            continue

        # Half-step velocity kick
        vel_half = old_bh_vel[i] + 0.5 * bh_accels[i] * dt

        # Full-step position drift
        new_bh_pos[i] = old_bh_pos[i] + vel_half * dt
        new_bh_vel[i] = vel_half  # Store for phase 4

    # ===== PHASE 4: Recalculate accelerations at new positions =====
    debris_accels_new = np.zeros_like(debris_pos)
    bh_accels_new = np.zeros_like(bh_positions)

    # Calculate debris accelerations at new positions
    for i in range(N_debris):
        if debris_accreted[i]:
            continue
        debris_accels_new[i] = calculate_debris_acceleration(
            i, new_debris_pos, new_debris_vel, debris_masses, debris_accreted,
            new_bh_pos, bh_masses, new_bh_vel, bh_is_static,
            use_relativistic
        )

    # Calculate BH accelerations at new positions
    for i in range(N_bh):
        if bh_is_static[i]:
            continue
        bh_accels_new[i] = calculate_bh_acceleration(
            i, new_bh_pos, new_bh_vel, bh_masses, bh_is_static,
            new_debris_pos, new_debris_vel, debris_masses, debris_accreted,
            use_relativistic
        )

    # ===== PHASE 5: Final half-step velocity kick =====
    # Update debris velocities
    for i in range(N_debris):
        if debris_accreted[i]:
            continue
        new_debris_vel[i] = new_debris_vel[i] + 0.5 * debris_accels_new[i] * dt

    # Update BH velocities
    for i in range(N_bh):
        if bh_is_static[i]:
            continue
        new_bh_vel[i] = new_bh_vel[i] + 0.5 * bh_accels_new[i] * dt

    # ===== PHASE 6: Update proper times for debris =====
    for i in range(N_debris):
        if debris_accreted[i]:
            continue
        gamma = lorentz_factor(new_debris_vel[i])
        debris_proper_times[i] += dt / gamma

    # ===== PHASE 7: Write back to original arrays =====
    debris_pos[:] = new_debris_pos
    debris_vel[:] = new_debris_vel
    bh_positions[:] = new_bh_pos
    bh_velocities[:] = new_bh_vel


@jit(nopython=True)
def update_debris_particles(
    debris_pos: np.ndarray,
    debris_vel: np.ndarray,
    debris_masses: np.ndarray,
    debris_proper_times: np.ndarray,
    debris_accreted: np.ndarray,
    bh_positions: np.ndarray,
    bh_masses: np.ndarray,
    bh_velocities: np.ndarray,
    bh_is_static: np.ndarray,
    dt: float,
    use_relativistic: bool
) -> None:
    """
    Update all debris particles for one timestep using leapfrog integration.

    Each debris particle feels gravitational force from:
    - All Ring black holes
    - All other debris particles (direct N-body)

    This is O(N_debris * (N_bh + N_debris)) operations.

    Uses kick-drift-kick leapfrog integration for symplectic,
    energy-conserving time evolution.

    Args:
        debris_pos: Debris positions (N_debris, 3) [meters]
        debris_vel: Debris velocities (N_debris, 3) [m/s]
        debris_masses: Debris masses (N_debris,) [kg]
        debris_proper_times: Debris proper times (N_debris,) [seconds]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        bh_positions: BH positions (N_bh, 3) [meters]
        bh_masses: BH masses (N_bh,) [kg]
        bh_velocities: BH velocities (N_bh, 3) [m/s]
        bh_is_static: BH static flags (N_bh,) [bool]
        dt: Timestep [seconds]
        use_relativistic: Whether to use relativistic mass

    Returns:
        None (modifies arrays in place)
    """
    N_debris = len(debris_pos)

    # Store old positions and velocities
    old_pos = debris_pos.copy()
    old_vel = debris_vel.copy()

    # Store accelerations at old positions
    old_accels = np.zeros_like(debris_pos)

    # Calculate accelerations at current positions
    for i in range(N_debris):
        if debris_accreted[i]:
            continue
        old_accels[i] = calculate_debris_acceleration(
            i, old_pos, old_vel, debris_masses, debris_accreted,
            bh_positions, bh_masses, bh_velocities, bh_is_static,
            use_relativistic
        )

    # Leapfrog step 1 & 2: Half-step velocity kick and position drift
    for i in range(N_debris):
        if debris_accreted[i]:
            continue

        # Half-step velocity kick
        debris_vel[i] = old_vel[i] + 0.5 * old_accels[i] * dt

        # Full-step position drift
        debris_pos[i] = old_pos[i] + debris_vel[i] * dt

    # Leapfrog step 3: Final half-step velocity kick at new positions
    for i in range(N_debris):
        if debris_accreted[i]:
            continue

        # Calculate acceleration at new position
        accel_new = calculate_debris_acceleration(
            i, debris_pos, debris_vel, debris_masses, debris_accreted,
            bh_positions, bh_masses, bh_velocities, bh_is_static,
            use_relativistic
        )

        # Final half-step velocity kick
        debris_vel[i] = debris_vel[i] + 0.5 * accel_new * dt

        # Update proper time (time dilation from special relativity)
        # dt_proper = dt / gamma where gamma = 1/sqrt(1 - v^2/c^2)
        gamma = lorentz_factor(debris_vel[i])
        debris_proper_times[i] += dt / gamma


@jit(nopython=True)
def calculate_bh_acceleration(
    bh_idx: int,
    bh_positions: np.ndarray,
    bh_velocities: np.ndarray,
    bh_masses: np.ndarray,
    bh_is_static: np.ndarray,
    debris_pos: np.ndarray,
    debris_vel: np.ndarray,
    debris_masses: np.ndarray,
    debris_accreted: np.ndarray,
    use_relativistic: bool
) -> np.ndarray:
    """
    Calculate acceleration on a single black hole from all forces.

    Args:
        bh_idx: Index of the BH to calculate acceleration for
        bh_positions: BH positions (N_bh, 3) [meters]
        bh_velocities: BH velocities (N_bh, 3) [m/s]
        bh_masses: BH masses (N_bh,) [kg]
        bh_is_static: BH static flags (N_bh,) [bool]
        debris_pos: Debris positions (N_debris, 3) [meters]
        debris_vel: Debris velocities (N_debris, 3) [m/s]
        debris_masses: Debris masses (N_debris,) [kg]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        use_relativistic: Whether to use relativistic mass

    Returns:
        Acceleration vector (3,) [m/s²]
    """
    i = bh_idx
    N_bh = len(bh_positions)
    N_debris = len(debris_pos)

    accel = np.zeros(3, dtype=np.float64)

    # Acceleration from other BHs (both static and dynamic)
    for j in range(N_bh):
        if j == i:
            continue

        # Vector from BH i to BH j
        r_vec_ij = bh_positions[j] - bh_positions[i]
        r_squared_ij = r_vec_ij[0]**2 + r_vec_ij[1]**2 + r_vec_ij[2]**2

        if r_squared_ij < 1e20:  # Avoid close encounters
            continue

        r_ij = np.sqrt(r_squared_ij)

        # Effective mass (relativistic if moving)
        if use_relativistic and not bh_is_static[j]:
            gamma = lorentz_factor(bh_velocities[j])
            m_eff = gamma * bh_masses[j]
        else:
            m_eff = bh_masses[j]

        # Gravitational acceleration: a = G * M / r^2
        accel_magnitude = const.G * m_eff / r_squared_ij
        accel += accel_magnitude * (r_vec_ij / r_ij)

    # Acceleration from debris particles
    for j in range(N_debris):
        if debris_accreted[j]:
            continue

        # Vector from BH i to debris j
        r_vec = debris_pos[j] - bh_positions[i]
        r_squared = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2

        if r_squared < 1e20:  # Avoid close encounters
            continue

        r = np.sqrt(r_squared)

        # Effective mass (relativistic if requested)
        if use_relativistic:
            gamma = lorentz_factor(debris_vel[j])
            m_eff = gamma * debris_masses[j]
        else:
            m_eff = debris_masses[j]

        # Gravitational acceleration: a = G * M / r^2
        accel_magnitude = const.G * m_eff / r_squared
        accel += accel_magnitude * (r_vec / r)

    return accel


@jit(nopython=True)
def update_dynamic_bhs(
    bh_positions: np.ndarray,
    bh_velocities: np.ndarray,
    bh_masses: np.ndarray,
    bh_ring_ids: np.ndarray,
    bh_is_static: np.ndarray,
    debris_pos: np.ndarray,
    debris_vel: np.ndarray,
    debris_masses: np.ndarray,
    debris_accreted: np.ndarray,
    dt: float,
    use_relativistic: bool
) -> None:
    """
    Update all dynamic (non-static) black holes for one timestep using leapfrog integration.

    Dynamic BHs feel gravitational force from:
    - All other black holes (both static and dynamic)
    - All active debris particles

    Static BHs are not updated (they remain fixed in space).

    Uses kick-drift-kick leapfrog integration for symplectic,
    energy-conserving time evolution.

    Args:
        bh_positions: BH positions (N_bh, 3) [meters]
        bh_velocities: BH velocities (N_bh, 3) [m/s]
        bh_masses: BH masses (N_bh,) [kg]
        bh_ring_ids: BH ring IDs (N_bh,) [int]
        bh_is_static: BH static flags (N_bh,) [bool]
        debris_pos: Debris positions (N_debris, 3) [meters]
        debris_vel: Debris velocities (N_debris, 3) [m/s]
        debris_masses: Debris masses (N_debris,) [kg]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        dt: Timestep [seconds]
        use_relativistic: Whether to use relativistic mass

    Returns:
        None (modifies arrays in place)
    """
    N_bh = len(bh_positions)

    # Store old positions and velocities
    old_pos = bh_positions.copy()
    old_vel = bh_velocities.copy()

    # Store accelerations at old positions
    old_accels = np.zeros_like(bh_positions)

    # Calculate accelerations at current positions
    for i in range(N_bh):
        if bh_is_static[i]:
            continue
        old_accels[i] = calculate_bh_acceleration(
            i, old_pos, old_vel, bh_masses, bh_is_static,
            debris_pos, debris_vel, debris_masses, debris_accreted,
            use_relativistic
        )

    # Leapfrog step 1 & 2: Half-step velocity kick and position drift
    for i in range(N_bh):
        if bh_is_static[i]:
            continue

        # Half-step velocity kick
        bh_velocities[i] = old_vel[i] + 0.5 * old_accels[i] * dt

        # Full-step position drift
        bh_positions[i] = old_pos[i] + bh_velocities[i] * dt

    # Leapfrog step 3: Final half-step velocity kick at new positions
    for i in range(N_bh):
        if bh_is_static[i]:
            continue

        # Calculate acceleration at new position
        accel_new = calculate_bh_acceleration(
            i, bh_positions, bh_velocities, bh_masses, bh_is_static,
            debris_pos, debris_vel, debris_masses, debris_accreted,
            use_relativistic
        )

        # Final half-step velocity kick
        bh_velocities[i] = bh_velocities[i] + 0.5 * accel_new * dt


@jit(nopython=True)
def detect_accretion(
    debris_pos: np.ndarray,
    debris_accreted: np.ndarray,
    debris_accreted_by: np.ndarray,
    bh_positions: np.ndarray,
    bh_capture_radii: np.ndarray,
    bh_ring_ids: np.ndarray
) -> int:
    """
    Detect which debris particles are accreted by Ring 0 black holes.

    A debris particle is accreted if it comes within the capture radius
    of a Ring 0 black hole.

    Args:
        debris_pos: Debris positions (N_debris, 3) [meters]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        debris_accreted_by: Debris accreted by BH index (N_debris,) [int]
        bh_positions: BH positions (N_bh, 3) [meters]
        bh_capture_radii: BH capture radii (N_bh,) [meters]
        bh_ring_ids: BH ring IDs (N_bh,) [int]

    Returns:
        Number of newly accreted particles this timestep
    """
    N_debris = len(debris_pos)
    N_bh = len(bh_positions)
    newly_accreted = 0

    for i in range(N_debris):
        # Skip already accreted particles
        if debris_accreted[i]:
            continue

        # Check distance to each Ring 0 BH
        for j in range(N_bh):
            # Only Ring 0 BHs can accrete
            if bh_ring_ids[j] != 0:
                continue

            # Skip if this BH has no capture radius
            if bh_capture_radii[j] <= 0:
                continue

            # Calculate distance from debris to BH
            r_vec = debris_pos[i] - bh_positions[j]
            r_squared = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            r = np.sqrt(r_squared)

            # Check if within capture radius
            if r < bh_capture_radii[j]:
                debris_accreted[i] = True
                debris_accreted_by[i] = j
                newly_accreted += 1
                break  # Only accrete to one BH

    return newly_accreted


@jit(nopython=True)
def apply_accretion_momentum_conservation(
    debris_pos: np.ndarray,
    debris_vel: np.ndarray,
    debris_masses: np.ndarray,
    debris_accreted: np.ndarray,
    debris_accreted_by: np.ndarray,
    bh_positions: np.ndarray,
    bh_velocities: np.ndarray,
    bh_masses: np.ndarray
) -> None:
    """
    Apply momentum conservation when debris is accreted by a black hole.

    When a debris particle is accreted:
    1. BH gains the debris momentum: m_debris * v_debris
    2. BH mass increases by m_debris
    3. BH velocity adjusts to conserve momentum

    Args:
        debris_pos: Debris positions (N_debris, 3) [meters]
        debris_vel: Debris velocities (N_debris, 3) [m/s]
        debris_masses: Debris masses (N_debris,) [kg]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        debris_accreted_by: Debris accreted by BH index (N_debris,) [int]
        bh_positions: BH positions (N_bh, 3) [meters]
        bh_velocities: BH velocities (N_bh, 3) [m/s]
        bh_masses: BH masses (N_bh,) [kg]

    Returns:
        None (modifies BH arrays in place)
    """
    N_debris = len(debris_pos)

    for i in range(N_debris):
        # Only process newly accreted particles
        if not debris_accreted[i]:
            continue

        bh_idx = debris_accreted_by[i]
        if bh_idx < 0:
            continue

        # Calculate momentum before accretion
        p_bh_before = bh_masses[bh_idx] * bh_velocities[bh_idx]
        p_debris = debris_masses[i] * debris_vel[i]

        # Total momentum (conserved)
        p_total = p_bh_before + p_debris

        # Update BH mass
        m_bh_new = bh_masses[bh_idx] + debris_masses[i]

        # Update BH velocity to conserve momentum
        if m_bh_new > 0:
            bh_velocities[bh_idx] = p_total / m_bh_new

        # Update BH mass
        bh_masses[bh_idx] = m_bh_new

        # Move BH position slightly toward accreted debris
        # (This is a simple model; in reality the BH would be at the center of mass)
        # For simplicity, we'll keep the BH position unchanged
        # bh_positions[bh_idx] = (bh_masses[bh_idx] * bh_positions[bh_idx] +
        #                          debris_masses[i] * debris_pos[i]) / m_bh_new


def evolve_system(
    state: SimulationState,
    params: SimulationParameters,
    n_steps: int,
    show_progress: bool = True
) -> dict:
    """
    Evolve the simulation forward for n_steps timesteps.

    This is the main simulation driver. It:
    1. Updates ALL particles (debris + BHs) simultaneously using leapfrog integration
    2. Detects accretion events
    3. Applies momentum conservation for accreted particles
    4. Updates simulation time and timestep counter

    The simultaneous update ensures that both debris and BHs see the same initial
    state at each timestep, which is critical for time-reversibility and energy
    conservation in the leapfrog integrator.

    Args:
        state: SimulationState object (modified in place)
        params: SimulationParameters object
        n_steps: Number of timesteps to evolve
        show_progress: Whether to show progress bar (tqdm)

    Returns:
        Dictionary with simulation statistics:
        - total_accreted: Total number of accreted particles
        - final_time: Final simulation time [seconds]
        - final_timestep: Final timestep number
    """
    # Extract parameters
    dt = params.dt
    use_relativistic = params.use_relativistic_mass

    # Statistics tracking
    total_accreted = 0

    # Progress bar
    if show_progress:
        pbar = tqdm(total=n_steps, desc="Evolving system", unit="steps")

    for step in range(n_steps):
        # 1. Update ALL particles simultaneously (debris + BHs)
        update_all_particles(
            state.debris_positions,
            state.debris_velocities,
            state.debris_masses,
            state.debris_proper_times,
            state.debris_accreted,
            state.bh_positions,
            state.bh_velocities,
            state.bh_masses,
            state.bh_ring_ids,
            state.bh_is_static,
            dt,
            use_relativistic
        )

        # 2. Detect accretion events
        newly_accreted = detect_accretion(
            state.debris_positions,
            state.debris_accreted,
            state.debris_accreted_by,
            state.bh_positions,
            state.bh_capture_radii,
            state.bh_ring_ids
        )

        total_accreted += newly_accreted

        # 4. Apply momentum conservation for newly accreted particles
        if newly_accreted > 0:
            apply_accretion_momentum_conservation(
                state.debris_positions,
                state.debris_velocities,
                state.debris_masses,
                state.debris_accreted,
                state.debris_accreted_by,
                state.bh_positions,
                state.bh_velocities,
                state.bh_masses
            )

        # 5. Update simulation time and timestep counter
        state.time += dt
        state.timestep_count += 1

        # Update progress bar
        if show_progress:
            pbar.update(1)
            if newly_accreted > 0:
                pbar.set_postfix({
                    'accreted': total_accreted,
                    'active': state.n_debris_active
                })

    if show_progress:
        pbar.close()

    return {
        'total_accreted': total_accreted,
        'final_time': state.time,
        'final_timestep': state.timestep_count
    }


def run_simulation(
    params: SimulationParameters,
    seed: int = 42,
    show_progress: bool = True
) -> tuple:
    """
    Run a complete simulation from initialization to completion.

    This is the top-level driver function that:
    1. Initializes the simulation state
    2. Runs the time evolution
    3. Returns the final state and statistics

    Args:
        params: SimulationParameters object
        seed: Random seed for reproducibility
        show_progress: Whether to show progress bar

    Returns:
        (state, stats) tuple:
        - state: Final SimulationState object
        - stats: Dictionary with simulation statistics
    """
    from .initialization import initialize_simulation

    # Initialize simulation
    print("Initializing simulation...")
    state = initialize_simulation(params, seed=seed)

    # Calculate number of steps
    n_steps = int(params.duration / params.dt)
    print(f"Running simulation: {n_steps} steps, dt={params.dt * const.s_to_Gyr:.6f} Gyr")

    # Run evolution
    stats = evolve_system(state, params, n_steps, show_progress=show_progress)

    print(f"\nSimulation complete!")
    print(f"  Final time: {state.time * const.s_to_Gyr:.2f} Gyr")
    print(f"  Total accreted: {stats['total_accreted']}/{state.n_debris} particles")
    print(f"  Active debris: {state.n_debris_active}/{state.n_debris} particles")

    return state, stats
