"""
Time evolution engine for black hole explosion simulation.

This module implements the UNIFIED PARTICLE SYSTEM evolution loop.
All particles (black holes and debris) are evolved identically using
a single leapfrog integrator to ensure perfect conservation laws.

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
from numba import jit
from tqdm import tqdm

from bhe import constants as const
from bhe.physics import update_all_particles_leapfrog
from bhe.state import SimulationState, BLACK_HOLE
from bhe.config import SimulationParameters


@jit(nopython=True)
def detect_accretion(
    positions: np.ndarray,
    accreted: np.ndarray,
    accreted_by: np.ndarray,
    particle_type: np.ndarray,
    ring_id: np.ndarray,
    capture_radius: np.ndarray
) -> int:
    """
    Detect which particles are accreted by Ring 0 black holes.

    A particle is accreted if it comes within the capture radius
    of a Ring 0 black hole. Uses metadata to identify black holes
    and capture radii.

    Args:
        positions: All particle positions [ly] (shape: (N, 3))
        accreted: Accreted flags (shape: (N,))
        accreted_by: Accreted by particle index (shape: (N,))
        particle_type: Particle type flags (BLACK_HOLE=0, DEBRIS=1) (shape: (N,))
        ring_id: Ring ID (0-3 for BHs, -1 for debris) (shape: (N,))
        capture_radius: Capture radius [ly] (shape: (N,))

    Returns:
        Number of newly accreted particles this timestep
    """
    n_total = len(positions)
    newly_accreted = 0

    for i in range(n_total):
        # Skip already accreted particles
        if accreted[i]:
            continue

        # Check distance to each Ring 0 BH with capture radius > 0
        for j in range(n_total):
            # Skip self-accretion
            if i == j:
                continue

            # Only Ring 0 BHs can accrete
            if particle_type[j] != BLACK_HOLE or ring_id[j] != 0:
                continue

            # Skip if this BH has no capture radius
            if capture_radius[j] <= 0:
                continue

            # Skip if accreted BH
            if accreted[j]:
                continue

            # Calculate distance from particle i to BH j
            r_vec = positions[i] - positions[j]
            r_squared = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            r = np.sqrt(r_squared)

            # Check if within capture radius
            if r < capture_radius[j]:
                accreted[i] = True
                accreted_by[i] = j
                newly_accreted += 1
                break  # Only accrete to one BH

    return newly_accreted


@jit(nopython=True)
def apply_accretion_momentum_conservation(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    accreted: np.ndarray,
    accreted_by: np.ndarray,
    just_accreted_mask: np.ndarray
) -> None:
    """
    Apply momentum conservation when particles are accreted by black holes.

    When a particle is accreted:
    1. BH gains the particle's momentum: m_particle * v_particle
    2. BH mass increases by m_particle
    3. BH velocity adjusts to conserve momentum

    Args:
        positions: All particle positions [ly] (shape: (N, 3))
        velocities: All particle velocities [fraction of c] (shape: (N, 3))
        masses: All particle masses [M_sun] (shape: (N,))
        accreted: Accreted flags (shape: (N,))
        accreted_by: Accreted by particle index (shape: (N,))
        just_accreted_mask: Mask of newly accreted particles this timestep (shape: (N,))

    Returns:
        None (modifies arrays in place)
    """
    n_total = len(positions)

    for i in range(n_total):
        # Only process newly accreted particles
        if not just_accreted_mask[i]:
            continue

        bh_idx = accreted_by[i]
        if bh_idx < 0:
            continue

        # Calculate momentum before accretion
        p_bh_before = masses[bh_idx] * velocities[bh_idx]
        p_particle = masses[i] * velocities[i]

        # Total momentum (conserved)
        p_total = p_bh_before + p_particle

        # Update BH mass
        m_bh_new = masses[bh_idx] + masses[i]

        # Update BH velocity to conserve momentum
        if m_bh_new > 0:
            velocities[bh_idx] = p_total / m_bh_new

        # Update BH mass
        masses[bh_idx] = m_bh_new


def evolve_system(
    state: SimulationState,
    params: SimulationParameters,
    n_steps: int,
    show_progress: bool = True,
    recorder=None
) -> dict:
    """
    Evolve the simulation forward for n_steps timesteps.

    This is the main simulation driver using the UNIFIED PARTICLE SYSTEM.
    All particles (black holes and debris) are evolved with the same integrator.

    Steps per timestep:
    1. Update ALL particles simultaneously using unified leapfrog integration
    2. Detect accretion events (using metadata to identify Ring 0 BHs)
    3. Apply momentum conservation for accreted particles
    4. Update simulation time and timestep counter
    5. Record data to HDF5 if recorder provided

    The unified update ensures Newton's 3rd law symmetry and perfect
    conservation of energy and momentum.

    Args:
        state: SimulationState object (modified in place)
        params: SimulationParameters object
        n_steps: Number of timesteps to evolve
        show_progress: Whether to show progress bar (tqdm)
        recorder: Optional SimulationRecorder for data output

    Returns:
        Dictionary with simulation statistics:
        - total_accreted: Total number of accreted particles
        - final_time: Final simulation time [yr]
        - final_timestep: Final timestep number
    """
    # Extract parameters
    dt = params.dt

    # Statistics tracking
    total_accreted = 0

    # Calculate output and checkpoint intervals
    if recorder is not None and params.output_interval > 0:
        output_every = max(1, int(params.output_interval / dt))
    else:
        output_every = None

    if recorder is not None and params.checkpoint_interval > 0:
        checkpoint_every = max(1, int(params.checkpoint_interval / dt))
    else:
        checkpoint_every = None

    # Record initial state
    if recorder is not None:
        recorder.record_timestep(state, check_conservation=True)

    # Progress bar
    if show_progress:
        pbar = tqdm(total=n_steps, desc="Evolving system", unit="steps")

    for step in range(n_steps):
        # Adaptive timestep for early evolution to handle closely-packed initial conditions
        # This prevents catastrophic accelerations when particles are very close together
        if state.timestep_count < 100:
            dt_adaptive = 1e-12  # yr
        elif state.timestep_count < 200:
            dt_adaptive = 1e-9   # yr
        elif state.timestep_count < 300:
            dt_adaptive = 1e-6   # yr
        elif state.timestep_count < 400:
            dt_adaptive = 1e-3   # yr
        elif state.timestep_count < 500:
            dt_adaptive = 1.0    # yr
        elif state.timestep_count < 600:
            dt_adaptive = 1e3    # yr
        else:
            dt_adaptive = dt     # Use configured timestep

        # Track accretion state before update
        old_accreted = state.accreted.copy()

        # 1. Update ALL particles simultaneously using unified leapfrog integrator
        update_all_particles_leapfrog(
            state.positions,
            state.velocities,
            state.masses,
            state.accreted,
            dt_adaptive
        )

        # 2. Detect accretion events using metadata
        newly_accreted = detect_accretion(
            state.positions,
            state.accreted,
            state.accreted_by,
            state.particle_type,
            state.ring_id,
            state.capture_radius
        )

        total_accreted += newly_accreted

        # 3. Apply momentum conservation for newly accreted particles
        if newly_accreted > 0:
            # Identify which particles were just accreted (not old ones)
            just_accreted_mask = state.accreted & ~old_accreted

            apply_accretion_momentum_conservation(
                state.positions,
                state.velocities,
                state.masses,
                state.accreted,
                state.accreted_by,
                just_accreted_mask
            )

        # 4. Update simulation time and timestep counter
        state.time += dt_adaptive
        state.timestep_count += 1

        # 5. Record timestep data if needed
        if recorder is not None and output_every is not None:
            if (step + 1) % output_every == 0:
                recorder.record_timestep(state, check_conservation=True)

        # 6. Save checkpoint if needed
        if recorder is not None and checkpoint_every is not None:
            if (step + 1) % checkpoint_every == 0:
                checkpoint_name = f"step_{state.timestep_count}"
                recorder.save_checkpoint(state, checkpoint_name)

        # Update progress bar
        if show_progress:
            pbar.update(1)
            if newly_accreted > 0:
                pbar.set_postfix({
                    'accreted': total_accreted,
                    'active': state.n_active
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
    from bhe.initialization import initialize_simulation

    # Initialize simulation
    print("Initializing simulation...")
    state = initialize_simulation(params, seed=seed)

    # Calculate number of steps
    n_steps = int(params.duration / params.dt)
    print(f"Running simulation: {n_steps} steps, dt={params.dt / 1.0e9:.6f} Gyr")
    print(f"Total particles: {state.n_total} ({state.n_bh} BHs + {state.n_debris} debris)")

    # Run evolution
    stats = evolve_system(state, params, n_steps, show_progress=show_progress)

    print(f"\nSimulation complete!")
    print(f"  Final time: {state.time / 1.0e9:.2f} Gyr")
    print(f"  Total accreted: {stats['total_accreted']}/{state.n_total} particles")
    print(f"  Active particles: {state.n_active}/{state.n_total}")

    return state, stats
