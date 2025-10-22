"""
Physics functions for black hole explosion simulation.

All performance-critical functions are JIT-compiled with Numba for near-C performance.
These functions must be Numba-compatible (NumPy arrays, no Python objects).
"""

import numpy as np
from numba import jit

from . import constants as const


@jit(nopython=True)
def lorentz_factor(velocity):
    """
    Calculate Lorentz factor γ for a 3D velocity vector.

    γ(v) = 1 / sqrt(1 - v²/c²)

    Args:
        velocity: 3D velocity array [vx, vy, vz] in m/s (shape: (3,))

    Returns:
        float: Lorentz factor γ >= 1.0

    Notes:
        - For v << c: γ ≈ 1.0
        - For v → c: γ → ∞
        - Values are clamped to prevent numerical issues near c
    """
    # Calculate velocity magnitude squared
    v_squared = velocity[0]**2 + velocity[1]**2 + velocity[2]**2

    # Calculate β² = v²/c²
    beta_squared = v_squared / const.c_squared

    # Clamp to avoid numerical issues (prevent β² >= 1.0)
    if beta_squared >= 0.9999:
        beta_squared = 0.9999

    # Calculate γ = 1 / sqrt(1 - β²)
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)

    return gamma


@jit(nopython=True)
def lorentz_factor_scalar(v):
    """
    Calculate Lorentz factor γ for a scalar velocity magnitude.

    Args:
        v: Velocity magnitude in m/s

    Returns:
        float: Lorentz factor γ >= 1.0
    """
    beta_squared = (v * v) / const.c_squared

    # Clamp to avoid numerical issues
    if beta_squared >= 0.9999:
        beta_squared = 0.9999

    gamma = 1.0 / np.sqrt(1.0 - beta_squared)

    return gamma


@jit(nopython=True)
def relativistic_mass(rest_mass, velocity):
    """
    Calculate relativistic mass for a moving object.

    m_rel = γ(v) × m_rest

    Args:
        rest_mass: Rest mass in kg
        velocity: 3D velocity array [vx, vy, vz] in m/s (shape: (3,))

    Returns:
        float: Relativistic mass in kg

    Notes:
        - For v << c: m_rel ≈ m_rest
        - For v → c: m_rel → ∞
    """
    gamma = lorentz_factor(velocity)
    return gamma * rest_mass


@jit(nopython=True)
def gravitational_acceleration_direct(pos_i, pos_j, mass_j, velocity_j, use_relativistic):
    """
    Calculate gravitational acceleration on particle i from particle/BH j.

    a_ij = G × m_j_eff / r² × r_hat

    where:
    - m_j_eff = γ(v_j) × m_j_rest (if use_relativistic)
    - r_hat = (pos_j - pos_i) / |pos_j - pos_i|

    Args:
        pos_i: Position of particle i [x, y, z] in meters (shape: (3,))
        pos_j: Position of particle/BH j [x, y, z] in meters (shape: (3,))
        mass_j: Rest mass of particle/BH j in kg
        velocity_j: Velocity of particle/BH j [vx, vy, vz] in m/s (shape: (3,))
        use_relativistic: Whether to use relativistic mass correction

    Returns:
        accel: 3D acceleration vector [ax, ay, az] in m/s² (shape: (3,))

    Notes:
        - Returns zero if distance < minimum threshold (avoid singularity)
        - Minimum distance threshold: 1e10 meters (~0.001 light-years)
        - Mass of particle i does NOT affect its acceleration (equivalence principle)
    """
    # Vector from i to j
    r_vec = pos_j - pos_i

    # Distance squared
    r_squared = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2

    # Minimum distance threshold to avoid singularities
    if r_squared < 1e20:  # (1e10 meters)^2
        return np.zeros(3)

    # Distance
    r = np.sqrt(r_squared)

    # Effective mass (relativistic if requested)
    if use_relativistic:
        gamma = lorentz_factor(velocity_j)
        m_eff = gamma * mass_j
    else:
        m_eff = mass_j

    # Gravitational acceleration magnitude: a = G × M / r²
    accel_magnitude = const.G * m_eff / r_squared

    # Acceleration vector: a_vec = a_mag × r_hat = a_mag × (r_vec / r)
    accel = accel_magnitude * (r_vec / r)

    return accel


@jit(nopython=True)
def calculate_acceleration_from_bhs(pos, bh_positions, bh_masses_rest,
                                    bh_velocities, bh_is_static, use_relativistic):
    """
    Calculate total gravitational acceleration on a particle from all black holes.

    This is the main acceleration calculation function for debris particles.

    Args:
        pos: Position of particle [x, y, z] in meters (shape: (3,))
        bh_positions: Positions of all BHs (shape: (N_bh, 3))
        bh_masses_rest: Rest masses of all BHs in kg (shape: (N_bh,))
        bh_velocities: Velocities of all BHs (shape: (N_bh, 3))
        bh_is_static: Static flags for BHs (shape: (N_bh,))
        use_relativistic: Whether to use relativistic mass

    Returns:
        accel: 3D acceleration vector [ax, ay, az] in m/s² (shape: (3,))

    Notes:
        - This function accumulates accelerations from all black holes
        - Particle mass does NOT affect acceleration (equivalence principle)
    """
    accel_total = np.zeros(3)
    n_bh = len(bh_positions)

    for j in range(n_bh):
        # Calculate acceleration from BH j
        accel = gravitational_acceleration_direct(
            pos,
            bh_positions[j],
            bh_masses_rest[j],
            bh_velocities[j],
            use_relativistic and not bh_is_static[j]
        )

        accel_total += accel

    return accel_total


@jit(nopython=True, parallel=True)
def update_particle_velocities_and_positions(
    positions, velocities, proper_times,
    bh_positions, bh_masses_rest, bh_velocities, bh_is_static,
    particle_masses, dt, use_relativistic
):
    """
    Update positions, velocities, and proper times for all particles.

    Uses leapfrog integration (velocity Verlet):
    1. v(t + dt/2) = v(t) + a(t) × dt/2
    2. x(t + dt) = x(t) + v(t + dt/2) × dt
    3. a(t + dt) = calculate_acceleration(x(t + dt))
    4. v(t + dt) = v(t + dt/2) + a(t + dt) × dt/2

    Args:
        positions: Particle positions (shape: (N, 3))
        velocities: Particle velocities (shape: (N, 3))
        proper_times: Particle proper times in seconds (shape: (N,))
        bh_positions: BH positions (shape: (N_bh, 3))
        bh_masses_rest: BH rest masses (shape: (N_bh,))
        bh_velocities: BH velocities (shape: (N_bh, 3))
        bh_is_static: BH static flags (shape: (N_bh,))
        particle_masses: Particle masses (shape: (N,))
        dt: Timestep in seconds
        use_relativistic: Whether to use relativistic mass

    Notes:
        - Arrays are modified in place
        - Uses Numba parallel execution for performance
        - Proper time updated with time dilation: dτ = dt / γ
    """
    n_particles = len(positions)

    for i in range(n_particles):
        # Calculate acceleration at current position
        accel = calculate_acceleration_from_bhs(
            positions[i],
            bh_positions,
            bh_masses_rest,
            bh_velocities,
            bh_is_static,
            use_relativistic
        )

        # Update velocity: v_new = v + a * dt
        velocities[i] += accel * dt

        # Update position: x_new = x + v * dt
        # (Using updated velocity - semi-implicit Euler)
        positions[i] += velocities[i] * dt

        # Update proper time with time dilation: dtau = dt / gamma
        gamma = lorentz_factor(velocities[i])
        proper_times[i] += dt / gamma


@jit(nopython=True)
def calculate_kinetic_energy(mass, velocity):
    """
    Calculate relativistic kinetic energy.

    KE = (γ - 1) × m × c²

    Args:
        mass: Rest mass in kg
        velocity: 3D velocity vector in m/s (shape: (3,))

    Returns:
        float: Kinetic energy in Joules
    """
    gamma = lorentz_factor(velocity)
    return (gamma - 1.0) * mass * const.c_squared


@jit(nopython=True)
def calculate_potential_energy(pos1, pos2, mass1, mass2):
    """
    Calculate gravitational potential energy between two objects.

    PE = -G × m1 × m2 / r

    Args:
        pos1: Position of object 1 (shape: (3,))
        pos2: Position of object 2 (shape: (3,))
        mass1: Mass of object 1 in kg
        mass2: Mass of object 2 in kg

    Returns:
        float: Potential energy in Joules (negative)
    """
    r_vec = pos2 - pos1
    r_squared = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2

    if r_squared < 1e20:  # Minimum distance threshold
        return 0.0

    r = np.sqrt(r_squared)

    return -const.G * mass1 * mass2 / r
