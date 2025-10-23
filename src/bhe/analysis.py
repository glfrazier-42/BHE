"""
Post-simulation analysis for black hole explosion simulation.

This module provides functions to analyze simulation results from HDF5 files:
- Calculate redshift from particle velocities
- Analyze escape fractions and accretion statistics
- Calculate proper time distributions
- Generate summary statistics

All functions work with HDF5 file paths (not SimulationState objects).
"""

import numpy as np
import h5py
from typing import Dict, Tuple, Optional
from pathlib import Path

from bhe import constants as const


def calculate_redshift(velocity: np.ndarray) -> float:
    """
    Calculate relativistic Doppler redshift from a velocity vector.

    Uses the special relativistic formula:
    z = sqrt((1 + beta) / (1 - beta)) - 1

    where beta = |v| / c

    Args:
        velocity: 3D velocity vector [vx, vy, vz] in units of c (shape: (3,))

    Returns:
        float: Redshift z (dimensionless)

    Notes:
        - For v << c: z ≈ v/c (classical Doppler)
        - For v → c: z → ∞
        - For approaching (v < 0): z < 0 (blueshift)
        - For receding (v > 0): z > 0 (redshift)
    """
    # Calculate velocity magnitude (already in units of c, so beta = v_mag)
    beta = np.sqrt(np.sum(velocity**2))

    # Clamp to avoid numerical issues at v ≈ c
    if beta >= 0.9999:
        beta = 0.9999

    # Calculate redshift: z = sqrt((1+β)/(1-β)) - 1
    z = np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0

    return z


def calculate_redshift_array(velocities: np.ndarray) -> np.ndarray:
    """
    Calculate redshift for an array of velocity vectors.

    Args:
        velocities: Array of velocity vectors (N, 3) in units of c

    Returns:
        Array of redshifts (N,)
    """
    # Calculate velocity magnitudes (already in units of c, so betas = v_mags)
    betas = np.sqrt(np.sum(velocities**2, axis=1))

    # Clamp to avoid numerical issues
    betas = np.clip(betas, 0, 0.9999)

    # Calculate redshifts
    redshifts = np.sqrt((1.0 + betas) / (1.0 - betas)) - 1.0

    return redshifts


def analyze_simulation(hdf5_filepath: str) -> Dict:
    """
    Analyze a completed simulation from HDF5 file.

    Loads the final timestep and calculates:
    - Number of particles (total, accreted, escaped)
    - Escape fraction (beyond 100 Gly)
    - Redshift distribution statistics
    - Proper time statistics
    - Energy and momentum conservation metrics

    Args:
        hdf5_filepath: Path to HDF5 simulation output file

    Returns:
        Dictionary containing analysis results with keys:
        - 'n_debris_total': Total debris particles
        - 'n_debris_accreted': Number accreted
        - 'n_debris_escaped': Number beyond 100 Gly
        - 'escape_fraction': Fraction escaped
        - 'redshift_mean': Mean redshift of escaped particles
        - 'redshift_std': Std dev of redshift
        - 'proper_time_mean': Mean proper time (Gyr)
        - 'proper_time_std': Std dev of proper time (Gyr)
        - 'energy_conservation_error': Final relative energy error
        - 'momentum_conservation_error': Final relative momentum error
        - 'final_time_gyr': Final simulation time (Gyr)
    """
    with h5py.File(hdf5_filepath, 'r') as f:
        # Get final timestep index
        n_timesteps = len(f['timeseries/time'])
        final_idx = n_timesteps - 1

        # Load final state (unified particle system)
        final_time = f['timeseries/time'][final_idx]
        all_positions = f['timeseries/positions'][final_idx]
        all_velocities = f['timeseries/velocities'][final_idx]
        all_accreted = f['timeseries/accreted'][final_idx]
        all_proper_times = f['timeseries/proper_times'][final_idx]

        # Get metadata to filter debris particles
        from bhe.state import DEBRIS
        particle_type = f['metadata/particle_type'][:]
        debris_mask = (particle_type == DEBRIS)

        # Extract debris data
        debris_pos = all_positions[debris_mask]
        debris_vel = all_velocities[debris_mask]
        debris_accreted = all_accreted[debris_mask]
        debris_proper_times = all_proper_times[debris_mask]

        # Calculate distances from origin
        distances = np.sqrt(np.sum(debris_pos**2, axis=1))

        # Statistics
        n_total = len(debris_pos)
        n_accreted = np.sum(debris_accreted)

        # Escape criterion: beyond 100 Gly and not accreted
        escape_distance = 100.0 * const.Gly
        escaped_mask = (distances > escape_distance) & (~debris_accreted)
        n_escaped = np.sum(escaped_mask)
        escape_fraction = n_escaped / n_total if n_total > 0 else 0.0

        # Redshift statistics for escaped particles
        if n_escaped > 0:
            escaped_velocities = debris_vel[escaped_mask]
            escaped_redshifts = calculate_redshift_array(escaped_velocities)
            redshift_mean = np.mean(escaped_redshifts)
            redshift_std = np.std(escaped_redshifts)
        else:
            redshift_mean = 0.0
            redshift_std = 0.0

        # Proper time statistics for escaped particles (times in yr, convert to Gyr)
        if n_escaped > 0:
            escaped_proper_times = debris_proper_times[escaped_mask]
            proper_time_mean = np.mean(escaped_proper_times) / 1.0e9  # yr to Gyr
            proper_time_std = np.std(escaped_proper_times) / 1.0e9  # yr to Gyr
        else:
            proper_time_mean = 0.0
            proper_time_std = 0.0

        # Conservation metrics
        if 'conservation' in f:
            energy_errors = f['conservation/energy_error'][:]
            momentum_errors = f['conservation/momentum_error'][:]
            energy_error = energy_errors[final_idx] if len(energy_errors) > 0 else 0.0
            momentum_error = momentum_errors[final_idx] if len(momentum_errors) > 0 else 0.0
        else:
            energy_error = 0.0
            momentum_error = 0.0

    results = {
        'n_debris_total': int(n_total),
        'n_debris_accreted': int(n_accreted),
        'n_debris_escaped': int(n_escaped),
        'escape_fraction': float(escape_fraction),
        'redshift_mean': float(redshift_mean),
        'redshift_std': float(redshift_std),
        'proper_time_mean_gyr': float(proper_time_mean),
        'proper_time_std_gyr': float(proper_time_std),
        'energy_conservation_error': float(energy_error),
        'momentum_conservation_error': float(momentum_error),
        'final_time_gyr': float(final_time / 1.0e9)  # yr to Gyr
    }

    return results


def calculate_escape_fraction_vs_time(
    hdf5_filepath: str,
    distance_threshold: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate escape fraction as a function of time.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file
        distance_threshold: Escape distance threshold in Gly (default: 100 Gly)

    Returns:
        Tuple of (times, escape_fractions):
        - times: Array of times in Gyr
        - escape_fractions: Fraction of debris beyond threshold at each time
    """
    threshold_ly = distance_threshold * const.Gly

    with h5py.File(hdf5_filepath, 'r') as f:
        times = f['timeseries/time'][:]
        all_positions = f['timeseries/positions'][:]
        all_accreted = f['timeseries/accreted'][:]

        # Get metadata to filter debris particles
        from bhe.state import DEBRIS
        particle_type = f['metadata/particle_type'][:]
        debris_mask = (particle_type == DEBRIS)

        n_timesteps = len(times)
        n_debris = np.sum(debris_mask)
        escape_fractions = np.zeros(n_timesteps)

        for i in range(n_timesteps):
            # Extract debris data at this timestep
            debris_pos = all_positions[i][debris_mask]
            debris_accreted_flags = all_accreted[i][debris_mask]

            # Calculate distances from origin
            distances = np.sqrt(np.sum(debris_pos**2, axis=1))

            # Count escaped: beyond threshold AND not accreted
            escaped = (distances > threshold_ly) & (~debris_accreted_flags)
            n_escaped = np.sum(escaped)

            escape_fractions[i] = n_escaped / n_debris if n_debris > 0 else 0.0

    # Convert times to Gyr (times are in yr)
    times_gyr = times / 1.0e9

    return times_gyr, escape_fractions


def get_final_debris_state(hdf5_filepath: str) -> Dict[str, np.ndarray]:
    """
    Extract final debris particle state from simulation.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file

    Returns:
        Dictionary containing:
        - 'positions': Positions (N, 3) in ly
        - 'velocities': Velocities (N, 3) in units of c
        - 'proper_times': Proper times (N,) in years
        - 'accreted': Accretion flags (N,) boolean
        - 'distances': Distances from origin (N,) in ly
        - 'redshifts': Redshifts (N,) dimensionless
        - 'time': Final simulation time in years
    """
    with h5py.File(hdf5_filepath, 'r') as f:
        # Get final timestep
        n_timesteps = len(f['timeseries/time'])
        final_idx = n_timesteps - 1

        # Load final state (unified particle system)
        all_positions = f['timeseries/positions'][final_idx]
        all_velocities = f['timeseries/velocities'][final_idx]
        all_proper_times = f['timeseries/proper_times'][final_idx]
        all_accreted = f['timeseries/accreted'][final_idx]
        time = f['timeseries/time'][final_idx]

        # Get metadata to filter debris particles
        from bhe.state import DEBRIS
        particle_type = f['metadata/particle_type'][:]
        debris_mask = (particle_type == DEBRIS)

        # Extract debris data
        positions = all_positions[debris_mask]
        velocities = all_velocities[debris_mask]
        proper_times = all_proper_times[debris_mask]
        accreted = all_accreted[debris_mask]

        # Calculate derived quantities
        distances = np.sqrt(np.sum(positions**2, axis=1))
        redshifts = calculate_redshift_array(velocities)

    return {
        'positions': positions,
        'velocities': velocities,
        'proper_times': proper_times,
        'accreted': accreted,
        'distances': distances,
        'redshifts': redshifts,
        'time': time
    }


def get_ring0_trajectories(hdf5_filepath: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract Ring 0 BH trajectories from simulation.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file

    Returns:
        Dictionary containing Ring 0 trajectory data, or None if no Ring 0:
        - 'times': Times (N_timesteps,) in years
        - 'positions': Positions (N_timesteps, N_ring0, 3) in ly
        - 'velocities': Velocities (N_timesteps, N_ring0, 3) in units of c
        - 'n_ring0': Number of Ring 0 BHs

        Returns None if there are no Ring 0 BHs in the simulation.
    """
    with h5py.File(hdf5_filepath, 'r') as f:
        # Check if we have metadata (unified particle system)
        if 'metadata' not in f or 'metadata/ring_id' not in f:
            return None

        # Get metadata to filter Ring 0 BHs
        from bhe.state import BLACK_HOLE
        particle_type = f['metadata/particle_type'][:]
        ring_id = f['metadata/ring_id'][:]

        # Ring 0 mask: black holes with ring_id == 0
        ring0_mask = (particle_type == BLACK_HOLE) & (ring_id == 0)
        n_ring0 = np.sum(ring0_mask)

        if n_ring0 == 0:
            return None

        # Load all particle data
        times = f['timeseries/time'][:]
        all_positions = f['timeseries/positions'][:]
        all_velocities = f['timeseries/velocities'][:]

        # Extract Ring 0 trajectories (all timesteps, Ring 0 particles only)
        ring0_positions = all_positions[:, ring0_mask, :]
        ring0_velocities = all_velocities[:, ring0_mask, :]

    return {
        'times': times,
        'positions': ring0_positions,
        'velocities': ring0_velocities,
        'n_ring0': n_ring0
    }
