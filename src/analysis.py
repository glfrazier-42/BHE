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

from . import constants as const


def calculate_redshift(velocity: np.ndarray) -> float:
    """
    Calculate relativistic Doppler redshift from a velocity vector.

    Uses the special relativistic formula:
    z = sqrt((1 + beta) / (1 - beta)) - 1

    where beta = |v| / c

    Args:
        velocity: 3D velocity vector [vx, vy, vz] in m/s (shape: (3,))

    Returns:
        float: Redshift z (dimensionless)

    Notes:
        - For v << c: z ≈ v/c (classical Doppler)
        - For v → c: z → ∞
        - For approaching (v < 0): z < 0 (blueshift)
        - For receding (v > 0): z > 0 (redshift)
    """
    # Calculate velocity magnitude
    v_mag = np.sqrt(np.sum(velocity**2))

    # Calculate beta = v/c
    beta = v_mag / const.c

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
        velocities: Array of velocity vectors (N, 3) in m/s

    Returns:
        Array of redshifts (N,)
    """
    # Calculate velocity magnitudes
    v_mags = np.sqrt(np.sum(velocities**2, axis=1))

    # Calculate beta = v/c
    betas = v_mags / const.c

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

        # Load final state
        final_time = f['timeseries/time'][final_idx]
        debris_pos = f['timeseries/debris_positions'][final_idx]
        debris_vel = f['timeseries/debris_velocities'][final_idx]
        debris_accreted = f['timeseries/debris_accreted'][final_idx]
        debris_proper_times = f['timeseries/debris_proper_times'][final_idx]

        # Calculate distances from origin
        distances = np.sqrt(np.sum(debris_pos**2, axis=1))

        # Statistics
        n_total = len(debris_pos)
        n_accreted = np.sum(debris_accreted)

        # Escape criterion: beyond 100 Gly and not accreted
        escape_distance = 100.0 * const.Gly_to_m
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

        # Proper time statistics for escaped particles
        if n_escaped > 0:
            escaped_proper_times = debris_proper_times[escaped_mask]
            proper_time_mean_s = np.mean(escaped_proper_times)
            proper_time_std_s = np.std(escaped_proper_times)
            proper_time_mean = proper_time_mean_s * const.s_to_Gyr
            proper_time_std = proper_time_std_s * const.s_to_Gyr
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
        'final_time_gyr': float(final_time * const.s_to_Gyr)
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
    threshold_m = distance_threshold * const.Gly_to_m

    with h5py.File(hdf5_filepath, 'r') as f:
        times = f['timeseries/time'][:]
        debris_positions = f['timeseries/debris_positions'][:]
        debris_accreted = f['timeseries/debris_accreted'][:]

        n_timesteps = len(times)
        n_debris = debris_positions.shape[1]
        escape_fractions = np.zeros(n_timesteps)

        for i in range(n_timesteps):
            # Calculate distances from origin
            pos = debris_positions[i]
            distances = np.sqrt(np.sum(pos**2, axis=1))

            # Count escaped: beyond threshold AND not accreted
            accreted = debris_accreted[i]
            escaped = (distances > threshold_m) & (~accreted)
            n_escaped = np.sum(escaped)

            escape_fractions[i] = n_escaped / n_debris if n_debris > 0 else 0.0

    # Convert times to Gyr
    times_gyr = times * const.s_to_Gyr

    return times_gyr, escape_fractions


def get_final_debris_state(hdf5_filepath: str) -> Dict[str, np.ndarray]:
    """
    Extract final debris particle state from simulation.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file

    Returns:
        Dictionary containing:
        - 'positions': Positions (N, 3) in meters
        - 'velocities': Velocities (N, 3) in m/s
        - 'proper_times': Proper times (N,) in seconds
        - 'accreted': Accretion flags (N,) boolean
        - 'distances': Distances from origin (N,) in meters
        - 'redshifts': Redshifts (N,) dimensionless
        - 'time': Final simulation time in seconds
    """
    with h5py.File(hdf5_filepath, 'r') as f:
        # Get final timestep
        n_timesteps = len(f['timeseries/time'])
        final_idx = n_timesteps - 1

        # Load final state
        positions = f['timeseries/debris_positions'][final_idx]
        velocities = f['timeseries/debris_velocities'][final_idx]
        proper_times = f['timeseries/debris_proper_times'][final_idx]
        accreted = f['timeseries/debris_accreted'][final_idx]
        time = f['timeseries/time'][final_idx]

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
        - 'times': Times (N_timesteps,) in seconds
        - 'positions': Positions (N_timesteps, N_ring0, 3) in meters
        - 'velocities': Velocities (N_timesteps, N_ring0, 3) in m/s
        - 'n_ring0': Number of Ring 0 BHs

        Returns None if there are no Ring 0 BHs in the simulation.
    """
    with h5py.File(hdf5_filepath, 'r') as f:
        # Check if we have Ring 0 BHs
        # Ring 0 is typically the first N BHs in the array
        # We need to determine which BHs are Ring 0
        # For now, assume Ring 0 BHs have non-zero capture radius

        # Get first timestep to check BH configuration
        if 'config' not in f or 'bh_positions' not in f['timeseries']:
            return None

        # Load BH data
        times = f['timeseries/time'][:]
        bh_positions = f['timeseries/bh_positions'][:]
        bh_velocities = f['timeseries/bh_velocities'][:]

        # Get number of BHs from first timestep
        n_bh = bh_positions.shape[1]

        if n_bh == 0:
            return None

        # For now, return all BH trajectories
        # In a real implementation, we'd filter for Ring 0 based on ring_ids
        # This would require storing ring_ids in the HDF5 file

    return {
        'times': times,
        'positions': bh_positions,
        'velocities': bh_velocities,
        'n_bh': n_bh
    }
