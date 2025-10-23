"""
Data recording and checkpointing for black hole explosion simulation.

This module handles:
- Time series recording to HDF5 files (unified particle system)
- Checkpoint/restart functionality for long simulations
- Energy and momentum conservation monitoring
- Configuration storage for reproducibility

All data is stored in HDF5 format with compression for efficiency.
"""

import h5py
import numpy as np
import yaml
from pathlib import Path
from typing import Optional
import warnings

from bhe import constants as const
from bhe.state import SimulationState, BLACK_HOLE, DEBRIS
from bhe.config import SimulationParameters


class SimulationRecorder:
    """
    Records simulation data to HDF5 file with checkpointing support.

    The HDF5 file structure for UNIFIED PARTICLE SYSTEM:
    /config (group) - Simulation configuration as attributes
    /timeseries (group) - Time series data
        /time (dataset) - Coordinate times [years]
        /positions (dataset) - All particle positions (n_steps, n_total, 3) [ly]
        /velocities (dataset) - All particle velocities (n_steps, n_total, 3) [fraction of c]
        /masses (dataset) - All particle masses (n_steps, n_total) [M_sun]
        /accreted (dataset) - Accretion flags (n_steps, n_total) [bool]
        /proper_times (dataset) - Proper times (n_steps, n_total) [years]
    /metadata (group) - Particle metadata (constant throughout simulation)
        /particle_type (dataset) - Particle types (n_total,) [BLACK_HOLE=0, DEBRIS=1]
        /ring_id (dataset) - Ring IDs (n_total,) [0-3 for BHs, -1 for debris]
        /capture_radius (dataset) - Capture radii (n_total,) [ly]
        /initial_speed (dataset) - Initial speeds (n_total,) [fraction of c]
        /initial_position (dataset) - Initial positions (n_total, 3) [ly]
    /conservation (group) - Conservation metrics
        /total_energy (dataset) - Total energy at each recorded timestep [M_sun × c²]
        /total_momentum (dataset) - Total momentum at each recorded timestep (n_steps, 3) [M_sun × c]
        /energy_error (dataset) - Relative energy error [dimensionless]
        /momentum_error (dataset) - Relative momentum error [dimensionless]
    /checkpoints (group) - Full state checkpoints
        /checkpoint_NNN (group) - Checkpoint at timestep NNN
    """

    def __init__(self, filepath: str, params: SimulationParameters, state: SimulationState):
        """
        Initialize recorder and create HDF5 file.

        Args:
            filepath: Path to HDF5 output file
            params: Simulation parameters
            state: Initial simulation state
        """
        self.filepath = Path(filepath)
        self.params = params

        # Create parent directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create HDF5 file
        self.file = h5py.File(str(self.filepath), 'w')

        # Store configuration
        self._save_configuration(params)

        # Calculate expected number of output steps
        n_total_steps = int(params.duration / params.dt)
        if params.output_interval > 0:
            output_every = max(1, int(params.output_interval / params.dt))
            self.n_output_steps = n_total_steps // output_every + 1
        else:
            self.n_output_steps = n_total_steps + 1

        # Create timeseries datasets with compression
        self._create_timeseries_datasets(state)

        # Save metadata (constant throughout simulation)
        self._save_metadata(state)

        # Create conservation datasets
        self._create_conservation_datasets()

        # Create checkpoints group
        self.file.create_group('checkpoints')

        # Track current output index
        self.current_output_idx = 0

        # Store initial energy and momentum for conservation checks
        self.initial_energy = None
        self.initial_momentum = None

    def _save_configuration(self, params: SimulationParameters):
        """Save simulation configuration to HDF5 file."""
        config_group = self.file.create_group('config')

        # Store key parameters as attributes
        config_group.attrs['simulation_name'] = params.simulation_name
        config_group.attrs['output_directory'] = params.output_directory
        config_group.attrs['dt'] = params.dt
        config_group.attrs['duration'] = params.duration
        config_group.attrs['output_interval'] = params.output_interval
        config_group.attrs['checkpoint_interval'] = params.checkpoint_interval
        config_group.attrs['use_newtonian_enhancements'] = params.use_newtonian_enhancements
        config_group.attrs['debris_count'] = params.debris_count
        config_group.attrs['central_bh_mass'] = params.M_central

    def _create_timeseries_datasets(self, state: SimulationState):
        """Create HDF5 datasets for time series data (unified particle system)."""
        ts_group = self.file.create_group('timeseries')

        n_steps = self.n_output_steps
        n_total = state.n_total

        # Time datasets
        ts_group.create_dataset('time', shape=(n_steps,), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('timestep', shape=(n_steps,), dtype=np.int64,
                               compression='gzip', compression_opts=4)

        # Unified particle datasets
        ts_group.create_dataset('positions', shape=(n_steps, n_total, 3), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('velocities', shape=(n_steps, n_total, 3), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('masses', shape=(n_steps, n_total), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('accreted', shape=(n_steps, n_total), dtype=np.bool_,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('proper_times', shape=(n_steps, n_total), dtype=np.float64,
                               compression='gzip', compression_opts=4)

    def _save_metadata(self, state: SimulationState):
        """Save particle metadata (constant throughout simulation)."""
        meta_group = self.file.create_group('metadata')

        # Save metadata arrays
        meta_group.create_dataset('particle_id', data=state.particle_id, compression='gzip')
        meta_group.create_dataset('particle_type', data=state.particle_type, compression='gzip')
        meta_group.create_dataset('ring_id', data=state.ring_id, compression='gzip')
        meta_group.create_dataset('capture_radius', data=state.capture_radius, compression='gzip')
        meta_group.create_dataset('initial_speed', data=state.initial_speed, compression='gzip')
        meta_group.create_dataset('initial_position', data=state.initial_position, compression='gzip')

        # Store counts for convenience
        meta_group.attrs['n_total'] = state.n_total
        meta_group.attrs['n_bh'] = state.n_bh
        meta_group.attrs['n_debris'] = state.n_debris

    def _create_conservation_datasets(self):
        """Create HDF5 datasets for conservation monitoring."""
        cons_group = self.file.create_group('conservation')

        n_steps = self.n_output_steps

        cons_group.create_dataset('total_energy', shape=(n_steps,), dtype=np.float64,
                                 compression='gzip', compression_opts=4)
        cons_group.create_dataset('total_momentum', shape=(n_steps, 3), dtype=np.float64,
                                 compression='gzip', compression_opts=4)
        cons_group.create_dataset('energy_error', shape=(n_steps,), dtype=np.float64,
                                 compression='gzip', compression_opts=4)
        cons_group.create_dataset('momentum_error', shape=(n_steps,), dtype=np.float64,
                                 compression='gzip', compression_opts=4)

    def record_timestep(self, state: SimulationState, check_conservation: bool = True):
        """
        Record current simulation state to timeseries.

        Args:
            state: Current simulation state
            check_conservation: Whether to check energy/momentum conservation
        """
        if self.current_output_idx >= self.n_output_steps:
            warnings.warn(f"Output buffer full (idx={self.current_output_idx}), skipping record")
            return

        idx = self.current_output_idx
        ts = self.file['timeseries']

        # Record time
        ts['time'][idx] = state.time
        ts['timestep'][idx] = state.timestep_count

        # Record unified particle state
        ts['positions'][idx] = state.positions
        ts['velocities'][idx] = state.velocities
        ts['masses'][idx] = state.masses
        ts['accreted'][idx] = state.accreted
        ts['proper_times'][idx] = state.proper_times

        # Check conservation if requested
        if check_conservation:
            self._record_conservation(state, idx)

        self.current_output_idx += 1

    def _record_conservation(self, state: SimulationState, idx: int):
        """
        Calculate and record conservation metrics.

        Args:
            state: Current simulation state
            idx: Output index
        """
        cons = self.file['conservation']

        # Calculate total energy and momentum using unified arrays
        total_energy = calculate_total_energy(
            state.positions,
            state.velocities,
            state.masses,
            state.accreted
        )

        total_momentum = calculate_total_momentum(
            state.velocities,
            state.masses,
            state.accreted
        )

        # Record
        cons['total_energy'][idx] = total_energy
        cons['total_momentum'][idx] = total_momentum

        # Establish baseline at idx=1 (after first output interval) to avoid t=0 near-zero momentum
        if idx == 0:
            # First output: record raw values but no error yet (no baseline)
            cons['energy_error'][idx] = np.nan
            cons['momentum_error'][idx] = np.nan
        elif idx == 1:
            # Second output: establish baseline
            self.initial_energy = total_energy
            self.initial_momentum = np.linalg.norm(total_momentum)
            cons['energy_error'][idx] = 0.0
            cons['momentum_error'][idx] = 0.0
        else:
            # Subsequent outputs: calculate errors relative to idx=1 baseline
            energy_error = abs(total_energy - self.initial_energy) / abs(self.initial_energy)
            momentum_error = abs(np.linalg.norm(total_momentum) - self.initial_momentum) / max(self.initial_momentum, 1e-100)

            cons['energy_error'][idx] = energy_error
            cons['momentum_error'][idx] = momentum_error

            # Warn if conservation violated
            if energy_error > 0.01:  # > 1%
                warnings.warn(f"Energy conservation violated by {energy_error*100:.2f}% at t={state.time/1e9:.3f} Gyr")
            if momentum_error > 0.01:  # > 1%
                warnings.warn(f"Momentum conservation violated by {momentum_error*100:.2f}% at t={state.time/1e9:.3f} Gyr")

    def save_checkpoint(self, state: SimulationState, checkpoint_name: Optional[str] = None):
        """
        Save full simulation state as checkpoint for restart.

        Args:
            state: Current simulation state
            checkpoint_name: Optional name for checkpoint (default: checkpoint_{timestep})
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{state.timestep_count:08d}"

        # Use SimulationState's built-in HDF5 save method
        ckpt_path = self.filepath.parent / f"{checkpoint_name}.h5"
        state.save_to_hdf5(str(ckpt_path))

    def close(self):
        """Close HDF5 file."""
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def load_checkpoint(filepath: str) -> SimulationState:
    """
    Load simulation state from checkpoint.

    Args:
        filepath: Path to checkpoint HDF5 file

    Returns:
        Restored SimulationState object
    """
    return SimulationState.load_from_hdf5(filepath)


def calculate_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    accreted: np.ndarray
) -> float:
    """
    Calculate total energy of the system (unified particle system).

    E_total = sum(KE) + sum(PE) + sum(rest_mass_energy)

    where:
    - KE = 0.5 * m * v^2 (Newtonian kinetic energy)
    - PE = -G * m1 * m2 / r (gravitational potential energy, pairwise)
    - rest_mass = m * c^2

    Args:
        positions: All particle positions (N, 3) [ly]
        velocities: All particle velocities (N, 3) [fraction of c]
        masses: All particle masses (N,) [M_sun]
        accreted: Accretion flags (N,) [bool]

    Returns:
        Total energy [M_sun × c²]
    """
    total_energy = 0.0
    n_total = len(positions)

    # Kinetic energy and rest mass energy (Newtonian only)
    for i in range(n_total):
        if accreted[i]:
            continue

        # Non-relativistic: KE = 0.5 * m * v^2, rest mass = m * c^2
        v_squared = np.sum(velocities[i]**2)
        total_energy += 0.5 * masses[i] * v_squared
        total_energy += masses[i] * const.c_squared

    # Gravitational potential energy (pairwise)
    for i in range(n_total):
        if accreted[i]:
            continue
        for j in range(i+1, n_total):
            if accreted[j]:
                continue
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r > 0.001:  # Avoid singularities (0.001 ly minimum)
                total_energy -= const.G * masses[i] * masses[j] / r

    return total_energy


def calculate_total_momentum(
    velocities: np.ndarray,
    masses: np.ndarray,
    accreted: np.ndarray
) -> np.ndarray:
    """
    Calculate total momentum of the system (unified particle system).

    P_total = sum(m * v) for all particles (Newtonian)

    Args:
        velocities: All particle velocities (N, 3) [fraction of c]
        masses: All particle masses (N,) [M_sun]
        accreted: Accretion flags (N,) [bool]

    Returns:
        Total momentum vector (3,) [M_sun × c]
    """
    total_momentum = np.zeros(3)
    n_total = len(velocities)

    for i in range(n_total):
        if accreted[i]:
            continue

        total_momentum += masses[i] * velocities[i]

    return total_momentum
