"""
Data recording and checkpointing for black hole explosion simulation.

This module handles:
- Time series recording to HDF5 files
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

from . import constants as const
from .state import SimulationState
from .config import SimulationParameters


class SimulationRecorder:
    """
    Records simulation data to HDF5 file with checkpointing support.

    The HDF5 file structure:
    /config (group) - Simulation configuration as attributes
    /timeseries (group) - Time series data
        /time (dataset) - Coordinate times [seconds]
        /bh_positions (dataset) - BH positions (n_steps, n_bh, 3) [meters]
        /bh_velocities (dataset) - BH velocities (n_steps, n_bh, 3) [m/s]
        /bh_masses (dataset) - BH masses (n_steps, n_bh) [kg]
        /debris_positions (dataset) - Debris positions (n_steps, n_debris, 3) [meters]
        /debris_velocities (dataset) - Debris velocities (n_steps, n_debris, 3) [m/s]
        /debris_proper_times (dataset) - Debris proper times (n_steps, n_debris) [seconds]
        /debris_accreted (dataset) - Debris accretion flags (n_steps, n_debris) [bool]
    /conservation (group) - Conservation metrics
        /total_energy (dataset) - Total energy at each recorded timestep [Joules]
        /total_momentum (dataset) - Total momentum at each recorded timestep (n_steps, 3) [kg*m/s]
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
        config_group.attrs['use_relativistic_mass'] = params.use_relativistic_mass
        config_group.attrs['debris_count'] = params.debris_count
        config_group.attrs['central_bh_mass'] = params.M_central

        # Store complete configuration as YAML string
        # (Would need to implement params.to_dict() method)
        # config_group.attrs['full_config_yaml'] = yaml.dump(params.to_dict())

    def _create_timeseries_datasets(self, state: SimulationState):
        """Create HDF5 datasets for time series data."""
        ts_group = self.file.create_group('timeseries')

        n_steps = self.n_output_steps
        n_bh = state.n_bh
        n_debris = state.n_debris

        # Time datasets
        ts_group.create_dataset('time', shape=(n_steps,), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('timestep', shape=(n_steps,), dtype=np.int64,
                               compression='gzip', compression_opts=4)

        # Black hole datasets
        ts_group.create_dataset('bh_positions', shape=(n_steps, n_bh, 3), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('bh_velocities', shape=(n_steps, n_bh, 3), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('bh_masses', shape=(n_steps, n_bh), dtype=np.float64,
                               compression='gzip', compression_opts=4)

        # Debris datasets
        ts_group.create_dataset('debris_positions', shape=(n_steps, n_debris, 3), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('debris_velocities', shape=(n_steps, n_debris, 3), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('debris_proper_times', shape=(n_steps, n_debris), dtype=np.float64,
                               compression='gzip', compression_opts=4)
        ts_group.create_dataset('debris_accreted', shape=(n_steps, n_debris), dtype=np.bool_,
                               compression='gzip', compression_opts=4)

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

        # Record BH state
        ts['bh_positions'][idx] = state.bh_positions
        ts['bh_velocities'][idx] = state.bh_velocities
        ts['bh_masses'][idx] = state.bh_masses

        # Record debris state
        ts['debris_positions'][idx] = state.debris_positions
        ts['debris_velocities'][idx] = state.debris_velocities
        ts['debris_proper_times'][idx] = state.debris_proper_times
        ts['debris_accreted'][idx] = state.debris_accreted

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

        # Calculate total energy and momentum
        total_energy = calculate_total_energy(
            state.debris_positions, state.debris_velocities, state.debris_masses,
            state.debris_accreted, state.bh_positions, state.bh_velocities,
            state.bh_masses, self.params.use_relativistic_mass
        )

        total_momentum = calculate_total_momentum(
            state.debris_velocities, state.debris_masses, state.debris_accreted,
            state.bh_velocities, state.bh_masses
        )

        # Record
        cons['total_energy'][idx] = total_energy
        cons['total_momentum'][idx] = total_momentum

        # Store initial values for error calculation
        if self.initial_energy is None:
            self.initial_energy = total_energy
            self.initial_momentum = np.linalg.norm(total_momentum)
            cons['energy_error'][idx] = 0.0
            cons['momentum_error'][idx] = 0.0
        else:
            # Calculate relative errors
            energy_error = abs(total_energy - self.initial_energy) / abs(self.initial_energy)
            momentum_error = abs(np.linalg.norm(total_momentum) - self.initial_momentum) / max(self.initial_momentum, 1e-100)

            cons['energy_error'][idx] = energy_error
            cons['momentum_error'][idx] = momentum_error

            # Warn if conservation violated
            if energy_error > 0.01:  # > 1%
                warnings.warn(f"Energy conservation violated by {energy_error*100:.2f}% at t={state.time*const.s_to_Gyr:.3f} Gyr")
            if momentum_error > 0.01:  # > 1%
                warnings.warn(f"Momentum conservation violated by {momentum_error*100:.2f}% at t={state.time*const.s_to_Gyr:.3f} Gyr")

    def save_checkpoint(self, state: SimulationState, checkpoint_name: Optional[str] = None):
        """
        Save full simulation state as checkpoint for restart.

        Args:
            state: Current simulation state
            checkpoint_name: Optional name for checkpoint (default: checkpoint_{timestep})
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{state.timestep_count:08d}"

        # Get checkpoints group
        ckpt_group = self.file['checkpoints']

        if checkpoint_name in ckpt_group:
            del ckpt_group[checkpoint_name]

        cp = ckpt_group.create_group(checkpoint_name)

        # Save metadata
        cp.attrs['time'] = state.time
        cp.attrs['timestep_count'] = state.timestep_count

        # Save BH state
        cp.create_dataset('bh_positions', data=state.bh_positions, compression='gzip')
        cp.create_dataset('bh_velocities', data=state.bh_velocities, compression='gzip')
        cp.create_dataset('bh_masses', data=state.bh_masses, compression='gzip')
        cp.create_dataset('bh_ring_ids', data=state.bh_ring_ids, compression='gzip')
        cp.create_dataset('bh_is_static', data=state.bh_is_static, compression='gzip')
        cp.create_dataset('bh_capture_radii', data=state.bh_capture_radii, compression='gzip')

        # Save debris state
        cp.create_dataset('debris_positions', data=state.debris_positions, compression='gzip')
        cp.create_dataset('debris_velocities', data=state.debris_velocities, compression='gzip')
        cp.create_dataset('debris_masses', data=state.debris_masses, compression='gzip')
        cp.create_dataset('debris_proper_times', data=state.debris_proper_times, compression='gzip')
        cp.create_dataset('debris_accreted', data=state.debris_accreted, compression='gzip')
        cp.create_dataset('debris_accreted_by', data=state.debris_accreted_by, compression='gzip')

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


def load_checkpoint(filepath: str, checkpoint_name: str) -> SimulationState:
    """
    Load simulation state from checkpoint.

    Args:
        filepath: Path to HDF5 file
        checkpoint_name: Name of checkpoint to load

    Returns:
        Restored SimulationState object
    """
    with h5py.File(filepath, 'r') as f:
        cp = f['checkpoints'][checkpoint_name]

        # Create state object
        # Note: We need to create SimulationState with correct sizes
        # This is a simplified version - might need adjustment
        state = SimulationState.__new__(SimulationState)

        # Load BH state
        state.bh_positions = cp['bh_positions'][:]
        state.bh_velocities = cp['bh_velocities'][:]
        state.bh_masses = cp['bh_masses'][:]
        state.bh_ring_ids = cp['bh_ring_ids'][:]
        state.bh_is_static = cp['bh_is_static'][:]
        state.bh_capture_radii = cp['bh_capture_radii'][:]

        # Load debris state
        state.debris_positions = cp['debris_positions'][:]
        state.debris_velocities = cp['debris_velocities'][:]
        state.debris_masses = cp['debris_masses'][:]
        state.debris_proper_times = cp['debris_proper_times'][:]
        state.debris_accreted = cp['debris_accreted'][:]
        state.debris_accreted_by = cp['debris_accreted_by'][:]

        # Load metadata
        state.time = cp.attrs['time']
        state.timestep_count = cp.attrs['timestep_count']

    return state


def calculate_total_energy(
    debris_pos: np.ndarray,
    debris_vel: np.ndarray,
    debris_masses: np.ndarray,
    debris_accreted: np.ndarray,
    bh_pos: np.ndarray,
    bh_vel: np.ndarray,
    bh_masses: np.ndarray,
    use_relativistic: bool
) -> float:
    """
    Calculate total energy of the system.

    E_total = sum(KE) + sum(PE) + sum(rest_mass_energy)

    where:
    - KE = (gamma - 1) * m * c^2 (relativistic kinetic energy)
    - PE = -G * m1 * m2 / r (gravitational potential energy, pairwise)
    - rest_mass = m * c^2

    Args:
        debris_pos: Debris positions (N_debris, 3) [meters]
        debris_vel: Debris velocities (N_debris, 3) [m/s]
        debris_masses: Debris masses (N_debris,) [kg]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        bh_pos: BH positions (N_bh, 3) [meters]
        bh_vel: BH velocities (N_bh, 3) [m/s]
        bh_masses: BH masses (N_bh,) [kg]
        use_relativistic: Whether to use relativistic formulas

    Returns:
        Total energy [Joules]
    """
    from .physics import lorentz_factor

    total_energy = 0.0

    # Kinetic energy and rest mass energy
    for i in range(len(debris_masses)):
        if debris_accreted[i]:
            continue

        if use_relativistic:
            gamma = lorentz_factor(debris_vel[i])
            # Total energy = gamma * m * c^2 = KE + rest_mass
            total_energy += gamma * debris_masses[i] * const.c_squared
        else:
            # Non-relativistic: KE = 0.5 * m * v^2, rest mass = m * c^2
            v_squared = np.sum(debris_vel[i]**2)
            total_energy += 0.5 * debris_masses[i] * v_squared
            total_energy += debris_masses[i] * const.c_squared

    for i in range(len(bh_masses)):
        if use_relativistic:
            gamma = lorentz_factor(bh_vel[i])
            total_energy += gamma * bh_masses[i] * const.c_squared
        else:
            v_squared = np.sum(bh_vel[i]**2)
            total_energy += 0.5 * bh_masses[i] * v_squared
            total_energy += bh_masses[i] * const.c_squared

    # Gravitational potential energy (pairwise)
    # Debris-debris
    for i in range(len(debris_masses)):
        if debris_accreted[i]:
            continue
        for j in range(i+1, len(debris_masses)):
            if debris_accreted[j]:
                continue
            r_vec = debris_pos[j] - debris_pos[i]
            r = np.linalg.norm(r_vec)
            if r > 1e10:  # Avoid singularities
                total_energy -= const.G * debris_masses[i] * debris_masses[j] / r

    # Debris-BH
    for i in range(len(debris_masses)):
        if debris_accreted[i]:
            continue
        for j in range(len(bh_masses)):
            r_vec = bh_pos[j] - debris_pos[i]
            r = np.linalg.norm(r_vec)
            if r > 1e10:
                total_energy -= const.G * debris_masses[i] * bh_masses[j] / r

    # BH-BH
    for i in range(len(bh_masses)):
        for j in range(i+1, len(bh_masses)):
            r_vec = bh_pos[j] - bh_pos[i]
            r = np.linalg.norm(r_vec)
            if r > 1e10:
                total_energy -= const.G * bh_masses[i] * bh_masses[j] / r

    return total_energy


def calculate_total_momentum(
    debris_vel: np.ndarray,
    debris_masses: np.ndarray,
    debris_accreted: np.ndarray,
    bh_vel: np.ndarray,
    bh_masses: np.ndarray
) -> np.ndarray:
    """
    Calculate total momentum of the system.

    P_total = sum(m * v) for all particles

    Args:
        debris_vel: Debris velocities (N_debris, 3) [m/s]
        debris_masses: Debris masses (N_debris,) [kg]
        debris_accreted: Debris accretion flags (N_debris,) [bool]
        bh_vel: BH velocities (N_bh, 3) [m/s]
        bh_masses: BH masses (N_bh,) [kg]

    Returns:
        Total momentum vector (3,) [kg*m/s]
    """
    total_momentum = np.zeros(3)

    # Debris momentum
    for i in range(len(debris_masses)):
        if not debris_accreted[i]:
            total_momentum += debris_masses[i] * debris_vel[i]

    # BH momentum
    for i in range(len(bh_masses)):
        total_momentum += bh_masses[i] * bh_vel[i]

    return total_momentum
