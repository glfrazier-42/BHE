"""
Simulation state management for black hole explosion simulation.

This module defines the SimulationState class which holds all particle data
in NumPy arrays for Numba compatibility and efficient computation.
"""

import numpy as np
from typing import Optional
from pathlib import Path
import h5py

from . import constants as const


class SimulationState:
    """
    Container for all simulation state using NumPy arrays.

    This class is designed to be Numba-compatible by storing all data
    in NumPy arrays rather than Python objects.

    Attributes:
        # Black holes
        bh_positions: (N_bh, 3) array of BH positions [m]
        bh_velocities: (N_bh, 3) array of BH velocities [m/s]
        bh_masses: (N_bh,) array of BH rest masses [kg]
        bh_ring_ids: (N_bh,) array of ring IDs (0, 1, 2, 3)
        bh_is_static: (N_bh,) array of static flags (True/False)
        bh_capture_radii: (N_bh,) array of capture radii [m]

        # Debris particles
        debris_positions: (N_debris, 3) array of debris positions [m]
        debris_velocities: (N_debris, 3) array of debris velocities [m/s]
        debris_masses: (N_debris,) array of debris masses [kg]
        debris_proper_times: (N_debris,) array of proper times [s]
        debris_accreted: (N_debris,) array of accretion flags (True/False)
        debris_accreted_by: (N_debris,) array of BH indices that accreted each particle (-1 if not accreted)

        # Simulation metadata
        time: Current simulation time [s]
        timestep_count: Number of timesteps completed

        # Configuration reference
        M_central: Central BH mass (for reference) [kg]
    """

    def __init__(self, n_bh: int, n_debris: int, M_central: float):
        """
        Initialize empty simulation state arrays.

        Args:
            n_bh: Number of black holes
            n_debris: Number of debris particles
            M_central: Central black hole mass [kg]
        """
        # Black hole arrays
        self.bh_positions = np.zeros((n_bh, 3), dtype=np.float64)
        self.bh_velocities = np.zeros((n_bh, 3), dtype=np.float64)
        self.bh_masses = np.zeros(n_bh, dtype=np.float64)
        self.bh_ring_ids = np.zeros(n_bh, dtype=np.int32)
        self.bh_is_static = np.zeros(n_bh, dtype=bool)
        self.bh_capture_radii = np.zeros(n_bh, dtype=np.float64)

        # Debris particle arrays
        self.debris_positions = np.zeros((n_debris, 3), dtype=np.float64)
        self.debris_velocities = np.zeros((n_debris, 3), dtype=np.float64)
        self.debris_masses = np.zeros(n_debris, dtype=np.float64)
        self.debris_proper_times = np.zeros(n_debris, dtype=np.float64)
        self.debris_accreted = np.zeros(n_debris, dtype=bool)
        self.debris_accreted_by = np.full(n_debris, -1, dtype=np.int32)

        # Simulation metadata
        self.time = 0.0
        self.timestep_count = 0

        # Configuration
        self.M_central = M_central

    @property
    def n_bh(self) -> int:
        """Number of black holes."""
        return len(self.bh_positions)

    @property
    def n_debris(self) -> int:
        """Number of debris particles."""
        return len(self.debris_positions)

    @property
    def n_debris_active(self) -> int:
        """Number of debris particles that have not been accreted."""
        return np.sum(~self.debris_accreted)

    @property
    def n_debris_accreted(self) -> int:
        """Number of debris particles that have been accreted."""
        return np.sum(self.debris_accreted)

    def get_active_debris_mask(self) -> np.ndarray:
        """
        Get boolean mask for active (non-accreted) debris particles.

        Returns:
            Boolean array of shape (N_debris,) where True = active
        """
        return ~self.debris_accreted

    def save_to_hdf5(self, filepath: str, compression: str = "gzip"):
        """
        Save simulation state to HDF5 file.

        Args:
            filepath: Path to HDF5 file
            compression: HDF5 compression method ("gzip", "lzf", or None)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['time'] = self.time
            f.attrs['timestep_count'] = self.timestep_count
            f.attrs['M_central'] = self.M_central
            f.attrs['n_bh'] = self.n_bh
            f.attrs['n_debris'] = self.n_debris

            # Black hole data
            bh_group = f.create_group('black_holes')
            bh_group.create_dataset('positions', data=self.bh_positions, compression=compression)
            bh_group.create_dataset('velocities', data=self.bh_velocities, compression=compression)
            bh_group.create_dataset('masses', data=self.bh_masses, compression=compression)
            bh_group.create_dataset('ring_ids', data=self.bh_ring_ids, compression=compression)
            bh_group.create_dataset('is_static', data=self.bh_is_static, compression=compression)
            bh_group.create_dataset('capture_radii', data=self.bh_capture_radii, compression=compression)

            # Debris data
            debris_group = f.create_group('debris')
            debris_group.create_dataset('positions', data=self.debris_positions, compression=compression)
            debris_group.create_dataset('velocities', data=self.debris_velocities, compression=compression)
            debris_group.create_dataset('masses', data=self.debris_masses, compression=compression)
            debris_group.create_dataset('proper_times', data=self.debris_proper_times, compression=compression)
            debris_group.create_dataset('accreted', data=self.debris_accreted, compression=compression)
            debris_group.create_dataset('accreted_by', data=self.debris_accreted_by, compression=compression)

    @classmethod
    def load_from_hdf5(cls, filepath: str) -> 'SimulationState':
        """
        Load simulation state from HDF5 file.

        Args:
            filepath: Path to HDF5 file

        Returns:
            SimulationState instance loaded from file
        """
        with h5py.File(filepath, 'r') as f:
            # Read metadata
            M_central = f.attrs['M_central']
            n_bh = f.attrs['n_bh']
            n_debris = f.attrs['n_debris']

            # Create empty state
            state = cls(n_bh=n_bh, n_debris=n_debris, M_central=M_central)

            # Load metadata
            state.time = f.attrs['time']
            state.timestep_count = f.attrs['timestep_count']

            # Load black hole data
            bh_group = f['black_holes']
            state.bh_positions[:] = bh_group['positions'][:]
            state.bh_velocities[:] = bh_group['velocities'][:]
            state.bh_masses[:] = bh_group['masses'][:]
            state.bh_ring_ids[:] = bh_group['ring_ids'][:]
            state.bh_is_static[:] = bh_group['is_static'][:]
            state.bh_capture_radii[:] = bh_group['capture_radii'][:]

            # Load debris data
            debris_group = f['debris']
            state.debris_positions[:] = debris_group['positions'][:]
            state.debris_velocities[:] = debris_group['velocities'][:]
            state.debris_masses[:] = debris_group['masses'][:]
            state.debris_proper_times[:] = debris_group['proper_times'][:]
            state.debris_accreted[:] = debris_group['accreted'][:]
            state.debris_accreted_by[:] = debris_group['accreted_by'][:]

        return state

    def __repr__(self) -> str:
        """String representation of simulation state."""
        lines = [
            f"SimulationState(time={self.time / const.Gyr_to_s:.3f} Gyr, "
            f"step={self.timestep_count})",
            f"  Black holes: {self.n_bh}",
            f"  Debris: {self.n_debris_active}/{self.n_debris} active "
            f"({self.n_debris_accreted} accreted)",
        ]

        # Show ring breakdown
        for ring_id in np.unique(self.bh_ring_ids):
            count = np.sum(self.bh_ring_ids == ring_id)
            lines.append(f"    Ring {ring_id}: {count} BHs")

        return "\n".join(lines)
