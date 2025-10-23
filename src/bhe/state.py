"""
Simulation state management for black hole explosion simulation.

This module defines the SimulationState class with a UNIFIED PARTICLE SYSTEM.
All particles (black holes and debris) are stored in the same arrays and
evolved with the same integration scheme to ensure perfect conservation laws.

Metadata arrays enable tracking particle identity and properties for analysis.
"""

import numpy as np
from typing import Optional
from pathlib import Path
import h5py

from bhe import constants as const

# Particle type constants
BLACK_HOLE = 0
DEBRIS = 1


class SimulationState:
    """
    Unified particle system for N-body simulation.

    All particles (black holes and debris) are stored in unified arrays and
    evolved identically to ensure conservation of energy and momentum.

    Metadata arrays track particle identity and properties for analysis:
    - particle_type: BLACK_HOLE=0, DEBRIS=1
    - ring_id: 0-3 for BHs, -1 for debris
    - capture_radius: accretion radius (>0 for Ring 0 BHs only)
    - initial_speed: speed at t=0 for filtering
    - initial_position: position at t=0 for tracking

    All physics arrays use natural units:
    - positions: light-years (ly)
    - velocities: fraction of speed of light (c = 1.0 ly/yr)
    - masses: solar masses (M_sun)
    - proper_times: years (yr)
    """

    def __init__(self, n_total: int, M_central: float):
        """
        Initialize unified particle system.

        Args:
            n_total: Total number of particles (BHs + debris)
            M_central: Central black hole mass [M_sun] (for reference)
        """
        # Core physics arrays - used by integration
        self.positions = np.zeros((n_total, 3), dtype=np.float64)  # [ly]
        self.velocities = np.zeros((n_total, 3), dtype=np.float64)  # [fraction of c]
        self.masses = np.zeros(n_total, dtype=np.float64)  # [M_sun]
        self.accreted = np.zeros(n_total, dtype=bool)  # Has been accreted?
        self.proper_times = np.zeros(n_total, dtype=np.float64)  # [yr]

        # Metadata arrays - for tracking and analysis
        self.particle_id = np.arange(n_total, dtype=np.int32)  # Unique ID
        self.particle_type = np.zeros(n_total, dtype=np.int32)  # BLACK_HOLE or DEBRIS
        self.ring_id = np.full(n_total, -1, dtype=np.int32)  # Ring 0-3 or -1
        self.capture_radius = np.zeros(n_total, dtype=np.float64)  # [ly]
        self.initial_speed = np.zeros(n_total, dtype=np.float64)  # [fraction of c]
        self.initial_position = np.zeros((n_total, 3), dtype=np.float64)  # [ly]
        self.accreted_by = np.full(n_total, -1, dtype=np.int32)  # Particle ID that accreted this

        # Simulation metadata
        self.time = 0.0  # [yr]
        self.timestep_count = 0
        self.M_central = M_central  # [M_sun]

    @property
    def n_total(self) -> int:
        """Total number of particles."""
        return len(self.positions)

    @property
    def n_bh(self) -> int:
        """Number of black hole particles."""
        return np.sum(self.particle_type == BLACK_HOLE)

    @property
    def n_debris(self) -> int:
        """Number of debris particles."""
        return np.sum(self.particle_type == DEBRIS)

    @property
    def n_active(self) -> int:
        """Number of active (non-accreted) particles."""
        return np.sum(~self.accreted)

    @property
    def n_accreted(self) -> int:
        """Number of accreted particles."""
        return np.sum(self.accreted)

    def get_black_hole_mask(self) -> np.ndarray:
        """Get boolean mask for black hole particles."""
        return self.particle_type == BLACK_HOLE

    def get_debris_mask(self) -> np.ndarray:
        """Get boolean mask for debris particles."""
        return self.particle_type == DEBRIS

    def get_active_mask(self) -> np.ndarray:
        """Get boolean mask for active (non-accreted) particles."""
        return ~self.accreted

    def get_ring_mask(self, ring_id: int) -> np.ndarray:
        """Get boolean mask for particles in specific ring."""
        return self.ring_id == ring_id

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
            f.attrs['n_total'] = self.n_total

            # Core physics arrays
            physics = f.create_group('physics')
            physics.create_dataset('positions', data=self.positions, compression=compression)
            physics.create_dataset('velocities', data=self.velocities, compression=compression)
            physics.create_dataset('masses', data=self.masses, compression=compression)
            physics.create_dataset('accreted', data=self.accreted, compression=compression)
            physics.create_dataset('proper_times', data=self.proper_times, compression=compression)

            # Metadata arrays
            metadata = f.create_group('metadata')
            metadata.create_dataset('particle_id', data=self.particle_id, compression=compression)
            metadata.create_dataset('particle_type', data=self.particle_type, compression=compression)
            metadata.create_dataset('ring_id', data=self.ring_id, compression=compression)
            metadata.create_dataset('capture_radius', data=self.capture_radius, compression=compression)
            metadata.create_dataset('initial_speed', data=self.initial_speed, compression=compression)
            metadata.create_dataset('initial_position', data=self.initial_position, compression=compression)
            metadata.create_dataset('accreted_by', data=self.accreted_by, compression=compression)

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
            n_total = f.attrs['n_total']

            # Create empty state
            state = cls(n_total=n_total, M_central=M_central)

            # Load simulation metadata
            state.time = f.attrs['time']
            state.timestep_count = f.attrs['timestep_count']

            # Load physics arrays
            physics = f['physics']
            state.positions[:] = physics['positions'][:]
            state.velocities[:] = physics['velocities'][:]
            state.masses[:] = physics['masses'][:]
            state.accreted[:] = physics['accreted'][:]
            state.proper_times[:] = physics['proper_times'][:]

            # Load metadata arrays
            metadata = f['metadata']
            state.particle_id[:] = metadata['particle_id'][:]
            state.particle_type[:] = metadata['particle_type'][:]
            state.ring_id[:] = metadata['ring_id'][:]
            state.capture_radius[:] = metadata['capture_radius'][:]
            state.initial_speed[:] = metadata['initial_speed'][:]
            state.initial_position[:] = metadata['initial_position'][:]
            state.accreted_by[:] = metadata['accreted_by'][:]

        return state

    def __repr__(self) -> str:
        """String representation of simulation state."""
        lines = [
            f"SimulationState(time={self.time / 1.0e9:.3f} Gyr, "
            f"step={self.timestep_count})",
            f"  Total particles: {self.n_total}",
            f"  Black holes: {self.n_bh}",
            f"  Debris: {self.n_debris}",
            f"  Active: {self.n_active}/{self.n_total} "
            f"({self.n_accreted} accreted)",
        ]

        # Show ring breakdown for black holes
        bh_mask = self.get_black_hole_mask()
        for ring_id in np.unique(self.ring_id[bh_mask]):
            count = np.sum((self.ring_id == ring_id) & bh_mask)
            lines.append(f"    Ring {ring_id}: {count} BHs")

        return "\n".join(lines)
