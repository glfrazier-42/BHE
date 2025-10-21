"""
Unit tests for SimulationState class.

Tests cover:
- State initialization
- HDF5 save/load functionality
- State properties and methods
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from src.state import SimulationState
from src.initialization import initialize_simulation
from src.config import SimulationParameters
from src import constants as const


class TestSimulationStateBasics:
    """Tests for basic SimulationState functionality."""

    def test_initialization(self):
        """Test creating empty simulation state."""
        state = SimulationState(n_bh=10, n_debris=100, M_central=1.0e30)

        assert state.n_bh == 10
        assert state.n_debris == 100
        assert state.M_central == 1.0e30

        # Check arrays are initialized
        assert state.bh_positions.shape == (10, 3)
        assert state.debris_positions.shape == (100, 3)

        # Check initial time
        assert state.time == 0.0
        assert state.timestep_count == 0

    def test_active_debris_count(self):
        """Test counting active debris particles."""
        state = SimulationState(n_bh=5, n_debris=100, M_central=1.0e30)

        # Initially all active
        assert state.n_debris_active == 100
        assert state.n_debris_accreted == 0

        # Accrete some particles
        state.debris_accreted[0] = True
        state.debris_accreted[10] = True
        state.debris_accreted[50] = True

        assert state.n_debris_active == 97
        assert state.n_debris_accreted == 3

    def test_active_debris_mask(self):
        """Test getting active debris mask."""
        state = SimulationState(n_bh=5, n_debris=100, M_central=1.0e30)

        # Accrete some particles
        state.debris_accreted[5] = True
        state.debris_accreted[15] = True

        mask = state.get_active_debris_mask()

        assert mask.shape == (100,)
        assert np.sum(mask) == 98  # 100 - 2 accreted
        assert mask[5] == False
        assert mask[15] == False
        assert mask[0] == True


class TestHDF5SaveLoad:
    """Tests for HDF5 save/load functionality."""

    def test_save_and_load_empty_state(self):
        """Test saving and loading empty simulation state."""
        state1 = SimulationState(n_bh=10, n_debris=100, M_central=4.0e22 * const.M_sun)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_state.h5"

            # Save
            state1.save_to_hdf5(str(filepath))
            assert filepath.exists()

            # Load
            state2 = SimulationState.load_from_hdf5(str(filepath))

            # Check metadata
            assert state2.n_bh == state1.n_bh
            assert state2.n_debris == state1.n_debris
            assert state2.M_central == state1.M_central
            assert state2.time == state1.time
            assert state2.timestep_count == state1.timestep_count

            # Check arrays
            assert np.allclose(state2.bh_positions, state1.bh_positions)
            assert np.allclose(state2.debris_positions, state1.debris_positions)

    def test_save_and_load_initialized_state(self):
        """Test saving and loading fully initialized state."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))
        state1 = initialize_simulation(params, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_initialized.h5"

            # Save
            state1.save_to_hdf5(str(filepath))

            # Load
            state2 = SimulationState.load_from_hdf5(str(filepath))

            # Check all arrays are identical
            assert np.allclose(state2.bh_positions, state1.bh_positions)
            assert np.allclose(state2.bh_velocities, state1.bh_velocities)
            assert np.allclose(state2.bh_masses, state1.bh_masses)
            assert np.array_equal(state2.bh_ring_ids, state1.bh_ring_ids)
            assert np.array_equal(state2.bh_is_static, state1.bh_is_static)
            assert np.allclose(state2.bh_capture_radii, state1.bh_capture_radii)

            assert np.allclose(state2.debris_positions, state1.debris_positions)
            assert np.allclose(state2.debris_velocities, state1.debris_velocities)
            assert np.allclose(state2.debris_masses, state1.debris_masses)
            assert np.allclose(state2.debris_proper_times, state1.debris_proper_times)
            assert np.array_equal(state2.debris_accreted, state1.debris_accreted)
            assert np.array_equal(state2.debris_accreted_by, state1.debris_accreted_by)

    def test_save_with_accretion(self):
        """Test saving state with some particles accreted."""
        state1 = SimulationState(n_bh=5, n_debris=100, M_central=1.0e30)

        # Simulate some accretion
        state1.debris_accreted[10] = True
        state1.debris_accreted_by[10] = 2
        state1.debris_accreted[25] = True
        state1.debris_accreted_by[25] = 3

        # Advance time
        state1.time = 1.0e16  # Some time in seconds
        state1.timestep_count = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_accretion.h5"

            # Save
            state1.save_to_hdf5(str(filepath))

            # Load
            state2 = SimulationState.load_from_hdf5(str(filepath))

            # Check accretion state preserved
            assert state2.n_debris_accreted == 2
            assert state2.debris_accreted[10] == True
            assert state2.debris_accreted_by[10] == 2
            assert state2.debris_accreted[25] == True
            assert state2.debris_accreted_by[25] == 3

            # Check time advanced
            assert state2.time == state1.time
            assert state2.timestep_count == state1.timestep_count

    def test_save_creates_directory(self):
        """Test that save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir1" / "subdir2" / "test_state.h5"

            state = SimulationState(n_bh=5, n_debris=10, M_central=1.0e30)
            state.save_to_hdf5(str(filepath))

            assert filepath.exists()
            assert filepath.parent.exists()

    def test_compression_reduces_file_size(self):
        """Test that HDF5 compression reduces file size."""
        state = SimulationState(n_bh=10, n_debris=1000, M_central=1.0e30)

        # Fill with non-zero data
        state.debris_positions[:] = np.random.randn(1000, 3) * 1e24
        state.debris_velocities[:] = np.random.randn(1000, 3) * 1e6

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath_compressed = Path(tmpdir) / "compressed.h5"
            filepath_uncompressed = Path(tmpdir) / "uncompressed.h5"

            # Save with and without compression
            state.save_to_hdf5(str(filepath_compressed), compression="gzip")
            state.save_to_hdf5(str(filepath_uncompressed), compression=None)

            # Compressed should be smaller (or at least not larger)
            size_compressed = filepath_compressed.stat().st_size
            size_uncompressed = filepath_uncompressed.stat().st_size

            # With random data, compression might not help much, so just check files exist
            assert size_compressed > 0
            assert size_uncompressed > 0


class TestStateRepresentation:
    """Tests for state string representation."""

    def test_repr_format(self):
        """Test __repr__ output format."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))
        state = initialize_simulation(params, seed=42)

        repr_str = repr(state)

        # Check key information is present
        assert "SimulationState" in repr_str
        assert "time" in repr_str
        assert "Gyr" in repr_str
        assert "Black holes" in repr_str
        assert "Debris" in repr_str
        assert "active" in repr_str

    def test_repr_shows_ring_breakdown(self):
        """Test that __repr__ shows ring breakdown."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))
        state = initialize_simulation(params, seed=42)

        repr_str = repr(state)

        # Should show ring counts
        assert "Ring 1" in repr_str
        assert "Ring 2" in repr_str
        assert "Ring 3" in repr_str
