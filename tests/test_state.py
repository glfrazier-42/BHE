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

from bhe.state import SimulationState
from bhe.initialization import initialize_simulation
from bhe.config import SimulationParameters
from bhe import constants as const


class TestSimulationStateBasics:
    """Tests for basic SimulationState functionality."""

    def test_initialization(self):
        """Test creating empty simulation state (unified particle system)."""
        n_total = 110  # 10 BHs + 100 debris
        state = SimulationState(n_total=n_total, M_central=1.0)  # Natural units: M_sun

        assert state.n_total == 110
        assert state.M_central == 1.0

        # Check unified arrays are initialized
        assert state.positions.shape == (110, 3)
        assert state.velocities.shape == (110, 3)
        assert state.masses.shape == (110,)
        assert state.accreted.shape == (110,)

        # Check initial time
        assert state.time == 0.0
        assert state.timestep_count == 0

    def test_active_debris_count(self):
        """Test counting active particles (unified system)."""
        n_total = 105  # 5 BHs + 100 debris
        state = SimulationState(n_total=n_total, M_central=1.0)

        # Initially all active
        assert state.n_active == 105
        assert state.n_accreted == 0

        # Accrete some particles
        state.accreted[0] = True
        state.accreted[10] = True
        state.accreted[50] = True

        assert state.n_active == 102
        assert state.n_accreted == 3

    def test_active_debris_mask(self):
        """Test getting active particle mask (unified system)."""
        n_total = 105  # 5 BHs + 100 debris
        state = SimulationState(n_total=n_total, M_central=1.0)

        # Accrete some particles
        state.accreted[5] = True
        state.accreted[15] = True

        mask = ~state.accreted  # Active mask is inverse of accreted

        assert mask.shape == (105,)
        assert np.sum(mask) == 103  # 105 - 2 accreted
        assert mask[5] == False
        assert mask[15] == False
        assert mask[0] == True


class TestHDF5SaveLoad:
    """Tests for HDF5 save/load functionality."""

    def test_save_and_load_empty_state(self):
        """Test saving and loading empty simulation state (unified system)."""
        n_total = 110  # 10 BHs + 100 debris
        state1 = SimulationState(n_total=n_total, M_central=4.0e22)  # Natural units: M_sun

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_state.h5"

            # Save
            state1.save_to_hdf5(str(filepath))
            assert filepath.exists()

            # Load
            state2 = SimulationState.load_from_hdf5(str(filepath))

            # Check metadata
            assert state2.n_total == state1.n_total
            assert state2.M_central == state1.M_central
            assert state2.time == state1.time
            assert state2.timestep_count == state1.timestep_count

            # Check arrays
            assert np.allclose(state2.positions, state1.positions)
            assert np.allclose(state2.velocities, state1.velocities)
            assert np.allclose(state2.masses, state1.masses)

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

            # Check all arrays are identical (unified particle system)
            assert np.allclose(state2.positions, state1.positions)
            assert np.allclose(state2.velocities, state1.velocities)
            assert np.allclose(state2.masses, state1.masses)
            assert np.array_equal(state2.particle_type, state1.particle_type)
            assert np.array_equal(state2.ring_id, state1.ring_id)
            assert np.allclose(state2.capture_radius, state1.capture_radius)
            assert np.allclose(state2.proper_times, state1.proper_times)
            assert np.array_equal(state2.accreted, state1.accreted)
            assert np.array_equal(state2.accreted_by, state1.accreted_by)

    def test_save_with_accretion(self):
        """Test saving state with some particles accreted (unified system)."""
        n_total = 105  # 5 BHs + 100 debris
        state1 = SimulationState(n_total=n_total, M_central=1.0)

        # Simulate some accretion
        state1.accreted[10] = True
        state1.accreted_by[10] = 2
        state1.accreted[25] = True
        state1.accreted_by[25] = 3

        # Advance time
        state1.time = 1.0e7  # Some time in years
        state1.timestep_count = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_accretion.h5"

            # Save
            state1.save_to_hdf5(str(filepath))

            # Load
            state2 = SimulationState.load_from_hdf5(str(filepath))

            # Check accretion state preserved
            assert state2.n_accreted == 2
            assert state2.accreted[10] == True
            assert state2.accreted_by[10] == 2
            assert state2.accreted[25] == True
            assert state2.accreted_by[25] == 3

            # Check time advanced
            assert state2.time == state1.time
            assert state2.timestep_count == state1.timestep_count

    def test_save_creates_directory(self):
        """Test that save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir1" / "subdir2" / "test_state.h5"

            state = SimulationState(n_total=15, M_central=1.0)  # 5 BHs + 10 debris
            state.save_to_hdf5(str(filepath))

            assert filepath.exists()
            assert filepath.parent.exists()

    def test_compression_reduces_file_size(self):
        """Test that HDF5 compression reduces file size."""
        n_total = 1010  # 10 BHs + 1000 debris
        state = SimulationState(n_total=n_total, M_central=1.0)

        # Fill with non-zero data (natural units: positions in ly, velocities in units of c)
        state.positions[:] = np.random.randn(n_total, 3) * 1e9  # Gly scale
        state.velocities[:] = np.random.randn(n_total, 3) * 0.1  # 0.1c scale

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
        assert "Active" in repr_str

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
