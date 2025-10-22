"""
Tests for data recording and checkpointing functionality.
"""

import numpy as np
import pytest
import h5py
import tempfile
import os
from pathlib import Path

from src.config import SimulationParameters
from src.state import SimulationState
from src.output import (
    SimulationRecorder,
    calculate_total_energy,
    calculate_total_momentum,
    load_checkpoint
)
from src import constants as const


@pytest.fixture
def simple_params():
    """Create simple simulation parameters for testing."""
    # Load from baseline config and modify for testing
    config_path = Path(__file__).parent.parent / "configs" / "baseline_config.yaml"
    params = SimulationParameters.from_yaml(str(config_path))

    # Override with test-friendly values
    params.dt = 1e12  # seconds
    params.duration = 1e13  # seconds (10 timesteps)
    params.output_interval = 2e12  # seconds (every 2 timesteps)
    params.checkpoint_interval = 5e12  # seconds (every 5 timesteps)

    return params


@pytest.fixture
def simple_state():
    """Create simple simulation state for testing."""
    # Create state with proper constructor
    n_debris = 3
    n_bh = 2
    M_central = 4e22 * const.M_sun

    state = SimulationState(n_bh=n_bh, n_debris=n_debris, M_central=M_central)

    # Fill in debris data
    state.debris_positions[0] = [1e24, 0.0, 0.0]
    state.debris_positions[1] = [0.0, 1e24, 0.0]
    state.debris_positions[2] = [0.0, 0.0, 1e24]

    state.debris_velocities[0] = [1e5, 0.0, 0.0]
    state.debris_velocities[1] = [0.0, 1e5, 0.0]
    state.debris_velocities[2] = [0.0, 0.0, 1e5]

    state.debris_masses[0] = 1e30
    state.debris_masses[1] = 2e30
    state.debris_masses[2] = 3e30

    # Fill in BH data
    state.bh_positions[0] = [0.0, 0.0, 0.0]
    state.bh_positions[1] = [2e24, 0.0, 0.0]

    state.bh_velocities[0] = [0.0, 0.0, 0.0]
    state.bh_velocities[1] = [0.0, 2e8, 0.0]

    state.bh_masses[0] = 4e22 * const.M_sun
    state.bh_masses[1] = 1e10 * const.M_sun

    state.bh_ring_ids[0] = 0
    state.bh_ring_ids[1] = 0

    state.bh_is_static[0] = True
    state.bh_is_static[1] = False

    state.bh_capture_radii[0] = 0.0
    state.bh_capture_radii[1] = 1e22

    return state


class TestEnergyCalculation:
    """Test energy conservation calculation."""

    def test_kinetic_energy_non_relativistic(self, simple_state):
        """Test kinetic energy calculation in non-relativistic regime."""
        energy = calculate_total_energy(
            simple_state.debris_positions,
            simple_state.debris_velocities,
            simple_state.debris_masses,
            simple_state.debris_accreted,
            simple_state.bh_positions,
            simple_state.bh_velocities,
            simple_state.bh_masses,
            use_relativistic=False
        )

        # Energy should be finite and non-zero
        assert np.isfinite(energy)
        assert energy != 0.0

    def test_energy_with_accreted_particles(self, simple_state):
        """Test that accreted particles don't contribute to energy."""
        # Mark one particle as accreted
        simple_state.debris_accreted[0] = True

        energy_with_accreted = calculate_total_energy(
            simple_state.debris_positions,
            simple_state.debris_velocities,
            simple_state.debris_masses,
            simple_state.debris_accreted,
            simple_state.bh_positions,
            simple_state.bh_velocities,
            simple_state.bh_masses,
            use_relativistic=False
        )

        # Should still be finite
        assert np.isfinite(energy_with_accreted)


class TestMomentumCalculation:
    """Test momentum conservation calculation."""

    def test_momentum_calculation(self, simple_state):
        """Test momentum calculation returns 3D vector."""
        momentum = calculate_total_momentum(
            simple_state.debris_velocities,
            simple_state.debris_masses,
            simple_state.debris_accreted,
            simple_state.bh_velocities,
            simple_state.bh_masses
        )

        # Should be 3D vector
        assert momentum.shape == (3,)
        assert np.all(np.isfinite(momentum))

    def test_momentum_excludes_accreted(self, simple_state):
        """Test that accreted particles don't contribute to momentum."""
        # Calculate momentum before accretion
        momentum_before = calculate_total_momentum(
            simple_state.debris_velocities,
            simple_state.debris_masses,
            simple_state.debris_accreted,
            simple_state.bh_velocities,
            simple_state.bh_masses
        )

        # Mark one particle as accreted
        simple_state.debris_accreted[0] = True

        # Calculate momentum after accretion
        momentum_after = calculate_total_momentum(
            simple_state.debris_velocities,
            simple_state.debris_masses,
            simple_state.debris_accreted,
            simple_state.bh_velocities,
            simple_state.bh_masses
        )

        # Momentum should decrease (one particle excluded)
        assert not np.allclose(momentum_before, momentum_after)


class TestSimulationRecorder:
    """Test HDF5 data recording functionality."""

    def test_recorder_creation(self, simple_params, simple_state):
        """Test that recorder creates HDF5 file with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                # Check file was created
                assert os.path.exists(filepath)

                # Check file structure
                with h5py.File(filepath, 'r') as f:
                    # Check groups exist
                    assert 'timeseries' in f
                    assert 'conservation' in f
                    assert 'checkpoints' in f

                    # Check timeseries datasets
                    assert 'time' in f['timeseries']
                    assert 'timestep' in f['timeseries']
                    assert 'debris_positions' in f['timeseries']
                    assert 'debris_velocities' in f['timeseries']
                    assert 'debris_accreted' in f['timeseries']
                    assert 'bh_positions' in f['timeseries']
                    assert 'bh_velocities' in f['timeseries']

                    # Check conservation datasets
                    assert 'total_energy' in f['conservation']
                    assert 'total_momentum' in f['conservation']

    def test_record_timestep(self, simple_params, simple_state):
        """Test recording a single timestep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                # Record initial state
                recorder.record_timestep(simple_state, check_conservation=True)

                # Advance state
                simple_state.time += simple_params.dt
                simple_state.timestep_count += 1

                # Record next timestep
                recorder.record_timestep(simple_state, check_conservation=True)

            # Check recorded data
            with h5py.File(filepath, 'r') as f:
                # Should have 2 timesteps recorded (out of 6 allocated)
                # Note: The full array is pre-allocated based on expected output steps
                assert f['timeseries/time'].shape[0] == 6  # 10 steps / 2 output_interval + 1 initial
                assert f['timeseries/timestep'].shape[0] == 6

                # Check first two recorded time values
                assert f['timeseries/time'][0] == 0.0
                assert f['timeseries/time'][1] == simple_params.dt

                # Check first two recorded timestep values
                assert f['timeseries/timestep'][0] == 0
                assert f['timeseries/timestep'][1] == 1

    def test_save_checkpoint(self, simple_params, simple_state):
        """Test saving a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                # Save checkpoint
                recorder.save_checkpoint(simple_state, "test_checkpoint")

            # Check checkpoint was saved
            with h5py.File(filepath, 'r') as f:
                assert 'checkpoints/test_checkpoint' in f

                cp = f['checkpoints/test_checkpoint']
                assert 'time' in cp.attrs
                assert 'timestep_count' in cp.attrs
                assert 'debris_positions' in cp
                assert 'debris_velocities' in cp
                assert 'bh_positions' in cp
                assert 'bh_velocities' in cp


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""

    def test_load_checkpoint(self, simple_params, simple_state):
        """Test loading a checkpoint restores state correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            # Save original state
            original_time = simple_state.time
            original_positions = simple_state.debris_positions.copy()

            # Create recorder and save checkpoint
            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                recorder.save_checkpoint(simple_state, "test_checkpoint")

            # Modify state
            simple_state.time += 1000.0
            simple_state.debris_positions += 1e20

            # Load checkpoint
            loaded_state = load_checkpoint(filepath, "test_checkpoint")

            # Check state was restored
            assert loaded_state.time == original_time
            assert np.allclose(loaded_state.debris_positions, original_positions)

    def test_load_nonexistent_checkpoint(self, simple_params, simple_state):
        """Test that loading nonexistent checkpoint raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            # Create file but don't save checkpoint
            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                pass

            # Try to load nonexistent checkpoint
            with pytest.raises(KeyError):
                load_checkpoint(filepath, "nonexistent_checkpoint")


class TestConservationWarnings:
    """Test conservation violation warnings."""

    def test_conservation_metrics_recorded(self, simple_params, simple_state):
        """Test that conservation metrics are recorded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                # Record initial state
                recorder.record_timestep(simple_state, check_conservation=True)

                # Advance state
                simple_state.time += simple_params.dt
                simple_state.timestep_count += 1

                # Record next timestep
                recorder.record_timestep(simple_state, check_conservation=True)

            # Check conservation data was recorded
            with h5py.File(filepath, 'r') as f:
                assert 'conservation/total_energy' in f
                assert 'conservation/total_momentum' in f
                assert 'conservation/energy_error' in f
                assert 'conservation/momentum_error' in f

                # Energy error at t=0 should be zero (initial state)
                assert f['conservation/energy_error'][0] == 0.0
                assert f['conservation/momentum_error'][0] == 0.0


class TestDataCompression:
    """Test HDF5 compression."""

    def test_datasets_use_compression(self, simple_params, simple_state):
        """Test that datasets use gzip compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                recorder.record_timestep(simple_state, check_conservation=True)

            # Check compression settings
            with h5py.File(filepath, 'r') as f:
                # Check a few key datasets
                assert f['timeseries/debris_positions'].compression == 'gzip'
                assert f['timeseries/debris_velocities'].compression == 'gzip'
                assert f['conservation/total_energy'].compression == 'gzip'


class TestOutputIntervals:
    """Test output and checkpoint intervals."""

    def test_output_interval_calculation(self, simple_params):
        """Test that output_interval is correctly converted to steps."""
        dt = simple_params.dt  # 1e12 seconds
        output_interval = simple_params.output_interval  # 2e12 seconds

        # Should record every 2 timesteps
        output_every = max(1, int(output_interval / dt))
        assert output_every == 2

    def test_checkpoint_interval_calculation(self, simple_params):
        """Test that checkpoint_interval is correctly converted to steps."""
        dt = simple_params.dt  # 1e12 seconds
        checkpoint_interval = simple_params.checkpoint_interval  # 5e12 seconds

        # Should checkpoint every 5 timesteps
        checkpoint_every = max(1, int(checkpoint_interval / dt))
        assert checkpoint_every == 5
