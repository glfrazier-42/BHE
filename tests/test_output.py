"""
Tests for data recording and checkpointing functionality.
"""

import numpy as np
import pytest
import h5py
import tempfile
import os
from pathlib import Path

from bhe.config import SimulationParameters
from bhe.state import SimulationState
from bhe.output import (
    SimulationRecorder,
    calculate_total_energy,
    calculate_total_momentum,
    load_checkpoint
)
from bhe import constants as const


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


@pytest.fixture(scope="function")
def simple_state():
    """Create simple simulation state for testing (unified particle system, natural units)."""
    n_total = 5  # 2 BHs + 3 debris
    M_central = 4e22  # M_sun (natural units)

    from bhe.state import BLACK_HOLE, DEBRIS

    state = SimulationState(n_total=n_total, M_central=M_central)

    # Manually set particle types (first 2 are BHs, last 3 are debris)
    state.particle_type[0] = BLACK_HOLE
    state.particle_type[1] = BLACK_HOLE
    state.particle_type[2] = DEBRIS
    state.particle_type[3] = DEBRIS
    state.particle_type[4] = DEBRIS

    # Fill in positions (natural units: ly)
    state.positions[0] = [0.0, 0.0, 0.0]  # BH
    state.positions[1] = [2.0e9, 0.0, 0.0]  # BH at 2 Gly
    state.positions[2] = [1.0e9, 0.0, 0.0]  # Debris at 1 Gly
    state.positions[3] = [0.0, 1.0e9, 0.0]  # Debris at 1 Gly
    state.positions[4] = [0.0, 0.0, 1.0e9]  # Debris at 1 Gly

    # Fill in velocities (natural units: fraction of c)
    state.velocities[0] = [0.0, 0.0, 0.0]
    state.velocities[1] = [0.0, 0.0002, 0.0]  # ~200 km/s
    state.velocities[2] = [0.0001, 0.0, 0.0]  # ~100 km/s
    state.velocities[3] = [0.0, 0.0001, 0.0]
    state.velocities[4] = [0.0, 0.0, 0.0001]

    # Fill in masses (natural units: M_sun)
    state.masses[0] = 4e22  # Large BH
    state.masses[1] = 1e10  # Large BH
    state.masses[2] = 1.0
    state.masses[3] = 2.0
    state.masses[4] = 3.0

    # Fill in metadata
    state.ring_id[0] = 0
    state.ring_id[1] = 0
    state.ring_id[2] = -1
    state.ring_id[3] = -1
    state.ring_id[4] = -1

    state.capture_radius[0] = 0.0
    state.capture_radius[1] = 1.0e9  # 1 Gly in natural units

    return state


class TestEnergyCalculation:
    """Test energy conservation calculation."""

    def test_kinetic_energy_non_relativistic(self, simple_state):
        """Test kinetic energy calculation (unified particle system)."""
        energy = calculate_total_energy(
            simple_state.positions,
            simple_state.velocities,
            simple_state.masses,
            simple_state.accreted
        )

        # Energy should be finite and non-zero
        assert np.isfinite(energy)
        assert energy != 0.0

    def test_energy_with_accreted_particles(self, simple_state):
        """Test that accreted particles don't contribute to energy."""
        # Mark one particle as accreted
        simple_state.accreted[0] = True

        energy_with_accreted = calculate_total_energy(
            simple_state.positions,
            simple_state.velocities,
            simple_state.masses,
            simple_state.accreted
        )

        # Should still be finite
        assert np.isfinite(energy_with_accreted)


class TestMomentumCalculation:
    """Test momentum conservation calculation."""

    def test_momentum_calculation(self, simple_state):
        """Test momentum calculation returns 3D vector (unified particle system)."""
        momentum = calculate_total_momentum(
            simple_state.velocities,
            simple_state.masses,
            simple_state.accreted
        )

        # Should be 3D vector
        assert momentum.shape == (3,)
        assert np.all(np.isfinite(momentum))

    def test_momentum_excludes_accreted(self, simple_state):
        """Test that accreted particles don't contribute to momentum."""
        # Calculate momentum before accretion
        momentum_before = calculate_total_momentum(
            simple_state.velocities,
            simple_state.masses,
            simple_state.accreted
        )

        # Mark particle with non-zero velocity as accreted (particle 1 has velocity [0, 0.0002, 0])
        simple_state.accreted[1] = True

        # Calculate momentum after accretion
        momentum_after = calculate_total_momentum(
            simple_state.velocities,
            simple_state.masses,
            simple_state.accreted
        )

        # Momentum should change (one particle with nonzero velocity excluded)
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

                    # Check timeseries datasets (unified particle system)
                    assert 'time' in f['timeseries']
                    assert 'timestep' in f['timeseries']
                    assert 'positions' in f['timeseries']
                    assert 'velocities' in f['timeseries']
                    assert 'masses' in f['timeseries']
                    assert 'accreted' in f['timeseries']
                    assert 'proper_times' in f['timeseries']

                    # Check metadata
                    assert 'metadata' in f
                    assert 'particle_type' in f['metadata']
                    assert 'ring_id' in f['metadata']

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

            # Check checkpoint was saved as separate file
            checkpoint_file = os.path.join(tmpdir, "test_checkpoint.h5")
            assert os.path.exists(checkpoint_file)

            # Check checkpoint structure (SimulationState HDF5 format)
            with h5py.File(checkpoint_file, 'r') as f:
                assert 'time' in f.attrs
                assert 'timestep_count' in f.attrs
                assert 'physics' in f
                assert 'positions' in f['physics']
                assert 'velocities' in f['physics']
                assert 'masses' in f['physics']
                assert 'metadata' in f


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""

    def test_load_checkpoint(self, simple_params, simple_state):
        """Test loading a checkpoint restores state correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            # Save original state (unified particle system)
            original_time = simple_state.time
            original_positions = simple_state.positions.copy()

            # Create recorder and save checkpoint
            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                recorder.save_checkpoint(simple_state, "test_checkpoint")

            # Modify state
            simple_state.time += 1000.0
            simple_state.positions += 1e20

            # Load checkpoint from separate file
            checkpoint_file = os.path.join(tmpdir, "test_checkpoint.h5")
            loaded_state = load_checkpoint(checkpoint_file)

            # Check state was restored
            assert loaded_state.time == original_time
            assert np.allclose(loaded_state.positions, original_positions)

    def test_load_nonexistent_checkpoint(self, simple_params, simple_state):
        """Test that loading nonexistent checkpoint raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            # Create file but don't save checkpoint
            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                pass

            # Try to load nonexistent checkpoint file
            nonexistent_file = os.path.join(tmpdir, "nonexistent_checkpoint.h5")
            with pytest.raises(FileNotFoundError):
                load_checkpoint(nonexistent_file)


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

                # Energy error at idx=0 is NaN (no baseline yet), idx=1 establishes baseline
                assert np.isnan(f['conservation/energy_error'][0])
                assert np.isnan(f['conservation/momentum_error'][0])
                # At idx=1, baseline is established, so error should be 0.0
                assert f['conservation/energy_error'][1] == 0.0
                assert f['conservation/momentum_error'][1] == 0.0


class TestDataCompression:
    """Test HDF5 compression."""

    def test_datasets_use_compression(self, simple_params, simple_state):
        """Test that datasets use gzip compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sim.h5")

            with SimulationRecorder(filepath, simple_params, simple_state) as recorder:
                recorder.record_timestep(simple_state, check_conservation=True)

            # Check compression settings (unified particle system)
            with h5py.File(filepath, 'r') as f:
                # Check a few key datasets
                assert f['timeseries/positions'].compression == 'gzip'
                assert f['timeseries/velocities'].compression == 'gzip'
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
