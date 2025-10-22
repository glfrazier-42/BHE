"""
Tests for analysis module.
"""

import pytest
import numpy as np
import h5py
import tempfile
import os
from pathlib import Path

from src.analysis import (
    calculate_redshift,
    calculate_redshift_array,
    analyze_simulation,
    calculate_escape_fraction_vs_time,
    get_final_debris_state,
    get_ring0_trajectories
)
from src import constants as const


class TestRedshiftCalculation:
    """Test redshift calculation functions."""

    def test_zero_velocity_zero_redshift(self):
        """Redshift should be zero for stationary particle."""
        velocity = np.array([0.0, 0.0, 0.0])
        z = calculate_redshift(velocity)
        assert abs(z) < 1e-10

    def test_low_velocity_classical_limit(self):
        """For v << c, z ≈ v/c (classical Doppler)."""
        v = 1.0e5  # 100 km/s << c
        velocity = np.array([v, 0.0, 0.0])
        z = calculate_redshift(velocity)

        # Classical approximation
        z_classical = v / const.c

        # Should be very close at low velocities
        assert abs(z - z_classical) / z_classical < 0.01  # Within 1%

    def test_half_speed_of_light(self):
        """Test redshift at v = 0.5c."""
        velocity = np.array([0.5 * const.c, 0.0, 0.0])
        z = calculate_redshift(velocity)

        # Expected: z = sqrt((1+0.5)/(1-0.5)) - 1 = sqrt(3) - 1 ≈ 0.732
        expected = np.sqrt(1.5 / 0.5) - 1.0
        assert abs(z - expected) / expected < 1e-6

    def test_high_velocity(self):
        """Test redshift at v = 0.9c."""
        velocity = np.array([0.9 * const.c, 0.0, 0.0])
        z = calculate_redshift(velocity)

        # Expected: z = sqrt((1.9)/(0.1)) - 1 ≈ 3.36
        expected = np.sqrt(1.9 / 0.1) - 1.0
        assert abs(z - expected) / expected < 1e-6

    def test_3d_velocity(self):
        """Test redshift with 3D velocity vector."""
        # v = (0.3c, 0.4c, 0.0) -> |v| = 0.5c
        velocity = np.array([0.3 * const.c, 0.4 * const.c, 0.0])
        z = calculate_redshift(velocity)

        # Expected for |v| = 0.5c
        expected = np.sqrt(1.5 / 0.5) - 1.0
        assert abs(z - expected) / expected < 1e-6

    def test_near_light_speed_clamped(self):
        """Test that near-c velocities are handled safely."""
        velocity = np.array([0.99999 * const.c, 0.0, 0.0])
        z = calculate_redshift(velocity)

        # Should be large but finite
        assert z > 100.0
        assert z < 1e10  # Should be clamped, not infinity

    def test_redshift_array(self):
        """Test batch calculation of redshifts."""
        velocities = np.array([
            [0.0, 0.0, 0.0],
            [0.5 * const.c, 0.0, 0.0],
            [0.9 * const.c, 0.0, 0.0]
        ])

        redshifts = calculate_redshift_array(velocities)

        assert len(redshifts) == 3
        assert abs(redshifts[0]) < 1e-10  # v=0
        assert abs(redshifts[1] - (np.sqrt(1.5/0.5) - 1.0)) < 1e-6  # v=0.5c
        assert abs(redshifts[2] - (np.sqrt(1.9/0.1) - 1.0)) < 1e-6  # v=0.9c


class TestAnalyzeSimulation:
    """Test simulation analysis functions."""

    @pytest.fixture
    def mock_hdf5_file(self):
        """Create a mock HDF5 file with simulation data."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "test_sim.h5")

        with h5py.File(filepath, 'w') as f:
            # Create timeseries group
            ts = f.create_group('timeseries')

            # Create simple test data (3 timesteps, 10 particles)
            n_steps = 3
            n_debris = 10

            times = np.array([0.0, 1e12, 2e12])  # seconds
            ts.create_dataset('time', data=times)

            # Debris positions - some far away, some close
            positions = np.zeros((n_steps, n_debris, 3))
            # Last timestep: half beyond 100 Gly, half close
            for i in range(5):
                positions[-1, i] = [(i + 1) * 150e9 * const.Gly_to_m / 150, 0, 0]  # Beyond 100 Gly
            for i in range(5, 10):
                positions[-1, i] = [(i + 1) * 10e9 * const.Gly_to_m / 10, 0, 0]  # Within 100 Gly
            ts.create_dataset('debris_positions', data=positions)

            # Velocities
            velocities = np.zeros((n_steps, n_debris, 3))
            velocities[-1, :, 0] = 0.5 * const.c  # All moving at 0.5c in x direction
            ts.create_dataset('debris_velocities', data=velocities)

            # Proper times
            proper_times = np.ones((n_steps, n_debris)) * 1e12  # 1e12 seconds
            ts.create_dataset('debris_proper_times', data=proper_times)

            # Accretion flags - mark 3 as accreted
            accreted = np.zeros((n_steps, n_debris), dtype=bool)
            accreted[-1, 0:3] = True
            ts.create_dataset('debris_accreted', data=accreted)

            # Create conservation group
            cons = f.create_group('conservation')
            energy_errors = np.array([0.0, 0.005, 0.01])
            momentum_errors = np.array([0.0, 0.003, 0.008])
            cons.create_dataset('energy_error', data=energy_errors)
            cons.create_dataset('momentum_error', data=momentum_errors)

        yield filepath

        # Cleanup
        os.remove(filepath)
        os.rmdir(tmpdir)

    def test_analyze_simulation(self, mock_hdf5_file):
        """Test full simulation analysis."""
        results = analyze_simulation(mock_hdf5_file)

        # Check result structure
        assert 'n_debris_total' in results
        assert 'n_debris_accreted' in results
        assert 'n_debris_escaped' in results
        assert 'escape_fraction' in results
        assert 'redshift_mean' in results
        assert 'energy_conservation_error' in results

        # Check values
        assert results['n_debris_total'] == 10
        assert results['n_debris_accreted'] == 3

        # Energy conservation should match last timestep
        assert abs(results['energy_conservation_error'] - 0.01) < 1e-10

    def test_get_final_debris_state(self, mock_hdf5_file):
        """Test extraction of final debris state."""
        state = get_final_debris_state(mock_hdf5_file)

        # Check structure
        assert 'positions' in state
        assert 'velocities' in state
        assert 'proper_times' in state
        assert 'accreted' in state
        assert 'distances' in state
        assert 'redshifts' in state
        assert 'time' in state

        # Check shapes
        assert state['positions'].shape == (10, 3)
        assert state['velocities'].shape == (10, 3)
        assert len(state['accreted']) == 10
        assert len(state['distances']) == 10
        assert len(state['redshifts']) == 10

        # Check accretion status
        assert np.sum(state['accreted']) == 3


class TestEscapeFraction:
    """Test escape fraction calculation."""

    @pytest.fixture
    def escape_test_file(self):
        """Create HDF5 file for escape fraction testing."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "escape_test.h5")

        with h5py.File(filepath, 'w') as f:
            ts = f.create_group('timeseries')

            # Create 5 timesteps, 4 particles
            n_steps = 5
            n_debris = 4

            times = np.linspace(0, 4e12, n_steps)
            ts.create_dataset('time', data=times)

            # Particles gradually move outward
            positions = np.zeros((n_steps, n_debris, 3))
            for i in range(n_steps):
                # Each particle moves outward at different rate
                for j in range(n_debris):
                    distance = (j + 1) * 30e9 * const.Gly_to_m * (i + 1) / n_steps
                    positions[i, j] = [distance, 0, 0]

            ts.create_dataset('debris_positions', data=positions)

            # No accretion
            accreted = np.zeros((n_steps, n_debris), dtype=bool)
            ts.create_dataset('debris_accreted', data=accreted)

        yield filepath

        os.remove(filepath)
        os.rmdir(tmpdir)

    def test_escape_fraction_increases(self, escape_test_file):
        """Test that escape fraction increases over time."""
        times, fractions = calculate_escape_fraction_vs_time(
            escape_test_file, distance_threshold=100.0
        )

        # Should have 5 timesteps
        assert len(times) == 5
        assert len(fractions) == 5

        # Escape fraction should be monotonically increasing
        # (particles move outward and cross 100 Gly threshold)
        for i in range(len(fractions) - 1):
            assert fractions[i] <= fractions[i + 1]

        # All fractions should be between 0 and 1
        assert np.all(fractions >= 0.0)
        assert np.all(fractions <= 1.0)


class TestRing0Trajectories:
    """Test Ring 0 trajectory extraction."""

    @pytest.fixture
    def ring0_test_file(self):
        """Create HDF5 file with Ring 0 data."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "ring0_test.h5")

        with h5py.File(filepath, 'w') as f:
            # Add config group
            f.create_group('config')

            ts = f.create_group('timeseries')

            # Create 3 timesteps, 2 BHs
            n_steps = 3
            n_bh = 2

            times = np.array([0.0, 1e12, 2e12])
            ts.create_dataset('time', data=times)

            # BH positions in circular orbits
            positions = np.zeros((n_steps, n_bh, 3))
            radius = 3e9 * const.Gly_to_m
            for i in range(n_steps):
                angle = i * np.pi / 4
                positions[i, 0] = [radius * np.cos(angle), radius * np.sin(angle), 0]
                positions[i, 1] = [-radius * np.cos(angle), -radius * np.sin(angle), 0]

            ts.create_dataset('bh_positions', data=positions)

            # BH velocities
            velocities = np.zeros((n_steps, n_bh, 3))
            ts.create_dataset('bh_velocities', data=velocities)

        yield filepath

        os.remove(filepath)
        os.rmdir(tmpdir)

    def test_get_ring0_trajectories(self, ring0_test_file):
        """Test extraction of Ring 0 trajectories."""
        trajectories = get_ring0_trajectories(ring0_test_file)

        assert trajectories is not None
        assert 'times' in trajectories
        assert 'positions' in trajectories
        assert 'velocities' in trajectories
        assert 'n_bh' in trajectories

        # Check shapes
        assert len(trajectories['times']) == 3
        assert trajectories['positions'].shape == (3, 2, 3)
        assert trajectories['n_bh'] == 2

    def test_no_bh_data_returns_none(self):
        """Test that missing BH data returns None."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "no_bh.h5")

        with h5py.File(filepath, 'w') as f:
            # Create minimal structure without BH data
            ts = f.create_group('timeseries')
            ts.create_dataset('time', data=np.array([0.0]))

        trajectories = get_ring0_trajectories(filepath)

        # Should return None when BH data missing
        assert trajectories is None

        os.remove(filepath)
        os.rmdir(tmpdir)
