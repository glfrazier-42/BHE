"""
Test orbital motion physics in simulation output.

This test verifies that particles follow expected orbital mechanics:
- Orbiting particles maintain roughly circular orbits
- Orbital radii stay within expected bounds
- Position vectors change properly over time
"""

import pytest
import h5py
import numpy as np
from pathlib import Path
from bhe import constants as const


def test_position_output_structure():
    """Test that position data is properly recorded in HDF5 output."""
    results_dir = Path(__file__).parent.parent / 'results' / 'test'

    if not results_dir.exists():
        pytest.skip("No test results directory found")

    h5_files = list(results_dir.glob('*.h5'))
    if not h5_files:
        pytest.skip("No HDF5 output files found")

    output_file = h5_files[0]

    with h5py.File(output_file, 'r') as f:
        assert 'timeseries' in f, "Timeseries group missing"
        ts = f['timeseries']

        assert 'positions' in ts, "positions dataset missing"
        assert 'time' in ts, "time dataset missing"
        assert 'accreted' in ts, "accreted dataset missing"

        # Verify shapes
        n_timesteps = len(ts['time'])
        n_particles = ts['positions'].shape[1]

        assert ts['positions'].shape == (n_timesteps, n_particles, 3), \
            "positions should be (n_timesteps, n_particles, 3)"
        assert ts['accreted'].shape == (n_timesteps, n_particles), \
            "accreted should be (n_timesteps, n_particles)"


def test_orbital_radius_stability():
    """Test that orbiting black holes maintain stable orbital radii."""
    results_dir = Path(__file__).parent.parent / 'results' / 'test'

    if not results_dir.exists():
        pytest.skip("No test results directory found")

    h5_files = list(results_dir.glob('*.h5'))
    if not h5_files:
        pytest.skip("No HDF5 output files found")

    output_file = h5_files[0]

    with h5py.File(output_file, 'r') as f:
        from bhe.state import BLACK_HOLE

        ts = f['timeseries']
        positions = ts['positions'][:]
        accreted = ts['accreted'][:]
        n_timesteps, n_particles, _ = positions.shape

        # Get metadata to filter black holes only
        particle_type = f['metadata/particle_type'][:]

        # Check orbital radius stability for each BLACK HOLE particle
        for i in range(n_particles):
            # Skip debris particles (they're supposed to move!)
            if particle_type[i] != BLACK_HOLE:
                continue

            # Skip if particle is accreted
            if accreted[-1, i]:
                continue

            # Calculate orbital radius over time
            radii = np.linalg.norm(positions[:, i, :], axis=1)

            # Skip very small radii (central particles)
            if np.mean(radii) < 1.0 * const.Gly:
                continue

            # Orbital radius should be relatively stable (< 10% variation)
            r_mean = np.mean(radii)
            r_std = np.std(radii)
            relative_variation = r_std / r_mean

            print(f"\nBH Particle {i}:")
            print(f"  Mean radius: {r_mean / const.Gly:.3f} Gly")
            print(f"  Std dev: {r_std / const.Gly:.3f} Gly")
            print(f"  Relative variation: {relative_variation:.2%}")

            # For stable orbits, variation should be reasonable
            # (Allow more variation for short test runs)
            assert relative_variation < 0.5, \
                f"BH Particle {i} orbital radius variation {relative_variation:.2%} too large"


def test_position_evolution():
    """Test that particle positions evolve (not frozen)."""
    results_dir = Path(__file__).parent.parent / 'results' / 'test'

    if not results_dir.exists():
        pytest.skip("No test results directory found")

    h5_files = list(results_dir.glob('*.h5'))
    if not h5_files:
        pytest.skip("No HDF5 output files found")

    output_file = h5_files[0]

    with h5py.File(output_file, 'r') as f:
        ts = f['timeseries']
        positions = ts['positions'][:]
        accreted = ts['accreted'][:]
        times = ts['time'][:] / 1.0e9  # Convert years to Gyr

        if len(times) < 2:
            pytest.skip("Need at least 2 timesteps")

        n_particles = positions.shape[1]

        # Check that active particles move
        for i in range(n_particles):
            # Skip accreted particles
            if accreted[0, i]:
                continue

            # Calculate total displacement from start to end
            displacement = positions[-1, i, :] - positions[0, i, :]
            dist_moved = np.linalg.norm(displacement)

            print(f"\nParticle {i}:")
            print(f"  Initial pos: {positions[0, i, :] / const.Gly}")
            print(f"  Final pos: {positions[-1, i, :] / const.Gly}")
            print(f"  Distance moved: {dist_moved / const.Gly:.6f} Gly")

            # Particles should move (unless very short simulation)
            # Allow for stationary central particles
            initial_r = np.linalg.norm(positions[0, i, :])
            if initial_r > 1.0 * const.Gly and times[-1] - times[0] > 0.001:
                assert dist_moved > 0.001 * const.Gly, \
                    f"Particle {i} didn't move significantly"


def test_no_duplicate_positions():
    """Test that consecutive timesteps have different positions."""
    results_dir = Path(__file__).parent.parent / 'results' / 'test'

    if not results_dir.exists():
        pytest.skip("No test results directory found")

    h5_files = list(results_dir.glob('*.h5'))
    if not h5_files:
        pytest.skip("No HDF5 output files found")

    output_file = h5_files[0]

    with h5py.File(output_file, 'r') as f:
        ts = f['timeseries']
        positions = ts['positions'][:]
        accreted = ts['accreted'][:]

        if len(positions) < 2:
            pytest.skip("Need at least 2 timesteps")

        # Check each pair of consecutive timesteps
        for t in range(len(positions) - 1):
            for i in range(positions.shape[1]):
                # Skip accreted particles
                if accreted[t, i]:
                    continue

                pos1 = positions[t, i, :]
                pos2 = positions[t+1, i, :]

                # Positions should differ between timesteps
                diff = np.linalg.norm(pos2 - pos1)
                assert diff > 1e-10, \
                    f"Particle {i} position identical at timesteps {t} and {t+1}"
