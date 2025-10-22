"""
Tests for visualization module.
"""

import pytest
import numpy as np
import h5py
import tempfile
import os
from pathlib import Path

from src.visualization import (
    plot_redshift_vs_distance,
    plot_proper_time_vs_redshift,
    plot_ring0_trajectories_3d,
    plot_escape_fraction_vs_time,
    generate_summary_report
)
from src import constants as const


@pytest.fixture
def simple_simulation_file():
    """Create a simple HDF5 simulation file for visualization testing."""
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "viz_test.h5")

    with h5py.File(filepath, 'w') as f:
        # Config
        f.create_group('config')

        # Timeseries data
        ts = f.create_group('timeseries')

        n_steps = 5
        n_debris = 20

        # Times
        times = np.linspace(0, 4e12, n_steps)
        ts.create_dataset('time', data=times)
        ts.create_dataset('timestep', data=np.arange(n_steps))

        # Debris positions - spread out in space
        positions = np.zeros((n_steps, n_debris, 3))
        for i in range(n_steps):
            for j in range(n_debris):
                # Particles move outward over time
                r = (j + 1) * 10e9 * const.Gly_to_m * (i + 1) / n_steps
                theta = j * 2 * np.pi / n_debris
                positions[i, j] = [
                    r * np.cos(theta),
                    r * np.sin(theta),
                    0.1 * r
                ]
        ts.create_dataset('debris_positions', data=positions)

        # Velocities - radial outward
        velocities = np.zeros((n_steps, n_debris, 3))
        for i in range(n_steps):
            for j in range(n_debris):
                v = 0.3 * const.c
                theta = j * 2 * np.pi / n_debris
                velocities[i, j] = [
                    v * np.cos(theta),
                    v * np.sin(theta),
                    0.1 * v
                ]
        ts.create_dataset('debris_velocities', data=velocities)

        # Proper times
        proper_times = np.ones((n_steps, n_debris)) * 2e12
        ts.create_dataset('debris_proper_times', data=proper_times)

        # Accretion - mark some as accreted
        accreted = np.zeros((n_steps, n_debris), dtype=bool)
        accreted[-1, 0:5] = True  # Last 5 accreted at end
        ts.create_dataset('debris_accreted', data=accreted)

        # BH data (Ring 0)
        n_bh = 4
        bh_positions = np.zeros((n_steps, n_bh, 3))
        radius = 3e9 * const.Gly_to_m
        for i in range(n_steps):
            for j in range(n_bh):
                angle = (j * 2 * np.pi / n_bh) + (i * np.pi / 10)
                bh_positions[i, j] = [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0
                ]
        ts.create_dataset('bh_positions', data=bh_positions)

        bh_velocities = np.zeros((n_steps, n_bh, 3))
        ts.create_dataset('bh_velocities', data=bh_velocities)

        bh_masses = np.ones((n_steps, n_bh)) * 1e10 * const.M_sun
        ts.create_dataset('bh_masses', data=bh_masses)

        # Conservation data
        cons = f.create_group('conservation')
        energy_errors = np.linspace(0.0, 0.005, n_steps)
        momentum_errors = np.linspace(0.0, 0.003, n_steps)
        cons.create_dataset('energy_error', data=energy_errors)
        cons.create_dataset('momentum_error', data=momentum_errors)
        cons.create_dataset('total_energy', data=np.ones(n_steps) * 1e60)
        cons.create_dataset('total_momentum', data=np.zeros((n_steps, 3)))

    yield filepath

    # Cleanup
    os.remove(filepath)
    os.rmdir(tmpdir)


class TestPlotGeneration:
    """Test that plotting functions create output files."""

    def test_plot_redshift_vs_distance(self, simple_simulation_file):
        """Test redshift vs distance plot creation."""
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "redshift_distance.png")

        # Create plot
        plot_redshift_vs_distance(simple_simulation_file, output_path)

        # Check file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Cleanup
        os.remove(output_path)
        os.rmdir(tmpdir)

    def test_plot_proper_time_vs_redshift(self, simple_simulation_file):
        """Test proper time vs redshift plot creation."""
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "proper_time_redshift.png")

        # Create plot
        plot_proper_time_vs_redshift(simple_simulation_file, output_path)

        # Check file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Cleanup
        os.remove(output_path)
        os.rmdir(tmpdir)

    def test_plot_ring0_trajectories_3d(self, simple_simulation_file):
        """Test Ring 0 3D trajectory plot creation."""
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "ring0_trajectories.png")

        # Create plot
        plot_ring0_trajectories_3d(simple_simulation_file, output_path)

        # Check file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Cleanup
        os.remove(output_path)
        os.rmdir(tmpdir)

    def test_plot_escape_fraction_vs_time(self, simple_simulation_file):
        """Test escape fraction vs time plot creation."""
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "escape_fraction.png")

        # Create plot
        plot_escape_fraction_vs_time(simple_simulation_file, output_path)

        # Check file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Cleanup
        os.remove(output_path)
        os.rmdir(tmpdir)

    def test_generate_summary_report(self, simple_simulation_file):
        """Test summary report generation."""
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "summary.txt")

        # Generate report
        generate_summary_report(simple_simulation_file, output_path)

        # Check file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Check content
        with open(output_path, 'r') as f:
            content = f.read()

        # Should contain key sections
        assert "SUMMARY REPORT" in content
        assert "PARTICLE STATISTICS" in content
        assert "REDSHIFT DISTRIBUTION" in content
        assert "PROPER TIME DISTRIBUTION" in content
        assert "CONSERVATION METRICS" in content

        # Cleanup
        os.remove(output_path)
        os.rmdir(tmpdir)


class TestPlotWithNoData:
    """Test plotting functions handle edge cases."""

    @pytest.fixture
    def empty_simulation_file(self):
        """Create HDF5 file with minimal data."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "empty_test.h5")

        with h5py.File(filepath, 'w') as f:
            f.create_group('config')
            ts = f.create_group('timeseries')

            # Minimal data - 1 timestep, 2 particles, no BHs
            ts.create_dataset('time', data=np.array([0.0]))
            ts.create_dataset('timestep', data=np.array([0]))

            positions = np.zeros((1, 2, 3))
            positions[0, 0] = [1e24, 0, 0]
            positions[0, 1] = [2e24, 0, 0]
            ts.create_dataset('debris_positions', data=positions)

            velocities = np.zeros((1, 2, 3))
            velocities[0, 0] = [0.1 * const.c, 0, 0]
            velocities[0, 1] = [0.2 * const.c, 0, 0]
            ts.create_dataset('debris_velocities', data=velocities)

            proper_times = np.array([[1e12, 1e12]])
            ts.create_dataset('debris_proper_times', data=proper_times)

            accreted = np.array([[False, False]])
            ts.create_dataset('debris_accreted', data=accreted)

            # No BH data
            ts.create_dataset('bh_positions', data=np.zeros((1, 0, 3)))
            ts.create_dataset('bh_velocities', data=np.zeros((1, 0, 3)))

            # Conservation data
            cons = f.create_group('conservation')
            cons.create_dataset('energy_error', data=np.array([0.0]))
            cons.create_dataset('momentum_error', data=np.array([0.0]))

        yield filepath

        os.remove(filepath)
        os.rmdir(tmpdir)

    def test_plot_with_no_bhs(self, empty_simulation_file):
        """Test Ring 0 plot with no BH data."""
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "no_bhs.png")

        # Should still create a plot (with message)
        plot_ring0_trajectories_3d(empty_simulation_file, output_path)

        assert os.path.exists(output_path)

        # Cleanup
        os.remove(output_path)
        os.rmdir(tmpdir)

    def test_report_with_minimal_data(self, empty_simulation_file):
        """Test report generation with minimal data."""
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "minimal_report.txt")

        # Should handle minimal data gracefully
        generate_summary_report(empty_simulation_file, output_path)

        assert os.path.exists(output_path)

        # Cleanup
        os.remove(output_path)
        os.rmdir(tmpdir)


class TestPlotCustomization:
    """Test plot customization options."""

    def test_escape_fraction_custom_threshold(self, simple_simulation_file):
        """Test escape fraction with custom distance threshold."""
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "escape_custom.png")

        # Use custom threshold
        plot_escape_fraction_vs_time(
            simple_simulation_file, output_path, distance_threshold=50.0
        )

        assert os.path.exists(output_path)

        # Cleanup
        os.remove(output_path)
        os.rmdir(tmpdir)
