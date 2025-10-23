"""
Tests for visualization module.
"""

import pytest
import numpy as np
import h5py
import tempfile
import os
from pathlib import Path

from bhe.visualization import (
    plot_redshift_vs_distance,
    plot_proper_time_vs_redshift,
    plot_ring0_trajectories_3d,
    plot_escape_fraction_vs_time,
    generate_summary_report
)
from bhe import constants as const


@pytest.fixture
def simple_simulation_file():
    """Create a simple HDF5 simulation file for visualization testing (unified particle system, natural units)."""
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "viz_test.h5")

    with h5py.File(filepath, 'w') as f:
        from bhe.state import BLACK_HOLE, DEBRIS

        # Config
        f.create_group('config')

        # Timeseries data
        ts = f.create_group('timeseries')

        n_steps = 5
        n_debris = 20
        n_bh = 4
        n_total = n_bh + n_debris  # 24 total particles

        # Times (years)
        times = np.linspace(0, 4e12, n_steps)
        ts.create_dataset('time', data=times)
        ts.create_dataset('timestep', data=np.arange(n_steps))

        # Unified positions (natural units: ly) - BHs first, then debris
        positions = np.zeros((n_steps, n_total, 3))

        # BH positions (Ring 0) - first 4 particles
        radius = 3.0e9  # 3 Gly in ly
        for i in range(n_steps):
            for j in range(n_bh):
                angle = (j * 2 * np.pi / n_bh) + (i * np.pi / 10)
                positions[i, j] = [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0
                ]

        # Debris positions - particles 4-23 spread out in space
        for i in range(n_steps):
            for j in range(n_debris):
                # Particles move outward over time
                r = (j + 1) * 10.0e9 * (i + 1) / n_steps  # ly
                theta = j * 2 * np.pi / n_debris
                positions[i, n_bh + j] = [
                    r * np.cos(theta),
                    r * np.sin(theta),
                    0.1 * r
                ]
        ts.create_dataset('positions', data=positions)

        # Unified velocities (natural units: fraction of c)
        velocities = np.zeros((n_steps, n_total, 3))

        # BH velocities (zero for simplicity)
        # Already zeros

        # Debris velocities - radial outward
        for i in range(n_steps):
            for j in range(n_debris):
                v = 0.3  # 0.3c
                theta = j * 2 * np.pi / n_debris
                velocities[i, n_bh + j] = [
                    v * np.cos(theta),
                    v * np.sin(theta),
                    0.1 * v
                ]
        ts.create_dataset('velocities', data=velocities)

        # Unified masses (natural units: M_sun)
        masses = np.zeros((n_steps, n_total))
        masses[:, :n_bh] = 1e10  # BH masses
        masses[:, n_bh:] = 1.0  # Debris masses
        ts.create_dataset('masses', data=masses)

        # Proper times (years)
        proper_times = np.ones((n_steps, n_total)) * 2e12
        ts.create_dataset('proper_times', data=proper_times)

        # Accretion - mark some debris as accreted
        accreted = np.zeros((n_steps, n_total), dtype=bool)
        accreted[-1, n_bh:n_bh+5] = True  # First 5 debris accreted at end
        ts.create_dataset('accreted', data=accreted)

        # Metadata (constant throughout simulation)
        meta = f.create_group('metadata')
        particle_type = np.zeros(n_total, dtype=int)
        particle_type[:n_bh] = BLACK_HOLE
        particle_type[n_bh:] = DEBRIS
        meta.create_dataset('particle_type', data=particle_type)

        ring_id = np.full(n_total, -1, dtype=int)
        ring_id[:n_bh] = 0  # BHs are Ring 0
        meta.create_dataset('ring_id', data=ring_id)

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
        """Create HDF5 file with minimal data (unified particle system)."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "empty_test.h5")

        with h5py.File(filepath, 'w') as f:
            from bhe.state import DEBRIS

            f.create_group('config')
            ts = f.create_group('timeseries')

            # Minimal data - 1 timestep, 2 debris particles (natural units)
            ts.create_dataset('time', data=np.array([0.0]))
            ts.create_dataset('timestep', data=np.array([0]))

            # Unified positions (2 particles, natural units: ly)
            positions = np.zeros((1, 2, 3))
            positions[0, 0] = [1.0e9, 0, 0]  # 1 Gly
            positions[0, 1] = [2.0e9, 0, 0]  # 2 Gly
            ts.create_dataset('positions', data=positions)

            # Unified velocities (natural units: fraction of c)
            velocities = np.zeros((1, 2, 3))
            velocities[0, 0] = [0.1, 0, 0]  # 0.1c
            velocities[0, 1] = [0.2, 0, 0]  # 0.2c
            ts.create_dataset('velocities', data=velocities)

            # Unified masses (natural units: M_sun)
            masses = np.ones((1, 2))
            ts.create_dataset('masses', data=masses)

            # Unified proper times (years)
            proper_times = np.array([[1e12, 1e12]])
            ts.create_dataset('proper_times', data=proper_times)

            # Unified accreted flags
            accreted = np.array([[False, False]])
            ts.create_dataset('accreted', data=accreted)

            # Metadata (all debris particles, no BHs)
            meta = f.create_group('metadata')
            meta.create_dataset('particle_type', data=np.array([DEBRIS, DEBRIS]))
            meta.create_dataset('ring_id', data=np.array([-1, -1]))
            meta.create_dataset('capture_radius', data=np.zeros(2))
            meta.attrs['n_total'] = 2
            meta.attrs['n_bh'] = 0
            meta.attrs['n_debris'] = 2

            # Conservation data
            cons = f.create_group('conservation')
            cons.create_dataset('total_energy', data=np.array([1.0]))
            cons.create_dataset('total_momentum', data=np.zeros((1, 3)))
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
