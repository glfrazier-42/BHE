"""
Unit tests for configuration loading and parsing.
"""

import pytest
from pathlib import Path
from bhe.config import SimulationParameters, RingConfig
from bhe import constants as const


def test_load_baseline_config():
    """Test loading the baseline configuration file."""
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    # Check basic metadata
    assert params.simulation_name == "baseline_ring0_test"
    assert params.output_directory == "./results/baseline"

    # Check central BH mass (4e22 solar masses)
    expected_mass = 4.0e22 
    assert abs(params.M_central - expected_mass) / expected_mass < 1e-10

    # Check debris field
    assert params.debris_count == 1000
    assert params.debris_distribution == "uniform"

    # Check debris velocities (converted from fraction of c)
    assert abs(params.debris_v_min - 0.01 * const.c) / const.c < 1e-10
    assert abs(params.debris_v_max - 0.92 * const.c) / const.c < 1e-10

    # Check debris mass per particle
    expected_debris_mass = params.M_central / params.debris_count
    assert params.debris_mass_per_particle == expected_debris_mass


def test_ring_configurations():
    """Test that rings are parsed correctly."""
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    # Baseline has Ring 0 disabled (count=0), so should have 3 rings
    assert len(params.rings) == 3
    assert params.total_bh_count == 4 + 6 + 8  # Ring 1, 2, 3 counts

    # Check Ring 1 (should be first in list since Ring 0 is disabled)
    ring1 = params.rings[0]
    assert ring1.ring_id == 1
    assert ring1.count == 4
    assert ring1.is_static == True
    assert abs(ring1.radius - 100.0 * const.Gly) / const.Gly < 1e-10
    assert abs(ring1.mass_per_bh - 5.0e21 ) / 1.0 < 1e-10

    # Check Ring 2
    ring2 = params.rings[1]
    assert ring2.ring_id == 2
    assert ring2.count == 6
    assert ring2.is_static == True

    # Check Ring 3
    ring3 = params.rings[2]
    assert ring3.ring_id == 3
    assert ring3.count == 8
    assert ring3.is_static == True


def test_simulation_control_parameters():
    """Test simulation control parameters are converted correctly."""
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    # Check time conversions (Gyr to seconds)
    assert abs(params.dt - 0.001 * 1.0e9) / 1.0e9 < 1e-10
    assert abs(params.duration - 50.0 * 1.0e9) / 1.0e9 < 1e-10
    assert abs(params.output_interval - 0.1 * 1.0e9) / 1.0e9 < 1e-10
    assert abs(params.checkpoint_interval - 5.0 * 1.0e9) / 1.0e9 < 1e-10


def test_physics_options():
    """Test physics options are loaded correctly."""
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    assert params.force_method == "direct"
    assert params.use_newtonian_enhancements == False

    # Barnes-Hut parameters
    assert params.barnes_hut_theta == 0.5
    assert params.barnes_hut_max_particles_per_leaf == 8
    assert params.barnes_hut_tree_rebuild_interval == 10


def test_diagnostics_options():
    """Test diagnostics options are loaded correctly."""
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    assert params.check_energy_conservation == True
    assert params.check_momentum_conservation == True
    assert params.log_level == "INFO"


def test_config_file_not_found():
    """Test that appropriate error is raised for missing config file."""
    with pytest.raises(FileNotFoundError):
        SimulationParameters.from_yaml('nonexistent_config.yaml')


def test_ring_config_repr():
    """Test RingConfig string representation."""
    # Static ring
    ring = RingConfig(
        ring_id=1,
        count=4,
        radius=100.0 * const.Gly,
        mass_per_bh=5.0e21 ,
        is_static=True
    )
    repr_str = repr(ring)
    assert "Ring 1" in repr_str
    assert "4 BHs" in repr_str
    assert "100.0 Gly" in repr_str
    assert "static" in repr_str

    # Dynamic ring (Ring 0)
    ring0 = RingConfig(
        ring_id=0,
        count=4,
        radius=3.0 * const.Gly,
        mass_per_bh=1.0e21 ,
        is_static=False,
        orbital_velocity=0.8 * const.c,
        capture_radius=0.5 * const.Gly
    )
    repr_str = repr(ring0)
    assert "Ring 0" in repr_str
    assert "v=0.80c" in repr_str


def test_simulation_parameters_repr():
    """Test SimulationParameters string representation."""
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    repr_str = repr(params)
    assert "baseline_ring0_test" in repr_str
    assert "Central BH" in repr_str
    assert "Debris: 1000 particles" in repr_str
    assert "Duration: 50.00 Gyr" in repr_str


def test_keplerian_velocity_mode():
    """Test that velocity_mode='keplerian' calculates correct orbital velocity."""
    import tempfile
    import yaml
    import numpy as np

    # Create a test config with Ring 0 in keplerian mode
    test_config = {
        'simulation_name': 'test_keplerian',
        'output_directory': './test',
        'central_black_hole': {
            'mass_solar_masses': 4.0e+22
        },
        'ring_0': {
            'count': 4,
            'radius_gly': 3.0,
            'mass_per_bh_solar_masses': 1.0e+21,
            'velocity_mode': 'keplerian',  # Auto-calculate
            'capture_radius_gly': 0.5
        },
        'debris_field': {
            'particle_count': 100,
            'position_min_gly': 0.01,
            'position_max_gly': 0.1,
            'velocity_min_fraction_c': 0.01,
            'velocity_max_fraction_c': 0.9,
            'distribution': 'uniform'
        },
        'simulation_control': {
            'timestep_gyr': 0.001,
            'duration_gyr': 10.0,
            'output_interval_gyr': 0.1,
            'checkpoint_interval_gyr': 1.0
        },
        'physics_options': {
            'force_method': 'direct',
            'use_newtonian_enhancements': False
        },
        'diagnostics': {
            'check_energy_conservation': True,
            'check_momentum_conservation': True,
            'log_level': 'INFO'
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = f.name

    try:
        params = SimulationParameters.from_yaml(temp_path)

        # Ring 0 should exist
        assert len(params.rings) == 1
        ring0 = params.rings[0]
        assert ring0.ring_id == 0

        # Calculate expected Keplerian velocity
        # v = sqrt(G * M / r)
        r = 3.0 * const.Gly
        M = 4.0e+22 
        v_expected = np.sqrt(const.G * M / r)

        # Check that orbital velocity matches Keplerian calculation
        assert abs(ring0.orbital_velocity - v_expected) / v_expected < 1e-10

        # For M=4e22 M_sun at r=3 Gly, Keplerian velocity is actually > c!
        # This shows the configuration is unphysical - no stable orbit possible
        # v = sqrt(G*M/r) ≈ 1.44c for these parameters
        assert ring0.orbital_velocity > const.c  # Faster than light! Unphysical!

    finally:
        import os
        os.unlink(temp_path)
