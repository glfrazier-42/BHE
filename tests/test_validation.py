"""
Unit tests for configuration validation.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from bhe.config import SimulationParameters
from bhe import constants as const


def test_valid_baseline_config():
    """Baseline config should pass all validation checks."""
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    warnings = params.validate()

    # Should have no errors
    errors = [w for w in warnings if w.startswith("ERROR")]
    assert len(errors) == 0, f"Unexpected errors: {errors}"


def test_ring0_inside_schwarzschild_radius():
    """Should error if Ring 0 is inside Schwarzschild radius."""
    test_config = {
        'simulation_name': 'test',
        'output_directory': './test',
        'central_black_hole': {'mass_solar_masses': 4.0e+22},
        'ring_0': {
            'count': 4,
            'radius_gly': 10.0,  # Inside r_s ≈ 12.5 Gly
            'mass_per_bh_solar_masses': 1.0e+21,
            'velocity_mode': 'keplerian',
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
        warnings = params.validate()

        # Should have error about being inside Schwarzschild radius
        errors = [w for w in warnings if "INSIDE" in w and "Schwarzschild" in w]
        assert len(errors) > 0, "Should detect Ring 0 inside Schwarzschild radius"

    finally:
        import os
        os.unlink(temp_path)


def test_debris_velocity_exceeds_c():
    """Should error if debris max velocity >= c."""
    test_config = {
        'simulation_name': 'test',
        'output_directory': './test',
        'central_black_hole': {'mass_solar_masses': 4.0e+22},
        'debris_field': {
            'particle_count': 100,
            'position_min_gly': 0.01,
            'position_max_gly': 0.1,
            'velocity_min_fraction_c': 0.01,
            'velocity_max_fraction_c': 1.2,  # > c!
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
        warnings = params.validate()

        # Should have error about velocity >= c
        errors = [w for w in warnings if "debris_v_max" in w and "must be < c" in w]
        assert len(errors) > 0, "Should detect debris velocity >= c"

    finally:
        import os
        os.unlink(temp_path)


def test_negative_timestep():
    """Should error if timestep is negative."""
    test_config = {
        'simulation_name': 'test',
        'output_directory': './test',
        'central_black_hole': {'mass_solar_masses': 4.0e+22},
        'debris_field': {
            'particle_count': 100,
            'position_min_gly': 0.01,
            'position_max_gly': 0.1,
            'velocity_min_fraction_c': 0.01,
            'velocity_max_fraction_c': 0.9,
            'distribution': 'uniform'
        },
        'simulation_control': {
            'timestep_gyr': -0.001,  # Negative!
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
        warnings = params.validate()

        # Should have error about negative timestep
        errors = [w for w in warnings if "timestep dt must be positive" in w]
        assert len(errors) > 0, "Should detect negative timestep"

    finally:
        import os
        os.unlink(temp_path)


def test_non_keplerian_velocity_warning():
    """Should warn if velocity differs significantly from Keplerian."""
    test_config = {
        'simulation_name': 'test',
        'output_directory': './test',
        'central_black_hole': {'mass_solar_masses': 4.0e+22},
        'ring_0': {
            'count': 4,
            'radius_gly': 14.0,
            'mass_per_bh_solar_masses': 1.0e+21,
            'velocity_mode': 'manual',
            'orbital_velocity_fraction_c': 0.3,  # Much less than Keplerian (~0.67c)
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
        warnings = params.validate()

        # Should warn about velocity being much less than Keplerian
        warns = [w for w in warnings if "much less than" in w and "Keplerian" in w]
        assert len(warns) > 0, "Should warn about non-Keplerian velocity"

    finally:
        import os
        os.unlink(temp_path)
