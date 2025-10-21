"""
Unit tests for initialization functions.

Tests cover:
- Ring 0 circular orbit setup
- Static ring spiral distribution
- Debris position sampling
- Debris velocity sampling
- Complete simulation initialization
- Initial condition validation
"""

import pytest
import numpy as np
from pathlib import Path
from src.initialization import (
    initialize_ring0_circular_orbit,
    initialize_static_ring_spiral,
    sample_debris_positions_uniform_sphere,
    sample_debris_velocities_uniform,
    initialize_simulation,
    validate_initial_conditions
)
from src.state import SimulationState
from src.config import SimulationParameters, RingConfig
from src import constants as const


class TestRing0CircularOrbit:
    """Tests for Ring 0 circular orbit initialization."""

    def test_positions_on_circle(self):
        """Ring 0 BHs should be positioned on a circle in xy-plane."""
        # Create Ring 0 configuration
        ring = RingConfig(
            ring_id=0,
            count=4,
            radius=3.0 * const.Gly_to_m,
            mass_per_bh=1.0e21 * const.M_sun,
            is_static=False,
            orbital_velocity=0.8 * const.c,
            capture_radius=0.5 * const.Gly_to_m
        )

        state = SimulationState(n_bh=4, n_debris=0, M_central=4e22 * const.M_sun)
        initialize_ring0_circular_orbit(ring, 0, state)

        # Check all BHs are at correct radius
        for i in range(4):
            r = np.linalg.norm(state.bh_positions[i])
            assert abs(r - ring.radius) / ring.radius < 1e-10

            # Check z-coordinate is zero (in xy-plane)
            assert abs(state.bh_positions[i, 2]) < 1e-10

    def test_evenly_spaced_angles(self):
        """Ring 0 BHs should be evenly spaced in angle."""
        ring = RingConfig(
            ring_id=0,
            count=4,
            radius=3.0 * const.Gly_to_m,
            mass_per_bh=1.0e21 * const.M_sun,
            is_static=False,
            orbital_velocity=0.8 * const.c,
            capture_radius=0.5 * const.Gly_to_m
        )

        state = SimulationState(n_bh=4, n_debris=0, M_central=4e22 * const.M_sun)
        initialize_ring0_circular_orbit(ring, 0, state)

        # Compute angles
        angles = []
        for i in range(4):
            x = state.bh_positions[i, 0]
            y = state.bh_positions[i, 1]
            angle = np.arctan2(y, x)
            angles.append(angle)

        angles = sorted(angles)

        # Check spacing is 2π/4 = π/2
        expected_spacing = 2.0 * np.pi / 4
        for i in range(3):
            spacing = angles[i + 1] - angles[i]
            assert abs(spacing - expected_spacing) < 1e-6

    def test_velocities_perpendicular_to_radius(self):
        """Velocities should be perpendicular to position vectors."""
        ring = RingConfig(
            ring_id=0,
            count=4,
            radius=3.0 * const.Gly_to_m,
            mass_per_bh=1.0e21 * const.M_sun,
            is_static=False,
            orbital_velocity=0.8 * const.c,
            capture_radius=0.5 * const.Gly_to_m
        )

        state = SimulationState(n_bh=4, n_debris=0, M_central=4e22 * const.M_sun)
        initialize_ring0_circular_orbit(ring, 0, state)

        for i in range(4):
            pos = state.bh_positions[i]
            vel = state.bh_velocities[i]

            # Dot product should be zero (perpendicular)
            dot_product = np.dot(pos, vel)
            assert abs(dot_product) < 1e-10

    def test_velocity_magnitude(self):
        """All Ring 0 BHs should have same velocity magnitude."""
        v_orbital = 0.8 * const.c
        ring = RingConfig(
            ring_id=0,
            count=4,
            radius=3.0 * const.Gly_to_m,
            mass_per_bh=1.0e21 * const.M_sun,
            is_static=False,
            orbital_velocity=v_orbital,
            capture_radius=0.5 * const.Gly_to_m
        )

        state = SimulationState(n_bh=4, n_debris=0, M_central=4e22 * const.M_sun)
        initialize_ring0_circular_orbit(ring, 0, state)

        for i in range(4):
            v = np.linalg.norm(state.bh_velocities[i])
            assert abs(v - v_orbital) / v_orbital < 1e-10

    def test_metadata_set_correctly(self):
        """Metadata (mass, ring_id, etc.) should be set correctly."""
        ring = RingConfig(
            ring_id=0,
            count=4,
            radius=3.0 * const.Gly_to_m,
            mass_per_bh=1.0e21 * const.M_sun,
            is_static=False,
            orbital_velocity=0.8 * const.c,
            capture_radius=0.5 * const.Gly_to_m
        )

        state = SimulationState(n_bh=4, n_debris=0, M_central=4e22 * const.M_sun)
        initialize_ring0_circular_orbit(ring, 0, state)

        for i in range(4):
            assert state.bh_masses[i] == ring.mass_per_bh
            assert state.bh_ring_ids[i] == ring.ring_id
            assert state.bh_is_static[i] == ring.is_static
            assert state.bh_capture_radii[i] == ring.capture_radius


class TestStaticRingSpiral:
    """Tests for static ring spiral distribution."""

    def test_positions_on_sphere(self):
        """Static ring BHs should be on sphere of given radius."""
        ring = RingConfig(
            ring_id=1,
            count=6,
            radius=100.0 * const.Gly_to_m,
            mass_per_bh=5.0e21 * const.M_sun,
            is_static=True
        )

        state = SimulationState(n_bh=6, n_debris=0, M_central=4e22 * const.M_sun)
        initialize_static_ring_spiral(ring, 0, state)

        for i in range(6):
            r = np.linalg.norm(state.bh_positions[i])
            assert abs(r - ring.radius) / ring.radius < 1e-10

    def test_velocities_are_zero(self):
        """Static ring BHs should have zero velocity."""
        ring = RingConfig(
            ring_id=1,
            count=6,
            radius=100.0 * const.Gly_to_m,
            mass_per_bh=5.0e21 * const.M_sun,
            is_static=True
        )

        state = SimulationState(n_bh=6, n_debris=0, M_central=4e22 * const.M_sun)
        initialize_static_ring_spiral(ring, 0, state)

        for i in range(6):
            v = np.linalg.norm(state.bh_velocities[i])
            assert v < 1e-10

    def test_fibonacci_distribution(self):
        """BHs should be roughly uniformly distributed (no clustering)."""
        ring = RingConfig(
            ring_id=1,
            count=100,
            radius=100.0 * const.Gly_to_m,
            mass_per_bh=5.0e21 * const.M_sun,
            is_static=True
        )

        state = SimulationState(n_bh=100, n_debris=0, M_central=4e22 * const.M_sun)
        initialize_static_ring_spiral(ring, 0, state)

        # Check that positions are well-distributed
        # Compute minimum pairwise distance
        min_distance = float('inf')
        for i in range(100):
            for j in range(i + 1, 100):
                dist = np.linalg.norm(state.bh_positions[i] - state.bh_positions[j])
                min_distance = min(min_distance, dist)

        # With Fibonacci distribution, minimum distance should be reasonable
        # (not too small, indicating clustering)
        # Rough estimate: for 100 points on sphere, expect min ~ r * sqrt(4π/100) ~ 0.35r
        expected_min = 0.2 * ring.radius  # Conservative lower bound
        assert min_distance > expected_min


class TestDebrisPositionSampling:
    """Tests for debris position sampling."""

    def test_positions_in_radial_range(self):
        """Debris positions should be within [r_min, r_max]."""
        n_debris = 100
        r_min = 1.0e24
        r_max = 5.0e24
        rng = np.random.default_rng(42)

        positions = sample_debris_positions_uniform_sphere(n_debris, r_min, r_max, rng)

        for i in range(n_debris):
            r = np.linalg.norm(positions[i])
            assert r >= r_min
            assert r <= r_max

    def test_uniform_solid_angle_coverage(self):
        """Debris should cover all solid angles roughly uniformly."""
        n_debris = 1000
        r_min = 1.0e24
        r_max = 1.0e24  # Fixed radius for angular distribution test
        rng = np.random.default_rng(42)

        positions = sample_debris_positions_uniform_sphere(n_debris, r_min, r_max, rng)

        # Convert to spherical coordinates (just angles)
        # Check that all octants have roughly equal numbers
        octant_counts = [0] * 8
        for i in range(n_debris):
            x, y, z = positions[i]
            octant = (0 if x >= 0 else 1) + (0 if y >= 0 else 2) + (0 if z >= 0 else 4)
            octant_counts[octant] += 1

        # Each octant should have ~125 particles (1000/8)
        expected = n_debris / 8
        for count in octant_counts:
            # Allow 50% deviation (Fibonacci sphere is deterministic, not random)
            assert count > expected * 0.3
            assert count < expected * 1.7


class TestDebrisVelocitySampling:
    """Tests for debris velocity sampling."""

    def test_velocities_in_magnitude_range(self):
        """Debris velocities should have magnitudes in [v_min, v_max]."""
        n_debris = 100
        v_min = 0.01 * const.c
        v_max = 0.9 * const.c
        rng = np.random.default_rng(42)

        velocities = sample_debris_velocities_uniform(n_debris, v_min, v_max, rng)

        for i in range(n_debris):
            v = np.linalg.norm(velocities[i])
            assert v >= v_min
            assert v <= v_max

    def test_uniform_direction_coverage(self):
        """Velocity directions should cover all solid angles."""
        n_debris = 1000
        v_min = 1.0e8
        v_max = 1.0e8  # Fixed magnitude for direction test
        rng = np.random.default_rng(42)

        velocities = sample_debris_velocities_uniform(n_debris, v_min, v_max, rng)

        # Check octant distribution
        octant_counts = [0] * 8
        for i in range(n_debris):
            vx, vy, vz = velocities[i]
            octant = (0 if vx >= 0 else 1) + (0 if vy >= 0 else 2) + (0 if vz >= 0 else 4)
            octant_counts[octant] += 1

        expected = n_debris / 8
        for count in octant_counts:
            assert count > expected * 0.3
            assert count < expected * 1.7


class TestCompleteInitialization:
    """Tests for complete simulation initialization."""

    def test_initialize_from_baseline_config(self):
        """Test initialization from baseline configuration file."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))

        state = initialize_simulation(params, seed=42)

        # Check counts
        assert state.n_bh == params.total_bh_count
        assert state.n_debris == params.debris_count

        # Check initial time
        assert state.time == 0.0
        assert state.timestep_count == 0

        # Check all debris masses are equal
        assert np.all(state.debris_masses == params.debris_mass_per_particle)

        # Check no particles accreted yet
        assert state.n_debris_accreted == 0
        assert state.n_debris_active == params.debris_count

    def test_deterministic_with_seed(self):
        """Same seed should produce identical initial conditions."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))

        state1 = initialize_simulation(params, seed=42)
        state2 = initialize_simulation(params, seed=42)

        # Debris positions should be identical
        assert np.allclose(state1.debris_positions, state2.debris_positions)
        assert np.allclose(state1.debris_velocities, state2.debris_velocities)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different initial conditions."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))

        state1 = initialize_simulation(params, seed=42)
        state2 = initialize_simulation(params, seed=123)

        # Debris positions should be different
        assert not np.allclose(state1.debris_positions, state2.debris_positions)


class TestInitialConditionValidation:
    """Tests for initial condition validation."""

    def test_validation_returns_diagnostics(self):
        """Validation should return dictionary with expected keys."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))
        state = initialize_simulation(params, seed=42)

        diagnostics = validate_initial_conditions(state)

        # Check expected keys exist
        assert 'total_momentum' in diagnostics
        assert 'total_angular_momentum' in diagnostics
        assert 'debris_v_min' in diagnostics
        assert 'debris_v_max' in diagnostics
        assert 'debris_r_min' in diagnostics
        assert 'debris_r_max' in diagnostics

    def test_debris_velocity_range(self):
        """Validation should report correct velocity range."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))
        state = initialize_simulation(params, seed=42)

        diagnostics = validate_initial_conditions(state)

        # Check velocity range matches configuration
        assert diagnostics['debris_v_min'] >= params.debris_v_min * 0.99
        assert diagnostics['debris_v_max'] <= params.debris_v_max * 1.01

    def test_ring_breakdown(self):
        """Validation should report correct ring counts."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))
        state = initialize_simulation(params, seed=42)

        diagnostics = validate_initial_conditions(state)

        # Check ring counts
        for ring in params.rings:
            key = f'ring_{ring.ring_id}_count'
            assert key in diagnostics
            assert diagnostics[key] == ring.count
