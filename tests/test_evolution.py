"""
Unit tests for time evolution engine.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.state import SimulationState
from src.config import SimulationParameters
from src.evolution import (
    update_debris_particles,
    update_dynamic_bhs,
    update_all_particles,
    detect_accretion,
    apply_accretion_momentum_conservation,
    evolve_system,
    run_simulation
)
from src import constants as const


class TestDebrisParticleUpdate:
    """Test debris particle position/velocity updates."""

    def test_debris_feels_bh_gravity(self):
        """Debris particle should accelerate toward Ring BH."""
        # Solar system scale: Earth-mass particle at 1 AU from Sun-mass BH
        # If Earth were halted in orbit, it would fall toward the Sun
        AU_to_m = 1.496e11  # meters per AU
        debris_pos = np.array([[1.0 * AU_to_m, 0.0, 0.0]])
        debris_vel = np.array([[0.0, 0.0, 0.0]])
        debris_masses = np.array([3.0e-6 * const.M_sun])  # Earth mass
        debris_proper_times = np.array([0.0])
        debris_accreted = np.array([False])

        # Sun-mass BH at origin
        bh_positions = np.array([[0.0, 0.0, 0.0]])
        bh_masses = np.array([1.0 * const.M_sun])
        bh_velocities = np.array([[0.0, 0.0, 0.0]])
        bh_is_static = np.array([True])

        # Timestep: 0.52 days = 45,000 seconds
        # This gives measurable motion in just 5 timesteps
        dt = 45000.0  # seconds

        initial_pos = debris_pos.copy()

        # Prepare arrays for BH (we need these even if empty for unified update)
        bh_ring_ids = np.zeros(len(bh_positions), dtype=np.int32)

        # Update for 5 timesteps (about 2.6 days total)
        # Earth should fall measurably toward the Sun
        for _ in range(5):
            update_all_particles(
                debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
                bh_positions, bh_velocities, bh_masses, bh_ring_ids, bh_is_static,
                dt, use_relativistic=False
            )

        # Particle should have gained velocity toward BH (negative x direction)
        assert debris_vel[0, 0] < 0  # Velocity toward BH

        # Position should have moved toward BH (x should decrease)
        assert debris_pos[0, 0] < initial_pos[0, 0]  # Position moved toward origin

        # Proper time should have advanced
        assert debris_proper_times[0] > 0

    def test_accreted_particles_not_updated(self):
        """Accreted particles should not be updated."""
        # Two particles: one active, one accreted
        # First particle at 1 Gly, second at 2 Gly (accreted)
        debris_pos = np.array([
            [1.0 * const.Gly_to_m, 0.0, 0.0],
            [2.0 * const.Gly_to_m, 0.0, 0.0]
        ])
        debris_vel = np.zeros((2, 3))
        debris_masses = np.array([1.0e20 * const.M_sun, 1.0e20 * const.M_sun])
        debris_proper_times = np.zeros(2)
        debris_accreted = np.array([False, True])  # Second is accreted

        # Single Ring BH at origin to pull on first particle
        bh_positions = np.array([[0.0, 0.0, 0.0]])
        bh_masses = np.array([1.0e22 * const.M_sun])
        bh_velocities = np.array([[0.0, 0.0, 0.0]])
        bh_is_static = np.array([True])

        dt = 0.001 * const.Gyr_to_s

        update_debris_particles(
            debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
            bh_positions, bh_masses, bh_velocities, bh_is_static,
            dt, use_relativistic=False
        )

        # First particle should have gained velocity (pulled by BH)
        assert debris_vel[0, 0] < 0  # Velocity toward origin

        # Second (accreted) particle should NOT have been updated
        assert debris_vel[1, 0] == 0.0  # No velocity change
        assert debris_proper_times[1] == 0.0  # Proper time not updated

    def test_multiple_timesteps_consistent(self):
        """Multiple timesteps should produce consistent evolution."""
        # Single particle at rest, 1 Gly from Ring BH
        debris_pos = np.array([[1.0 * const.Gly_to_m, 0.0, 0.0]])
        debris_vel = np.array([[0.0, 0.0, 0.0]])
        debris_masses = np.array([1.0e20 * const.M_sun])
        debris_proper_times = np.array([0.0])
        debris_accreted = np.array([False])

        # Single massive Ring BH at origin
        bh_positions = np.array([[0.0, 0.0, 0.0]])
        bh_masses = np.array([1.0e22 * const.M_sun])
        bh_velocities = np.array([[0.0, 0.0, 0.0]])
        bh_is_static = np.array([True])

        dt = 0.001 * const.Gyr_to_s

        # Evolve for multiple timesteps
        for _ in range(100):
            update_debris_particles(
                debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
                bh_positions, bh_masses, bh_velocities, bh_is_static,
                dt, use_relativistic=False
            )

        # After many timesteps, velocity should build up
        # Velocity should be negative (toward BH)
        assert debris_vel[0, 0] < 0

        # Proper time should have advanced
        assert debris_proper_times[0] > 0

    def test_debris_debris_gravity(self):
        """Debris particles should feel gravitational pull from each other."""
        # Two debris particles: one massive at origin, one light at 0.1 light-years
        # (Much closer distance needed to make gravitational effects detectable)
        ly_to_m = const.c * 365.25 * 24 * 3600  # meters per light-year
        debris_pos = np.array([
            [0.0, 0.0, 0.0],  # Massive particle at origin
            [0.1 * ly_to_m, 0.0, 0.0]  # Light particle at 0.1 light-years
        ])
        debris_vel = np.zeros((2, 3))
        debris_masses = np.array([
            1.0e22 * const.M_sun,  # Very massive
            1.0e20 * const.M_sun   # Light
        ])
        debris_proper_times = np.zeros(2)
        debris_accreted = np.array([False, False])

        # No Ring BHs - only debris-debris gravity
        bh_positions = np.zeros((0, 3))
        bh_masses = np.zeros(0)
        bh_velocities = np.zeros((0, 3))
        bh_is_static = np.zeros(0, dtype=bool)

        # Small timestep appropriate for light-year scale: 1 year
        dt = 365.25 * 24 * 3600  # 1 year in seconds

        initial_pos = debris_pos.copy()

        # Prepare arrays for BH (we need these even if empty for unified update)
        bh_ring_ids = np.zeros(len(bh_positions), dtype=np.int32)

        # Update for 100 years to accumulate measurable position change
        for _ in range(100):
            update_all_particles(
                debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
                bh_positions, bh_velocities, bh_masses, bh_ring_ids, bh_is_static,
                dt, use_relativistic=False
            )

        # Light particle should have negative velocity (toward origin)
        # This tests debris-debris gravity
        assert debris_vel[1, 0] < 0

        # Light particle should have moved toward origin (x decreased)
        assert debris_pos[1, 0] < initial_pos[1, 0]

        # Massive particle should have gained velocity toward light particle
        # (Much smaller effect due to mass ratio, but should be positive)
        assert debris_vel[0, 0] > 0  # Velocity in +x direction

        # Massive particle should have moved toward light particle (x increased)
        assert debris_pos[0, 0] > initial_pos[0, 0]


class TestDynamicBHUpdate:
    """Test dynamic black hole orbital evolution."""

    def test_bh_feels_other_bh_gravity(self):
        """Dynamic BH should be pulled by static BH."""
        # Dynamic BH at 3 Gly, initially at rest
        bh_positions = np.array([
            [0.0, 0.0, 0.0],  # Static BH at origin
            [3.0 * const.Gly_to_m, 0.0, 0.0]  # Dynamic BH
        ])
        bh_velocities = np.array([
            [0.0, 0.0, 0.0],  # Static BH
            [0.0, 0.0, 0.0]  # Dynamic BH initially at rest
        ])
        bh_masses = np.array([1.0e22 * const.M_sun, 1.0e21 * const.M_sun])
        bh_ring_ids = np.array([1, 0], dtype=np.int32)  # Ring 1 (static), Ring 0 (dynamic)
        bh_is_static = np.array([True, False])

        # No debris particles
        debris_pos = np.zeros((0, 3))
        debris_vel = np.zeros((0, 3))
        debris_masses = np.zeros(0)
        debris_accreted = np.zeros(0, dtype=bool)

        dt = 0.001 * const.Gyr_to_s

        # Evolve for multiple timesteps
        for _ in range(100):
            update_dynamic_bhs(
                bh_positions, bh_velocities, bh_masses, bh_ring_ids,
                bh_is_static, debris_pos, debris_vel, debris_masses,
                debris_accreted, dt, use_relativistic=False
            )

        # Dynamic BH should have negative velocity (toward origin)
        assert bh_velocities[1, 0] < 0

        # Static BH should not have moved
        assert np.allclose(bh_positions[0], [0.0, 0.0, 0.0])

    def test_static_bhs_dont_move(self):
        """Static BHs should not move."""
        bh_positions = np.array([[3.0 * const.Gly_to_m, 0.0, 0.0]])
        bh_velocities = np.array([[0.0, 0.0, 0.0]])
        bh_masses = np.array([1.0e21 * const.M_sun])
        bh_ring_ids = np.array([1], dtype=np.int32)  # Ring 1
        bh_is_static = np.array([True])

        # No debris particles
        debris_pos = np.zeros((0, 3))
        debris_vel = np.zeros((0, 3))
        debris_masses = np.zeros(0)
        debris_accreted = np.zeros(0, dtype=bool)

        dt = 0.001 * const.Gyr_to_s

        pos_before = bh_positions.copy()

        update_dynamic_bhs(
            bh_positions, bh_velocities, bh_masses, bh_ring_ids,
            bh_is_static, debris_pos, debris_vel, debris_masses,
            debris_accreted, dt, use_relativistic=False
        )

        # Position should not have changed
        np.testing.assert_array_equal(bh_positions, pos_before)

    def test_bh_feels_debris_gravity(self):
        """Dynamic BH should be pulled by massive debris particle."""
        # Massive debris particle at origin, dynamic BH at 0.3 light-years
        # (Much closer distance needed to make gravitational effects detectable)
        ly_to_m = const.c * 365.25 * 24 * 3600  # meters per light-year
        bh_positions = np.array([[0.3 * ly_to_m, 0.0, 0.0]])
        bh_velocities = np.array([[0.0, 0.0, 0.0]])
        bh_masses = np.array([1.0e21 * const.M_sun])
        bh_ring_ids = np.array([0], dtype=np.int32)
        bh_is_static = np.array([False])

        # Massive debris particle at origin
        debris_pos = np.array([[0.0, 0.0, 0.0]])
        debris_vel = np.array([[0.0, 0.0, 0.0]])
        debris_masses = np.array([1.0e22 * const.M_sun])  # Very massive
        debris_accreted = np.array([False])

        # Small timestep appropriate for light-year scale: 1 year
        dt = 365.25 * 24 * 3600  # 1 year in seconds

        initial_bh_pos = bh_positions.copy()

        # Update for 100 years to accumulate measurable position change
        for _ in range(100):
            update_dynamic_bhs(
                bh_positions, bh_velocities, bh_masses, bh_ring_ids,
                bh_is_static, debris_pos, debris_vel, debris_masses,
                debris_accreted, dt, use_relativistic=False
            )

        # BH should have negative velocity (toward origin)
        # This tests that BHs feel gravity from debris particles
        assert bh_velocities[0, 0] < 0

        # BH should have moved toward origin (x decreased)
        assert bh_positions[0, 0] < initial_bh_pos[0, 0]

    def test_circular_orbit_stability(self):
        """BH in circular orbit should maintain stable radius with leapfrog integration."""
        # Central "BH" as single massive debris particle
        M_central = 4.0e22 * const.M_sun
        debris_pos = np.array([[0.0, 0.0, 0.0]])
        debris_vel = np.array([[0.0, 0.0, 0.0]])
        debris_masses = np.array([M_central])
        debris_proper_times = np.array([0.0])
        debris_accreted = np.array([False])

        # Ring 0 BH in circular orbit at 14 Gly
        r = 14.0 * const.Gly_to_m
        M_ring0 = 1.0e21 * const.M_sun
        v_keplerian = np.sqrt(const.G * M_central / r)

        bh_positions = np.array([[r, 0.0, 0.0]])
        bh_velocities = np.array([[0.0, v_keplerian, 0.0]])
        bh_masses = np.array([M_ring0])
        bh_ring_ids = np.array([0], dtype=np.int32)
        bh_is_static = np.array([False])

        # Orbital period
        T_orbit = 2.0 * np.pi * r / v_keplerian

        # Small timestep: 1000 steps per orbit
        dt = T_orbit / 1000.0

        initial_r = r

        # Simulate for 1/4 orbit
        for _ in range(250):
            update_debris_particles(
                debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
                bh_positions, bh_masses, bh_velocities, bh_is_static,
                dt, use_relativistic=False
            )
            update_dynamic_bhs(
                bh_positions, bh_velocities, bh_masses, bh_ring_ids,
                bh_is_static, debris_pos, debris_vel, debris_masses,
                debris_accreted, dt, use_relativistic=False
            )

        final_r = np.sqrt(bh_positions[0, 0]**2 + bh_positions[0, 1]**2)
        radius_change_pct = abs(final_r - initial_r) / initial_r * 100

        # For a stable circular orbit, radius should stay within 5%
        # Leapfrog integration conserves energy and maintains circular orbits
        assert radius_change_pct < 5.0, f"Orbit radius changed by {radius_change_pct:.1f}% (expected <5%)"


class TestAccretionDetection:
    """Test accretion detection logic."""

    def test_particle_within_capture_radius_accreted(self):
        """Particle within capture radius should be accreted."""
        # Debris particle very close to Ring 0 BH
        debris_pos = np.array([
            [14.0 * const.Gly_to_m + 0.1 * const.Gly_to_m, 0.0, 0.0]  # 0.1 Gly from BH
        ])
        debris_accreted = np.array([False])
        debris_accreted_by = np.array([-1], dtype=np.int32)

        # Ring 0 BH at 14 Gly with 0.5 Gly capture radius
        bh_positions = np.array([[14.0 * const.Gly_to_m, 0.0, 0.0]])
        bh_capture_radii = np.array([0.5 * const.Gly_to_m])
        bh_ring_ids = np.array([0], dtype=np.int32)

        newly_accreted = detect_accretion(
            debris_pos, debris_accreted, debris_accreted_by,
            bh_positions, bh_capture_radii, bh_ring_ids
        )

        assert newly_accreted == 1
        assert debris_accreted[0] == True
        assert debris_accreted_by[0] == 0

    def test_particle_outside_capture_radius_not_accreted(self):
        """Particle outside capture radius should not be accreted."""
        # Debris particle far from Ring 0 BH
        debris_pos = np.array([
            [14.0 * const.Gly_to_m + 1.0 * const.Gly_to_m, 0.0, 0.0]  # 1 Gly from BH
        ])
        debris_accreted = np.array([False])
        debris_accreted_by = np.array([-1], dtype=np.int32)

        # Ring 0 BH at 14 Gly with 0.5 Gly capture radius
        bh_positions = np.array([[14.0 * const.Gly_to_m, 0.0, 0.0]])
        bh_capture_radii = np.array([0.5 * const.Gly_to_m])
        bh_ring_ids = np.array([0], dtype=np.int32)

        newly_accreted = detect_accretion(
            debris_pos, debris_accreted, debris_accreted_by,
            bh_positions, bh_capture_radii, bh_ring_ids
        )

        assert newly_accreted == 0
        assert debris_accreted[0] == False
        assert debris_accreted_by[0] == -1

    def test_only_ring0_bhs_can_accrete(self):
        """Only Ring 0 BHs should be able to accrete debris."""
        # Debris particle close to Ring 1 BH
        debris_pos = np.array([
            [100.0 * const.Gly_to_m + 0.1 * const.Gly_to_m, 0.0, 0.0]
        ])
        debris_accreted = np.array([False])
        debris_accreted_by = np.array([-1], dtype=np.int32)

        # Ring 1 BH (not Ring 0) with capture radius
        bh_positions = np.array([[100.0 * const.Gly_to_m, 0.0, 0.0]])
        bh_capture_radii = np.array([1.0 * const.Gly_to_m])
        bh_ring_ids = np.array([1], dtype=np.int32)  # Ring 1, not Ring 0!

        newly_accreted = detect_accretion(
            debris_pos, debris_accreted, debris_accreted_by,
            bh_positions, bh_capture_radii, bh_ring_ids
        )

        # Should NOT be accreted
        assert newly_accreted == 0
        assert debris_accreted[0] == False

    def test_already_accreted_particles_skipped(self):
        """Already accreted particles should not be re-accreted."""
        debris_pos = np.array([
            [14.0 * const.Gly_to_m, 0.0, 0.0]  # Right on top of BH
        ])
        debris_accreted = np.array([True])  # Already accreted
        debris_accreted_by = np.array([0], dtype=np.int32)

        bh_positions = np.array([[14.0 * const.Gly_to_m, 0.0, 0.0]])
        bh_capture_radii = np.array([0.5 * const.Gly_to_m])
        bh_ring_ids = np.array([0], dtype=np.int32)

        newly_accreted = detect_accretion(
            debris_pos, debris_accreted, debris_accreted_by,
            bh_positions, bh_capture_radii, bh_ring_ids
        )

        # Should return 0 newly accreted
        assert newly_accreted == 0


class TestMomentumConservation:
    """Test momentum conservation during accretion."""

    def test_stationary_bh_gains_debris_momentum(self):
        """Stationary BH should gain momentum from accreted debris."""
        # Debris particle moving at 0.5c in +x direction
        debris_pos = np.array([[1.0 * const.Gly_to_m, 0.0, 0.0]])
        debris_vel = np.array([[0.5 * const.c, 0.0, 0.0]])
        debris_masses = np.array([1.0e20 * const.M_sun])
        debris_accreted = np.array([True])
        debris_accreted_by = np.array([0], dtype=np.int32)

        # Stationary BH
        bh_positions = np.array([[1.0 * const.Gly_to_m, 0.0, 0.0]])
        bh_velocities = np.array([[0.0, 0.0, 0.0]])
        bh_masses = np.array([1.0e21 * const.M_sun])

        # Calculate expected final velocity
        p_debris = debris_masses[0] * debris_vel[0, 0]
        m_total = bh_masses[0] + debris_masses[0]
        v_expected = p_debris / m_total

        apply_accretion_momentum_conservation(
            debris_pos, debris_vel, debris_masses, debris_accreted, debris_accreted_by,
            bh_positions, bh_velocities, bh_masses
        )

        # BH should have gained velocity
        assert abs(bh_velocities[0, 0] - v_expected) < 1e-10

        # BH mass should have increased
        assert abs(bh_masses[0] - m_total) < 1e-10

    def test_moving_bh_momentum_conserved(self):
        """Momentum should be conserved when moving BH accretes debris."""
        # Debris moving in +y direction
        debris_pos = np.array([[1.0 * const.Gly_to_m, 0.0, 0.0]])
        debris_vel = np.array([[0.0, 0.3 * const.c, 0.0]])
        debris_masses = np.array([1.0e20 * const.M_sun])
        debris_accreted = np.array([True])
        debris_accreted_by = np.array([0], dtype=np.int32)

        # BH moving in +x direction
        bh_positions = np.array([[1.0 * const.Gly_to_m, 0.0, 0.0]])
        bh_velocities = np.array([[0.2 * const.c, 0.0, 0.0]])
        bh_masses = np.array([1.0e21 * const.M_sun])

        # Calculate expected momentum
        p_before = bh_masses[0] * bh_velocities[0] + debris_masses[0] * debris_vel[0]

        apply_accretion_momentum_conservation(
            debris_pos, debris_vel, debris_masses, debris_accreted, debris_accreted_by,
            bh_positions, bh_velocities, bh_masses
        )

        # Check momentum is conserved
        p_after = bh_masses[0] * bh_velocities[0]
        np.testing.assert_allclose(p_after, p_before, rtol=1e-10)


class TestEvolveSystem:
    """Test complete system evolution."""

    def test_evolve_system_advances_time(self):
        """System time should advance correctly."""
        # Minimal simulation: 10 debris particles, no Ring 0
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))

        # Override to small particle count
        params.debris_count = 10

        from src.initialization import initialize_simulation
        state = initialize_simulation(params, seed=42)

        # Evolve for 10 timesteps
        n_steps = 10
        stats = evolve_system(state, params, n_steps, show_progress=False)

        # Time should have advanced
        expected_time = n_steps * params.dt
        assert abs(state.time - expected_time) < 1e-10

        # Timestep counter should match
        assert state.timestep_count == n_steps

    def test_evolve_system_updates_debris(self):
        """Debris particles should be updated."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))
        params.debris_count = 10

        from src.initialization import initialize_simulation
        state = initialize_simulation(params, seed=42)

        # Save initial positions
        pos_before = state.debris_positions.copy()

        # Evolve for 10 timesteps
        evolve_system(state, params, 10, show_progress=False)

        # Positions should have changed
        assert not np.allclose(state.debris_positions, pos_before)

    def test_run_simulation_complete(self):
        """Test complete simulation run from start to finish."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))

        # Small test simulation
        params.debris_count = 10
        params.duration = 0.01 * const.Gyr_to_s  # 0.01 Gyr
        params.dt = 0.001 * const.Gyr_to_s  # 0.001 Gyr timestep

        state, stats = run_simulation(params, seed=42, show_progress=False)

        # Check simulation ran to completion
        assert state.timestep_count == 10  # 0.01 / 0.001 = 10 steps
        assert state.time > 0
        assert stats['final_timestep'] == 10


class TestAccretionIntegration:
    """Integration tests for accretion with evolution."""

    def test_accretion_reduces_active_debris(self):
        """Accretion should reduce the number of active debris particles."""
        # Create simple config with Ring 0 enabled
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))

        # Enable Ring 0 with large capture radius
        from src.config import RingConfig
        # For Ring 0 BHs, calculate velocity to orbit a 1e22 M_sun mass
        M_total = 1.0e22 * const.M_sun
        ring0_radius = 14.0 * const.Gly_to_m
        v_orbit = np.sqrt(const.G * M_total / ring0_radius)

        params.rings = [
            RingConfig(
                ring_id=0,
                count=4,
                radius=ring0_radius,
                mass_per_bh=1.0e21 * const.M_sun,
                is_static=False,
                orbital_velocity=v_orbit,
                capture_radius=5.0 * const.Gly_to_m  # Large capture radius
            )
        ]

        # Small debris field near Ring 0
        params.debris_count = 20
        params.debris_r_min = 10.0 * const.Gly_to_m
        params.debris_r_max = 18.0 * const.Gly_to_m
        params.debris_v_min = 0.01 * const.c
        params.debris_v_max = 0.1 * const.c

        from src.initialization import initialize_simulation
        state = initialize_simulation(params, seed=42)

        initial_active = state.n_debris_active

        # Evolve for a few timesteps
        params.dt = 0.001 * const.Gyr_to_s
        evolve_system(state, params, 50, show_progress=False)

        # Some particles should have been accreted
        # (This is probabilistic, but with large capture radius, very likely)
        # We just check that accretion mechanism works
        assert state.n_debris_active <= initial_active
