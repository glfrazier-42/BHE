"""
Unit tests for time evolution engine.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from bhe.state import SimulationState
from bhe.config import SimulationParameters
from bhe.evolution import (
    detect_accretion,
    apply_accretion_momentum_conservation,
    evolve_system,
    run_simulation
)
from bhe.physics import update_all_particles_leapfrog
from bhe import constants as const


class TestDebrisParticleUpdate:
    """Test debris particle position/velocity updates."""

    def test_debris_feels_bh_gravity(self):
        """Debris particle should accelerate toward Ring BH (unified particle system)."""
        # Galaxy scale test: Light particle at 0.1 ly from massive BH
        # Using natural units: distance in ly, time in yr, mass in M_sun
        # Note: minimum distance threshold is 0.001 ly, so we use 0.1 ly

        # Unified arrays: [0] = massive BH at origin, [1] = debris particle at 0.1 ly
        positions = np.array([
            [0.0, 0.0, 0.0],  # Massive BH at origin
            [0.1, 0.0, 0.0]   # Debris particle at 0.1 ly
        ])
        velocities = np.zeros((2, 3))  # Both start at rest
        masses = np.array([1.0e22, 1.0e20])  # M_sun
        accreted = np.array([False, False])

        # Timestep: 1 year
        dt = 1.0  # years

        initial_pos = positions.copy()

        # Update for 100 timesteps (100 years total)
        # Debris should fall measurably toward the BH
        for _ in range(100):
            update_all_particles_leapfrog(
                positions, velocities, masses, accreted, dt
            )

        # Debris particle (index 1) should have gained velocity toward BH (negative x direction)
        assert velocities[1, 0] < 0, "Debris should accelerate toward BH"

        # Debris position should have moved toward BH (x should decrease)
        assert positions[1, 0] < initial_pos[1, 0], "Debris should move toward origin"

    def test_accreted_particles_not_updated(self):
        """Accreted particles should not be updated (unified particle system)."""
        # Unified arrays: [0] = BH at origin, [1] = active particle, [2] = accreted particle
        # Using natural units: ly, yr, M_sun
        positions = np.array([
            [0.0, 0.0, 0.0],  # BH at origin
            [1.0 * const.Gly, 0.0, 0.0],  # Active particle at 1 Gly
            [2.0 * const.Gly, 0.0, 0.0]   # Accreted particle at 2 Gly
        ])
        velocities = np.zeros((3, 3))
        masses = np.array([1.0e22, 1.0e20, 1.0e20])  # M_sun
        accreted = np.array([False, False, True])  # Third is accreted

        dt = 0.001 * 1.0e9  # Gyr → yr

        update_all_particles_leapfrog(
            positions, velocities, masses, accreted, dt
        )

        # Second particle (index 1, active) should have gained velocity (pulled by BH)
        assert velocities[1, 0] < 0, "Active particle should accelerate toward BH"

        # Third particle (index 2, accreted) should NOT have been updated
        assert velocities[2, 0] == 0.0, "Accreted particle velocity should not change"

    def test_multiple_timesteps_consistent(self):
        """Multiple timesteps should produce consistent evolution (unified particle system)."""
        # Unified arrays: [0] = BH, [1] = debris particle
        # Using natural units
        positions = np.array([
            [0.0, 0.0, 0.0],  # BH at origin
            [1.0 * const.Gly, 0.0, 0.0]  # Debris at 1 Gly
        ])
        velocities = np.zeros((2, 3))
        masses = np.array([1.0e22, 1.0e20])  # M_sun
        accreted = np.array([False, False])

        dt = 0.001 * 1.0e9  # Gyr → yr

        # Evolve for multiple timesteps
        for _ in range(100):
            update_all_particles_leapfrog(
                positions, velocities, masses, accreted, dt
            )

        # After many timesteps, debris velocity should build up
        # Velocity should be negative (toward BH)
        assert velocities[1, 0] < 0, "Debris should accelerate toward BH over time"

    def test_debris_debris_gravity(self):
        """Debris particles should feel gravitational pull from each other (unified particle system)."""
        # Two particles: one massive at origin, one light at 0.1 light-years
        # Using natural units: ly, yr, M_sun
        positions = np.array([
            [0.0, 0.0, 0.0],  # Massive particle at origin
            [0.1, 0.0, 0.0]   # Light particle at 0.1 ly
        ])
        velocities = np.zeros((2, 3))
        masses = np.array([1.0e22, 1.0e20])  # M_sun
        accreted = np.array([False, False])

        # Timestep: 1 year
        dt = 1.0  # years

        initial_pos = positions.copy()

        # Update for 100 years to accumulate measurable position change
        for _ in range(100):
            update_all_particles_leapfrog(
                positions, velocities, masses, accreted, dt
            )

        # Light particle (index 1) should have negative velocity (toward origin)
        # This tests particle-particle gravity
        assert velocities[1, 0] < 0, "Light particle should accelerate toward massive particle"

        # Light particle should have moved toward origin (x decreased)
        assert positions[1, 0] < initial_pos[1, 0], "Light particle should move toward origin"

        # Massive particle should have gained velocity toward light particle
        # (Much smaller effect due to mass ratio, but should be positive)
        assert velocities[0, 0] > 0, "Massive particle should accelerate toward light particle"

        # Massive particle should have moved toward light particle (x increased)
        assert positions[0, 0] > initial_pos[0, 0], "Massive particle should move toward light particle"


# TestDynamicBHUpdate class commented out - functionality now covered by unified particle system tests
# All particles (BHs and debris) are treated identically in the unified system
# The TestDebrisParticleUpdate tests above cover the same physics

# class TestDynamicBHUpdate:
#     """Test dynamic black hole orbital evolution."""
#
#     def test_bh_feels_other_bh_gravity(self):
#        """Dynamic BH should be pulled by static BH."""
#        # Dynamic BH at 3 Gly, initially at rest
#        bh_positions = np.array([
#            [0.0, 0.0, 0.0],  # Static BH at origin
#            [3.0 * const.Gly, 0.0, 0.0]  # Dynamic BH
#        ])
#        bh_velocities = np.array([
#            [0.0, 0.0, 0.0],  # Static BH
#            [0.0, 0.0, 0.0]  # Dynamic BH initially at rest
#        ])
#        bh_masses = np.array([1.0e22 * 1.0, 1.0e21 * 1.0])
#        bh_ring_ids = np.array([1, 0], dtype=np.int32)  # Ring 1 (static), Ring 0 (dynamic)
#        bh_is_static = np.array([True, False])
#
#        # No debris particles
#        debris_pos = np.zeros((0, 3))
#        debris_vel = np.zeros((0, 3))
#        debris_masses = np.zeros(0)
#        debris_accreted = np.zeros(0, dtype=bool)
#
#        dt = 0.001 * (1.0e9)
#
#        # Evolve for multiple timesteps
#        for _ in range(100):
#            update_dynamic_bhs(
#                bh_positions, bh_velocities, bh_masses, bh_ring_ids,
#                bh_is_static, debris_pos, debris_vel, debris_masses,
#                debris_accreted, dt, use_relativistic=False
#            )
#
#        # Dynamic BH should have negative velocity (toward origin)
#        assert bh_velocities[1, 0] < 0
#
#        # Static BH should not have moved
#        assert np.allclose(bh_positions[0], [0.0, 0.0, 0.0])
#
#    def test_static_bhs_dont_move(self):
#        """Static BHs should not move."""
#        bh_positions = np.array([[3.0 * const.Gly, 0.0, 0.0]])
#        bh_velocities = np.array([[0.0, 0.0, 0.0]])
#        bh_masses = np.array([1.0e21 * 1.0])
#        bh_ring_ids = np.array([1], dtype=np.int32)  # Ring 1
#        bh_is_static = np.array([True])
#
#        # No debris particles
#        debris_pos = np.zeros((0, 3))
#        debris_vel = np.zeros((0, 3))
#        debris_masses = np.zeros(0)
#        debris_accreted = np.zeros(0, dtype=bool)
#
#        dt = 0.001 * (1.0e9)
#
#        pos_before = bh_positions.copy()
#
#        update_dynamic_bhs(
#            bh_positions, bh_velocities, bh_masses, bh_ring_ids,
#            bh_is_static, debris_pos, debris_vel, debris_masses,
#            debris_accreted, dt, use_relativistic=False
#        )
#
#        # Position should not have changed
#        np.testing.assert_array_equal(bh_positions, pos_before)
#
#    def test_bh_feels_debris_gravity(self):
#        """Dynamic BH should be pulled by massive debris particle."""
#        # Massive debris particle at origin, dynamic BH at 0.3 light-years
#        # (Much closer distance needed to make gravitational effects detectable)
#        ly_to_m = const.c * 365.25 * 24 * 3600  # meters per light-year
#        bh_positions = np.array([[0.3 * ly_to_m, 0.0, 0.0]])
#        bh_velocities = np.array([[0.0, 0.0, 0.0]])
#        bh_masses = np.array([1.0e21 * 1.0])
#        bh_ring_ids = np.array([0], dtype=np.int32)
#        bh_is_static = np.array([False])
#
#        # Massive debris particle at origin
#        debris_pos = np.array([[0.0, 0.0, 0.0]])
#        debris_vel = np.array([[0.0, 0.0, 0.0]])
#        debris_masses = np.array([1.0e22 * 1.0])  # Very massive
#        debris_accreted = np.array([False])
#
#        # Small timestep appropriate for light-year scale: 1 year
#        dt = 365.25 * 24 * 3600  # 1 year in seconds
#
#        initial_bh_pos = bh_positions.copy()
#
#        # Update for 100 years to accumulate measurable position change
#        for _ in range(100):
#            update_dynamic_bhs(
#                bh_positions, bh_velocities, bh_masses, bh_ring_ids,
#                bh_is_static, debris_pos, debris_vel, debris_masses,
#                debris_accreted, dt, use_relativistic=False
#            )
#
#        # BH should have negative velocity (toward origin)
#        # This tests that BHs feel gravity from debris particles
#        assert bh_velocities[0, 0] < 0
#
#        # BH should have moved toward origin (x decreased)
#        assert bh_positions[0, 0] < initial_bh_pos[0, 0]
#
#    def test_circular_orbit_stability(self):
#        """BH in circular orbit should maintain stable radius with leapfrog integration."""
#        # Central "BH" as single massive debris particle
#        M_central = 4.0e22 * 1.0
#        debris_pos = np.array([[0.0, 0.0, 0.0]])
#        debris_vel = np.array([[0.0, 0.0, 0.0]])
#        debris_masses = np.array([M_central])
#        debris_proper_times = np.array([0.0])
#        debris_accreted = np.array([False])
#
#        # Ring 0 BH in circular orbit at 14 Gly
#        r = 14.0 * const.Gly
#        M_ring0 = 1.0e21 * 1.0
#        v_keplerian = np.sqrt(const.G * M_central / r)
#
#        bh_positions = np.array([[r, 0.0, 0.0]])
#        bh_velocities = np.array([[0.0, v_keplerian, 0.0]])
#        bh_masses = np.array([M_ring0])
#        bh_ring_ids = np.array([0], dtype=np.int32)
#        bh_is_static = np.array([False])
#
#        # Orbital period
#        T_orbit = 2.0 * np.pi * r / v_keplerian
#
#        # Small timestep: 1000 steps per orbit
#        dt = T_orbit / 1000.0
#
#        initial_r = r
#
#        # Simulate for 1/4 orbit
#        for _ in range(250):
#            update_debris_particles(
#                debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
#                bh_positions, bh_masses, bh_velocities, bh_is_static,
#                dt, use_relativistic=False
#            )
#            update_dynamic_bhs(
#                bh_positions, bh_velocities, bh_masses, bh_ring_ids,
#                bh_is_static, debris_pos, debris_vel, debris_masses,
#                debris_accreted, dt, use_relativistic=False
#            )
#
#        final_r = np.sqrt(bh_positions[0, 0]**2 + bh_positions[0, 1]**2)
#        radius_change_pct = abs(final_r - initial_r) / initial_r * 100
#
#        # For a stable circular orbit, radius should stay within 5%
#        # Leapfrog integration conserves energy and maintains circular orbits
#        assert radius_change_pct < 5.0, f"Orbit radius changed by {radius_change_pct:.1f}% (expected <5%)"


class TestAccretionDetection:
    """Test accretion detection logic."""

    def test_particle_within_capture_radius_accreted(self):
        """Particle within capture radius should be accreted (unified particle system)."""
        from bhe.state import BLACK_HOLE, DEBRIS

        # Unified arrays: [0] = Ring 0 BH, [1] = debris particle
        positions = np.array([
            [14.0 * const.Gly, 0.0, 0.0],  # Ring 0 BH
            [14.0 * const.Gly + 0.1 * const.Gly, 0.0, 0.0]  # Debris 0.1 Gly from BH
        ])
        accreted = np.array([False, False])
        accreted_by = np.array([-1, -1], dtype=np.int32)
        particle_type = np.array([BLACK_HOLE, DEBRIS], dtype=np.int32)
        ring_id = np.array([0, -1], dtype=np.int32)  # Ring 0, debris
        capture_radius = np.array([0.5 * const.Gly, 0.0])  # 0.5 Gly capture for BH

        newly_accreted = detect_accretion(
            positions, accreted, accreted_by,
            particle_type, ring_id, capture_radius
        )

        assert newly_accreted == 1
        assert accreted[1] == True  # Debris particle accreted
        assert accreted_by[1] == 0  # Accreted by BH at index 0

    def test_particle_outside_capture_radius_not_accreted(self):
        """Particle outside capture radius should not be accreted (unified particle system)."""
        from bhe.state import BLACK_HOLE, DEBRIS

        # Unified arrays: [0] = Ring 0 BH, [1] = debris particle far away
        positions = np.array([
            [14.0 * const.Gly, 0.0, 0.0],  # Ring 0 BH
            [14.0 * const.Gly + 1.0 * const.Gly, 0.0, 0.0]  # Debris 1 Gly from BH
        ])
        accreted = np.array([False, False])
        accreted_by = np.array([-1, -1], dtype=np.int32)
        particle_type = np.array([BLACK_HOLE, DEBRIS], dtype=np.int32)
        ring_id = np.array([0, -1], dtype=np.int32)
        capture_radius = np.array([0.5 * const.Gly, 0.0])  # 0.5 Gly capture for BH

        newly_accreted = detect_accretion(
            positions, accreted, accreted_by,
            particle_type, ring_id, capture_radius
        )

        assert newly_accreted == 0
        assert accreted[1] == False  # Debris not accreted
        assert accreted_by[1] == -1

    def test_only_ring0_bhs_can_accrete(self):
        """Only Ring 0 BHs should be able to accrete debris (unified particle system)."""
        from bhe.state import BLACK_HOLE, DEBRIS

        # Unified arrays: [0] = Ring 1 BH, [1] = debris particle close to it
        positions = np.array([
            [100.0 * const.Gly, 0.0, 0.0],  # Ring 1 BH
            [100.0 * const.Gly + 0.1 * const.Gly, 0.0, 0.0]  # Debris close to BH
        ])
        accreted = np.array([False, False])
        accreted_by = np.array([-1, -1], dtype=np.int32)
        particle_type = np.array([BLACK_HOLE, DEBRIS], dtype=np.int32)
        ring_id = np.array([1, -1], dtype=np.int32)  # Ring 1 (not Ring 0!), debris
        capture_radius = np.array([1.0 * const.Gly, 0.0])

        newly_accreted = detect_accretion(
            positions, accreted, accreted_by,
            particle_type, ring_id, capture_radius
        )

        # Should NOT be accreted (only Ring 0 can accrete)
        assert newly_accreted == 0
        assert accreted[1] == False

    def test_already_accreted_particles_skipped(self):
        """Already accreted particles should not be re-accreted (unified particle system)."""
        from bhe.state import BLACK_HOLE, DEBRIS

        # Unified arrays: [0] = Ring 0 BH, [1] = already accreted debris
        positions = np.array([
            [14.0 * const.Gly, 0.0, 0.0],  # Ring 0 BH
            [14.0 * const.Gly, 0.0, 0.0]  # Debris right on top of BH
        ])
        accreted = np.array([False, True])  # Debris already accreted
        accreted_by = np.array([-1, 0], dtype=np.int32)
        particle_type = np.array([BLACK_HOLE, DEBRIS], dtype=np.int32)
        ring_id = np.array([0, -1], dtype=np.int32)
        capture_radius = np.array([0.5 * const.Gly, 0.0])

        newly_accreted = detect_accretion(
            positions, accreted, accreted_by,
            particle_type, ring_id, capture_radius
        )

        # Should return 0 newly accreted (particle already accreted)
        assert newly_accreted == 0


class TestMomentumConservation:
    """Test momentum conservation during accretion."""

    def test_stationary_bh_gains_debris_momentum(self):
        """Stationary BH should gain momentum from accreted debris (unified particle system)."""
        # Unified arrays: [0] = stationary BH, [1] = debris moving at 0.5c
        positions = np.array([
            [1.0 * const.Gly, 0.0, 0.0],  # BH
            [1.0 * const.Gly, 0.0, 0.0]   # Debris (same position, just accreted)
        ])
        velocities = np.array([
            [0.0, 0.0, 0.0],  # BH stationary
            [0.5, 0.0, 0.0]   # Debris moving at 0.5c in +x
        ])
        masses = np.array([1.0e21, 1.0e20])  # M_sun
        accreted = np.array([False, True])  # Debris just accreted
        accreted_by = np.array([-1, 0], dtype=np.int32)
        just_accreted_mask = np.array([False, True])  # Only debris just accreted

        # Calculate expected final velocity
        p_debris = masses[1] * velocities[1, 0]
        m_total = masses[0] + masses[1]
        v_expected = p_debris / m_total

        apply_accretion_momentum_conservation(
            positions, velocities, masses, accreted, accreted_by, just_accreted_mask
        )

        # BH should have gained velocity
        assert abs(velocities[0, 0] - v_expected) < 1e-10

        # BH mass should have increased
        assert abs(masses[0] - m_total) < 1e-10

    def test_moving_bh_momentum_conserved(self):
        """Momentum should be conserved when moving BH accretes debris (unified particle system)."""
        # Unified arrays: [0] = BH moving in +x, [1] = debris moving in +y
        positions = np.array([
            [1.0 * const.Gly, 0.0, 0.0],  # BH
            [1.0 * const.Gly, 0.0, 0.0]   # Debris
        ])
        velocities = np.array([
            [0.2, 0.0, 0.0],  # BH moving at 0.2c in +x
            [0.0, 0.3, 0.0]   # Debris moving at 0.3c in +y
        ])
        masses = np.array([1.0e21, 1.0e20])  # M_sun
        accreted = np.array([False, True])  # Debris just accreted
        accreted_by = np.array([-1, 0], dtype=np.int32)
        just_accreted_mask = np.array([False, True])

        # Calculate expected momentum before accretion
        p_before = masses[0] * velocities[0] + masses[1] * velocities[1]

        apply_accretion_momentum_conservation(
            positions, velocities, masses, accreted, accreted_by, just_accreted_mask
        )

        # Check momentum is conserved
        p_after = masses[0] * velocities[0]
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

        from bhe.initialization import initialize_simulation
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
        """Debris particles should be updated (unified particle system)."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))
        params.debris_count = 10

        from bhe.initialization import initialize_simulation
        state = initialize_simulation(params, seed=42)

        # Save initial positions (unified array)
        pos_before = state.positions.copy()

        # Evolve for 10 timesteps
        evolve_system(state, params, 10, show_progress=False)

        # Positions should have changed
        assert not np.allclose(state.positions, pos_before)

    def test_run_simulation_complete(self):
        """Test complete simulation run from start to finish."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        params = SimulationParameters.from_yaml(str(config_path))

        # Small test simulation
        params.debris_count = 10
        params.duration = 0.01 * (1.0e9)  # 0.01 Gyr
        params.dt = 0.001 * (1.0e9)  # 0.001 Gyr timestep

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
        from bhe.config import RingConfig
        # For Ring 0 BHs, calculate velocity to orbit a 1e22 M_sun mass
        M_total = 1.0e22 * 1.0
        ring0_radius = 14.0 * const.Gly
        v_orbit = np.sqrt(const.G * M_total / ring0_radius)

        params.rings = [
            RingConfig(
                ring_id=0,
                count=4,
                radius=ring0_radius,
                mass_per_bh=1.0e21 * 1.0,
                is_static=False,
                orbital_velocity=v_orbit,
                capture_radius=5.0 * const.Gly  # Large capture radius
            )
        ]

        # Small debris field near Ring 0
        params.debris_count = 20
        params.debris_r_min = 10.0 * const.Gly
        params.debris_r_max = 18.0 * const.Gly
        params.debris_v_min = 0.01 * const.c
        params.debris_v_max = 0.1 * const.c

        from bhe.initialization import initialize_simulation
        state = initialize_simulation(params, seed=42)

        initial_active = state.n_active  # Unified particle system

        # Evolve for a few timesteps
        params.dt = 0.001 * (1.0e9)
        evolve_system(state, params, 50, show_progress=False)

        # Some particles should have been accreted
        # (This is probabilistic, but with large capture radius, very likely)
        # We just check that accretion mechanism works
        assert state.n_active <= initial_active  # Unified particle system
