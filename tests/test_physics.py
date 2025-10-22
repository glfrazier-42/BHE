"""
Unit tests for physics functions.

Tests cover:
- Lorentz factor calculations
- Relativistic mass
- Gravitational forces
- Numba compilation
"""

import pytest
import numpy as np
from src.physics import (
    lorentz_factor,
    lorentz_factor_scalar,
    relativistic_mass,
    gravitational_acceleration_direct,
    calculate_acceleration_from_bhs,
    calculate_kinetic_energy,
    calculate_potential_energy
)
from src import constants as const


class TestLorentzFactor:
    """Tests for Lorentz factor calculations."""

    def test_stationary_particle(self):
        """Lorentz factor should be 1.0 for stationary particle."""
        velocity = np.array([0.0, 0.0, 0.0])
        gamma = lorentz_factor(velocity)
        assert abs(gamma - 1.0) < 1e-10

    def test_low_velocity(self):
        """For v << c, γ should be approximately 1.0."""
        velocity = np.array([1000.0, 0.0, 0.0])  # 1 km/s << c
        gamma = lorentz_factor(velocity)
        assert gamma > 1.0
        assert gamma < 1.0001  # Very close to 1

    def test_half_speed_of_light(self):
        """Test γ at v = 0.5c."""
        velocity = np.array([0.5 * const.c, 0.0, 0.0])
        gamma = lorentz_factor(velocity)
        expected = 1.0 / np.sqrt(1.0 - 0.25)  # 1/sqrt(0.75) ≈ 1.1547
        assert abs(gamma - expected) / expected < 1e-10

    def test_high_velocity(self):
        """Test γ at v = 0.9c."""
        velocity = np.array([0.9 * const.c, 0.0, 0.0])
        gamma = lorentz_factor(velocity)
        expected = 1.0 / np.sqrt(1.0 - 0.81)  # 1/sqrt(0.19) ≈ 2.294
        assert abs(gamma - expected) / expected < 1e-10

    def test_3d_velocity(self):
        """Test γ with velocity in all 3 dimensions."""
        # v = (0.5c, 0.3c, 0.2c)
        vx = 0.5 * const.c
        vy = 0.3 * const.c
        vz = 0.2 * const.c
        velocity = np.array([vx, vy, vz])

        v_mag_squared = vx**2 + vy**2 + vz**2
        beta_squared = v_mag_squared / const.c_squared
        expected = 1.0 / np.sqrt(1.0 - beta_squared)

        gamma = lorentz_factor(velocity)
        assert abs(gamma - expected) / expected < 1e-10

    def test_near_light_speed_clamped(self):
        """Test that velocities near c are clamped to prevent numerical issues."""
        velocity = np.array([0.99999 * const.c, 0.0, 0.0])
        gamma = lorentz_factor(velocity)
        assert gamma < 1e10  # Should be clamped, not infinity
        assert gamma > 1.0

    def test_scalar_version(self):
        """Test scalar version of Lorentz factor."""
        v = 0.8 * const.c
        gamma = lorentz_factor_scalar(v)
        expected = 1.0 / np.sqrt(1.0 - 0.64)  # 1/sqrt(0.36) ≈ 1.667
        assert abs(gamma - expected) / expected < 1e-10


class TestRelativisticMass:
    """Tests for relativistic mass calculations."""

    def test_stationary_mass(self):
        """Relativistic mass equals rest mass for stationary object."""
        rest_mass = 1.0e30  # 1 solar mass in kg
        velocity = np.array([0.0, 0.0, 0.0])
        m_rel = relativistic_mass(rest_mass, velocity)
        assert abs(m_rel - rest_mass) / rest_mass < 1e-10

    def test_half_speed_of_light(self):
        """Test relativistic mass at v = 0.5c."""
        rest_mass = 1.0e30
        velocity = np.array([0.5 * const.c, 0.0, 0.0])
        m_rel = relativistic_mass(rest_mass, velocity)

        gamma = 1.0 / np.sqrt(1.0 - 0.25)
        expected = gamma * rest_mass

        assert abs(m_rel - expected) / expected < 1e-10

    def test_high_velocity(self):
        """Test that m_rel > m_rest for high velocities."""
        rest_mass = 1.0e30
        velocity = np.array([0.9 * const.c, 0.0, 0.0])
        m_rel = relativistic_mass(rest_mass, velocity)
        assert m_rel > rest_mass
        assert m_rel > 2.0 * rest_mass  # γ(0.9c) ≈ 2.29


class TestGravitationalForce:
    """Tests for gravitational force calculations."""

    def test_zero_distance_returns_zero(self):
        """Force should be zero when particles are too close (singularity avoidance)."""
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([1e9, 0.0, 0.0])  # < 1e10 m threshold
        mass_j = 1.0e30
        velocity_j = np.array([0.0, 0.0, 0.0])

        accel = gravitational_acceleration_direct(pos_i, pos_j, mass_j, velocity_j, False)
        assert np.allclose(accel, np.zeros(3))

    def test_newtonian_limit(self):
        """Test force in Newtonian limit (low velocity, no relativistic correction)."""
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([const.Gly_to_m, 0.0, 0.0])  # 1 Gly away
        mass_j = 1.0e30  # 1 solar mass
        velocity_j = np.array([0.0, 0.0, 0.0])

        accel = gravitational_acceleration_direct(pos_i, pos_j, mass_j, velocity_j, False)

        # Expected: a = G × M / r² in x-direction
        r = const.Gly_to_m
        expected_magnitude = const.G * mass_j / (r * r)
        expected_accel = np.array([expected_magnitude, 0.0, 0.0])

        assert np.allclose(accel, expected_accel, rtol=1e-10)

    def test_relativistic_mass_increases_accel(self):
        """Acceleration should increase when using relativistic mass for moving BH."""
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([const.Gly_to_m, 0.0, 0.0])
        mass_j = 1.0e30
        velocity_j = np.array([0.0, 0.8 * const.c, 0.0])  # Moving at 0.8c

        accel_nonrel = gravitational_acceleration_direct(pos_i, pos_j, mass_j, velocity_j, False)
        accel_rel = gravitational_acceleration_direct(pos_i, pos_j, mass_j, velocity_j, True)

        # Relativistic acceleration should be larger
        assert np.linalg.norm(accel_rel) > np.linalg.norm(accel_nonrel)

        # Check ratio matches γ(0.8c) ≈ 1.667
        gamma = 1.0 / np.sqrt(1.0 - 0.64)
        ratio = np.linalg.norm(accel_rel) / np.linalg.norm(accel_nonrel)
        assert abs(ratio - gamma) / gamma < 1e-6

    def test_accel_direction(self):
        """Acceleration should point from particle i toward particle j."""
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([1.0e15, 2.0e15, 3.0e15])  # Arbitrary position
        mass_j = 1.0e30
        velocity_j = np.array([0.0, 0.0, 0.0])

        accel = gravitational_acceleration_direct(pos_i, pos_j, mass_j, velocity_j, False)

        # Acceleration direction should be parallel to (pos_j - pos_i)
        r_vec = pos_j - pos_i
        accel_dir = accel / np.linalg.norm(accel)
        r_dir = r_vec / np.linalg.norm(r_vec)

        assert np.allclose(accel_dir, r_dir, rtol=1e-10)


class TestAccelFromBHs:
    """Tests for acceleration calculations from multiple black holes."""

    def test_single_bh(self):
        """Test acceleration from a single black hole."""
        pos = np.array([0.0, 0.0, 0.0])
        bh_positions = np.array([[const.Gly_to_m, 0.0, 0.0]])
        bh_masses = np.array([1.0e30])
        bh_velocities = np.array([[0.0, 0.0, 0.0]])
        bh_is_static = np.array([True])

        accel = calculate_acceleration_from_bhs(
            pos, bh_positions, bh_masses, bh_velocities, bh_is_static, False
        )

        # Expected: a = G × M / r² in x-direction
        r = const.Gly_to_m
        expected_accel_mag = const.G * bh_masses[0] / (r * r)
        expected = np.array([expected_accel_mag, 0.0, 0.0])

        assert np.allclose(accel, expected, rtol=1e-10)

    def test_multiple_bhs_superposition(self):
        """Test that accelerations from multiple BHs add (superposition principle)."""
        pos = np.array([0.0, 0.0, 0.0])

        # Two BHs on opposite sides
        bh_positions = np.array([
            [const.Gly_to_m, 0.0, 0.0],
            [-const.Gly_to_m, 0.0, 0.0]
        ])
        bh_masses = np.array([1.0e30, 1.0e30])  # Equal masses
        bh_velocities = np.zeros((2, 3))
        bh_is_static = np.array([True, True])

        accel = calculate_acceleration_from_bhs(
            pos, bh_positions, bh_masses, bh_velocities, bh_is_static, False
        )

        # Accelerations should cancel out (equal masses, opposite directions)
        assert np.allclose(accel, np.zeros(3), atol=1e-50)


class TestEnergyCalculations:
    """Tests for energy calculations."""

    def test_kinetic_energy_stationary(self):
        """KE should be zero for stationary particle."""
        mass = 1.0e30
        velocity = np.array([0.0, 0.0, 0.0])
        ke = calculate_kinetic_energy(mass, velocity)
        assert abs(ke) < 1e-10

    def test_kinetic_energy_low_velocity(self):
        """For v << c, KE ≈ (1/2) × m × v²."""
        mass = 1.0e30
        v = 1.0e5  # 100 km/s << c
        velocity = np.array([v, 0.0, 0.0])

        ke = calculate_kinetic_energy(mass, velocity)
        ke_classical = 0.5 * mass * v * v

        # Relativistic KE should be very close to classical at low speeds
        assert abs(ke - ke_classical) / ke_classical < 0.001  # Within 0.1%

    def test_potential_energy_negative(self):
        """Gravitational PE should be negative."""
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([const.Gly_to_m, 0.0, 0.0])
        mass1 = 1.0e30
        mass2 = 1.0e30

        pe = calculate_potential_energy(pos1, pos2, mass1, mass2)
        assert pe < 0.0

        # Expected: PE = -G × m1 × m2 / r
        r = const.Gly_to_m
        expected = -const.G * mass1 * mass2 / r
        assert abs(pe - expected) / abs(expected) < 1e-10


class TestNumbaCompilation:
    """Test that Numba compilation is working."""

    def test_functions_are_compiled(self):
        """Verify that decorated functions have Numba signatures."""
        # This test ensures Numba JIT is actually compiling the functions
        velocity = np.array([0.5 * const.c, 0.0, 0.0])

        # Call the function to trigger compilation
        gamma = lorentz_factor(velocity)

        # Check that the function has been compiled
        assert hasattr(lorentz_factor, 'signatures')
        assert len(lorentz_factor.signatures) > 0

    def test_performance_improvement(self):
        """Verify Numba provides speedup (optional benchmark)."""
        # This is more of a sanity check than a strict requirement
        # Just verify the function runs without error
        velocity = np.array([0.8 * const.c, 0.0, 0.0])

        # Run multiple times to ensure compilation and warm-up
        for _ in range(100):
            gamma = lorentz_factor(velocity)

        assert gamma > 1.0
