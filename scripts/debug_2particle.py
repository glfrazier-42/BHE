"""
Debug script to test 2-particle gravity.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import update_debris_particles
from src import constants as const

def main():
    print("=" * 70)
    print("2-PARTICLE GRAVITY DEBUG")
    print("=" * 70)
    print()

    # Two debris particles: one massive at origin, one light at 1 Gly
    debris_pos = np.array([
        [0.0, 0.0, 0.0],  # Massive particle at origin
        [1.0 * const.Gly_to_m, 0.0, 0.0]  # Light particle at 1 Gly
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

    dt = 0.001 * const.Gyr_to_s

    print(f"Initial state:")
    print(f"  Particle 0 (massive): pos = {debris_pos[0] / const.Gly_to_m} Gly, mass = {debris_masses[0] / const.M_sun:.2e} M_sun")
    print(f"  Particle 1 (light):   pos = {debris_pos[1] / const.Gly_to_m} Gly, mass = {debris_masses[1] / const.M_sun:.2e} M_sun")
    print(f"  Timestep: {dt * const.s_to_Gyr:.6f} Gyr")
    print()

    # Calculate expected acceleration manually
    r = debris_pos[1, 0] - debris_pos[0, 0]  # Distance
    r_squared = r * r
    force_on_1 = -const.G * debris_masses[0] / r_squared  # Negative because toward origin
    accel_on_1 = force_on_1 / debris_masses[1]
    v_expected = accel_on_1 * dt
    x_expected = debris_pos[1, 0] + v_expected * dt

    print(f"Expected (manual calculation for particle 1):")
    print(f"  Distance r = {r / const.Gly_to_m:.6f} Gly")
    print(f"  Force = {force_on_1:.6e} N (toward origin)")
    print(f"  Accel = {accel_on_1:.6e} m/s²")
    print(f"  Delta v = {v_expected:.6e} m/s")
    print(f"  Delta x = {(x_expected - debris_pos[1, 0]) / const.Gly_to_m:.6e} Gly")
    print(f"  New pos = {x_expected / const.Gly_to_m:.10f} Gly")
    print()

    # Save initial values
    pos_before = debris_pos.copy()
    vel_before = debris_vel.copy()

    # Update for one timestep
    update_debris_particles(
        debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
        bh_positions, bh_masses, bh_velocities, bh_is_static,
        dt, use_relativistic=False
    )

    print(f"After update:")
    print(f"  Particle 0: pos = {debris_pos[0] / const.Gly_to_m} Gly, vel = {debris_vel[0]} m/s")
    print(f"  Particle 1: pos = {debris_pos[1] / const.Gly_to_m} Gly, vel = {debris_vel[1]} m/s")
    print()

    print(f"Changes:")
    print(f"  Particle 0 Delta pos: {(debris_pos[0] - pos_before[0]) / const.Gly_to_m} Gly")
    print(f"  Particle 0 Delta vel: {debris_vel[0] - vel_before[0]} m/s")
    print(f"  Particle 1 Delta pos: {(debris_pos[1] - pos_before[1]) / const.Gly_to_m} Gly")
    print(f"  Particle 1 Delta vel: {debris_vel[1] - vel_before[1]} m/s")
    print()

    # Check if particles gained velocity (which is what we can detect in one timestep)
    vel_changed = not np.allclose(debris_vel, vel_before, atol=0.0)

    if not vel_changed:
        print("[ERROR] Particles did NOT gain velocity!")
    else:
        print("[OK] Particles gained velocity as expected (debris-debris gravity works!)")
        print(f"  Particle 0 pulled toward particle 1 (v > 0): {debris_vel[0, 0] > 0}")
        print(f"  Particle 1 pulled toward particle 0 (v < 0): {debris_vel[1, 0] < 0}")


if __name__ == "__main__":
    main()
