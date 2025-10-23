"""
Verify that gravity actually causes particles to move over time.

This script runs a long simulation to confirm that particles
starting at rest will eventually move measurably due to gravity.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import update_debris_particles
from src import constants as const

def main():
    print("=" * 70)
    print("VERIFYING GRAVITY CAUSES POSITION CHANGES")
    print("=" * 70)
    print()

    # Single debris particle at rest, 1 Gly from massive BH at origin
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

    # Large timestep and many iterations
    dt = 0.1 * const.Gyr_to_s  # 0.1 Gyr per step
    n_steps = 10000  # Total: 1000 Gyr

    print(f"Configuration:")
    print(f"  Initial position: {debris_pos[0, 0] / const.Gly_to_m:.6f} Gly")
    print(f"  Initial velocity: {debris_vel[0, 0]:.6e} m/s")
    print(f"  BH mass: {bh_masses[0] / const.M_sun:.2e} M_sun")
    print(f"  Timestep: {dt * const.s_to_Gyr:.3f} Gyr")
    print(f"  Number of steps: {n_steps}")
    print(f"  Total time: {dt * const.s_to_Gyr * n_steps:.1f} Gyr")
    print()

    initial_position = debris_pos[0, 0]

    print("Evolving...")
    for step in range(n_steps):
        update_debris_particles(
            debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
            bh_positions, bh_masses, bh_velocities, bh_is_static,
            dt, use_relativistic=False
        )

        if (step + 1) % 1000 == 0:
            distance = debris_pos[0, 0] / const.Gly_to_m
            velocity = debris_vel[0, 0]
            print(f"  Step {step+1:5d}: pos = {distance:.6f} Gly, vel = {velocity:.6e} m/s")

    final_position = debris_pos[0, 0]
    final_velocity = debris_vel[0, 0]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Initial position: {initial_position / const.Gly_to_m:.10f} Gly")
    print(f"Final position:   {final_position / const.Gly_to_m:.10f} Gly")
    print(f"Position change:  {(initial_position - final_position) / const.Gly_to_m:.10f} Gly")
    print(f"Final velocity:   {final_velocity:.6e} m/s ({final_velocity / const.c:.6f} c)")
    print()

    # Check if particle moved
    position_changed = final_position < initial_position
    distance_moved_gly = (initial_position - final_position) / const.Gly_to_m

    if position_changed and distance_moved_gly > 0.001:  # Moved more than 0.001 Gly
        print(f"[OK] Particle moved {distance_moved_gly:.6f} Gly toward BH due to gravity!")
        print(f"[OK] Gravity is working correctly - position changes are measurable.")
        return 0
    else:
        print(f"[ERROR] Particle did not move measurably!")
        print(f"[ERROR] Distance moved: {distance_moved_gly:.10f} Gly")
        return 1

if __name__ == "__main__":
    exit(main())
