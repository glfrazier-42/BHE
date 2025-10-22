"""
Debug the unified update_all_particles function.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import update_all_particles
from src import constants as const

def main():
    print("=" * 70)
    print("DEBUG: update_all_particles")
    print("=" * 70)
    print()

    # Single debris particle at rest, 100 AU from Ring BH at origin
    AU_to_m = 1.496e11  # meters per astronomical unit
    debris_pos = np.array([[100.0 * AU_to_m, 0.0, 0.0]])
    debris_vel = np.array([[0.0, 0.0, 0.0]])
    debris_masses = np.array([1.0e20 * const.M_sun])
    debris_proper_times = np.array([0.0])
    debris_accreted = np.array([False])

    # Single massive Ring BH at origin
    bh_positions = np.array([[0.0, 0.0, 0.0]])
    bh_masses = np.array([1.0e22 * const.M_sun])
    bh_velocities = np.array([[0.0, 0.0, 0.0]])
    bh_is_static = np.array([True])
    bh_ring_ids = np.array([0], dtype=np.int32)

    dt = 1.0 * 24 * 3600  # 1 day in seconds

    print(f"Initial state:")
    print(f"  Debris position: {debris_pos[0, 0] / AU_to_m} AU")
    print(f"  Debris velocity: {debris_vel[0] / const.c} c")
    print(f"  BH position: {bh_positions[0] / const.Gly_to_m} Gly")
    print(f"  Timestep: {dt / (24*3600)} days")
    print()

    # Single timestep
    print("Calling update_all_particles...")
    update_all_particles(
        debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
        bh_positions, bh_velocities, bh_masses, bh_ring_ids, bh_is_static,
        dt, use_relativistic=False
    )

    initial_x = debris_pos[0, 0]
    print(f"After 1 timestep:")
    print(f"  Debris position: {debris_pos[0, 0] / AU_to_m} AU")
    print(f"  Debris velocity: {debris_vel[0] / const.c} c")
    print(f"  Position change: {(debris_pos[0, 0] - initial_x) / AU_to_m} AU")
    print()

    # Do 999 more timesteps (1000 total)
    for i in range(999):
        update_all_particles(
            debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
            bh_positions, bh_velocities, bh_masses, bh_ring_ids, bh_is_static,
            dt, use_relativistic=False
        )

    print(f"After 1000 total timesteps:")
    print(f"  Debris position: {debris_pos[0, 0] / AU_to_m} AU")
    print(f"  Debris velocity: {debris_vel[0] / const.c} c")
    print(f"  Position change: {(debris_pos[0, 0] - initial_x) / AU_to_m} AU")
    print(f"  Position change: {(debris_pos[0, 0] - initial_x)} m")
    print()

    # Expected behavior:
    # - Velocity should be negative (toward BH at origin)
    # - Position x should decrease (moving toward origin)

    if debris_vel[0, 0] < 0:
        print("[OK] Velocity is negative (toward BH)")
    else:
        print("[FAIL] Velocity is NOT negative!")

    if debris_pos[0, 0] < initial_x:
        print("[OK] Position moved toward BH")
    else:
        print("[FAIL] Position did NOT move toward BH!")

if __name__ == "__main__":
    main()
