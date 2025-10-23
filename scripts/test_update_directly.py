"""
Test update_all_particles directly with print statements.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import update_all_particles
from src import constants as const

def main():
    print("=" * 70)
    print("DIRECT TEST OF update_all_particles")
    print("=" * 70)
    print()

    # Single debris particle at rest, 0.01 Gly from Ring BH at origin
    debris_pos = np.array([[0.01 * const.Gly_to_m, 0.0, 0.0]])
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

    dt = 0.001 * const.Gyr_to_s

    print(f"Before update:")
    print(f"  debris_pos: {debris_pos}")
    print(f"  debris_vel: {debris_vel}")
    print(f"  debris_pos ID: {id(debris_pos)}")
    print()

    # Call update
    update_all_particles(
        debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
        bh_positions, bh_velocities, bh_masses, bh_ring_ids, bh_is_static,
        dt, use_relativistic=False
    )

    print(f"After 1 update:")
    print(f"  debris_pos: {debris_pos}")
    print(f"  debris_vel: {debris_vel}")
    print(f"  debris_pos ID: {id(debris_pos)}")
    print()

    # Check if velocity changed
    if debris_vel[0, 0] != 0.0:
        print(f"[OK] Velocity changed: {debris_vel[0, 0]} m/s = {debris_vel[0, 0]/const.c} c")
    else:
        print("[FAIL] Velocity did NOT change!")

    # Check if position changed
    if debris_pos[0, 0] != 0.01 * const.Gly_to_m:
        print(f"[OK] Position changed: {debris_pos[0, 0]} m = {debris_pos[0, 0]/const.Gly_to_m} Gly")
    else:
        print("[FAIL] Position did NOT change!")

if __name__ == "__main__":
    main()
