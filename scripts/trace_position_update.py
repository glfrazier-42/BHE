"""
Trace position update step by step without Numba to find the bug.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants as const

# Simplified version of update_all_particles WITHOUT Numba
def update_all_particles_traced(
    debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
    bh_positions, bh_velocities, bh_masses, bh_ring_ids, bh_is_static,
    dt, use_relativistic
):
    N_debris = len(debris_pos)
    N_bh = len(bh_positions)

    print(f"=== PHASE 1: Save initial state ===")
    old_debris_pos = debris_pos.copy()
    old_debris_vel = debris_vel.copy()
    print(f"old_debris_pos[0]: {old_debris_pos[0]}")
    print(f"old_debris_vel[0]: {old_debris_vel[0]}")
    print()

    print(f"=== PHASE 2: Calculate acceleration ===")
    # Simple direct calculation for debris from BH
    r_vec = bh_positions[0] - debris_pos[0]
    r = np.sqrt(np.sum(r_vec**2))
    force_magnitude = const.G * bh_masses[0] / (r**2)
    force = force_magnitude * (r_vec / r)
    accel = force / debris_masses[0]
    print(f"r_vec: {r_vec}")
    print(f"r: {r}")
    print(f"force_magnitude: {force_magnitude}")
    print(f"accel: {accel}")
    print()

    print(f"=== PHASE 3: Half-step velocity kick and position drift ===")
    vel_half = old_debris_vel[0] + 0.5 * accel * dt
    print(f"vel_half = old_vel + 0.5*accel*dt")
    print(f"vel_half = {old_debris_vel[0]} + 0.5*{accel}*{dt}")
    print(f"vel_half = {vel_half}")

    new_pos = old_debris_pos[0] + vel_half * dt
    print(f"new_pos = old_pos + vel_half*dt")
    print(f"new_pos = {old_debris_pos[0]} + {vel_half}*{dt}")
    print(f"new_pos = {new_pos}")
    print()

    print(f"=== PHASE 7: Write back ===")
    print(f"Before writeback: debris_pos[0] = {debris_pos[0]}")
    debris_pos[0] = new_pos
    print(f"After writeback: debris_pos[0] = {debris_pos[0]}")
    debris_vel[0] = vel_half  # Simplified - skipping final kick for clarity
    print()

def main():
    print("=" * 70)
    print("TRACE POSITION UPDATE")
    print("=" * 70)
    print()

    # Solar system scale
    AU_to_m = 1.496e11
    debris_pos = np.array([[1.0 * AU_to_m, 0.0, 0.0]])
    debris_vel = np.array([[0.0, 0.0, 0.0]])
    debris_masses = np.array([3.0e-6 * const.M_sun])
    debris_proper_times = np.array([0.0])
    debris_accreted = np.array([False])

    bh_positions = np.array([[0.0, 0.0, 0.0]])
    bh_masses = np.array([1.0 * const.M_sun])
    bh_velocities = np.array([[0.0, 0.0, 0.0]])
    bh_is_static = np.array([True])
    bh_ring_ids = np.array([0], dtype=np.int32)

    dt = 45000.0

    print(f"Initial debris_pos[0]: {debris_pos[0]}")
    print(f"Initial debris_pos ID: {id(debris_pos)}")
    print()

    update_all_particles_traced(
        debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
        bh_positions, bh_velocities, bh_masses, bh_ring_ids, bh_is_static,
        dt, use_relativistic=False
    )

    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Final debris_pos[0]: {debris_pos[0]}")
    print(f"Position changed: {not np.array_equal(debris_pos[0], [1.0 * AU_to_m, 0.0, 0.0])}")

if __name__ == "__main__":
    main()
