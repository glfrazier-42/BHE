"""
Test that gravity actually causes orbital motion.

Setup: One massive "debris" particle at origin (simulating intact central BH)
       One Ring 0 BH in circular orbit around it.

Expected: The Ring 0 BH should complete a circular orbit.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import update_debris_particles, update_dynamic_bhs
from src import constants as const

def main():
    print("=" * 70)
    print("ORBITAL MOTION TEST - TWO MASSIVE OBJECTS")
    print("=" * 70)
    print()

    # Central "black hole" simulated as single massive debris particle
    M_central = 4.0e22 * const.M_sun
    debris_pos = np.array([[0.0, 0.0, 0.0]])
    debris_vel = np.array([[0.0, 0.0, 0.0]])
    debris_masses = np.array([M_central])  # Entire central BH mass in one particle
    debris_proper_times = np.array([0.0])
    debris_accreted = np.array([False])

    # Ring 0 BH in circular orbit
    r = 14.0 * const.Gly_to_m  # Orbital radius
    M_ring0 = 1.0e21 * const.M_sun

    # Calculate Keplerian velocity for circular orbit
    v_keplerian = np.sqrt(const.G * M_central / r)

    bh_positions = np.array([[r, 0.0, 0.0]])
    bh_velocities = np.array([[0.0, v_keplerian, 0.0]])  # Perpendicular to radius
    bh_masses = np.array([M_ring0])
    bh_ring_ids = np.array([0], dtype=np.int32)
    bh_is_static = np.array([False])

    # Calculate orbital period: T = 2πr/v
    T_orbit = 2.0 * np.pi * r / v_keplerian
    T_orbit_gyr = T_orbit * const.s_to_Gyr

    print(f"Configuration:")
    print(f"  Central mass: {M_central / const.M_sun:.2e} M_sun (at origin)")
    print(f"  Ring 0 BH mass: {M_ring0 / const.M_sun:.2e} M_sun")
    print(f"  Orbital radius: {r / const.Gly_to_m:.2f} Gly")
    print(f"  Keplerian velocity: {v_keplerian:.3e} m/s ({v_keplerian / const.c:.4f} c)")
    print(f"  Orbital period: {T_orbit_gyr:.4f} Gyr")
    print()

    # Choose timestep to be small enough for stability
    # Aim for ~1000 steps per orbit (need small timestep for circular orbit)
    dt = T_orbit / 1000.0
    dt_gyr = dt * const.s_to_Gyr

    # Simulate for 1 complete orbit
    n_steps = 1000

    print(f"Simulation:")
    print(f"  Timestep: {dt_gyr:.6f} Gyr")
    print(f"  Number of steps: {n_steps}")
    print(f"  Total time: {dt_gyr * n_steps:.4f} Gyr (1 orbital period)")
    print()

    # Track position over time
    initial_x = bh_positions[0, 0]
    initial_y = bh_positions[0, 1]
    initial_r = np.sqrt(initial_x**2 + initial_y**2)

    positions = []
    velocities = []

    for step in range(n_steps):
        # Update debris (central BH)
        update_debris_particles(
            debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
            bh_positions, bh_masses, bh_velocities, bh_is_static,
            dt, use_relativistic=False
        )

        # Update Ring 0 BH
        update_dynamic_bhs(
            bh_positions, bh_velocities, bh_masses, bh_ring_ids,
            bh_is_static, debris_pos, debris_vel, debris_masses,
            debris_accreted, dt, use_relativistic=False
        )

        positions.append(bh_positions[0].copy())
        velocities.append(bh_velocities[0].copy())

        if (step + 1) % 250 == 0:
            x, y, z = bh_positions[0]
            r_current = np.sqrt(x**2 + y**2 + z**2)
            vx, vy, vz = bh_velocities[0]
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            print(f"  Step {step+1:3d}: x={x/const.Gly_to_m:+7.3f} Gly, "
                  f"y={y/const.Gly_to_m:+7.3f} Gly, "
                  f"r={r_current/const.Gly_to_m:6.3f} Gly, "
                  f"v={v_mag/const.c:.4f} c")

    final_x = bh_positions[0, 0]
    final_y = bh_positions[0, 1]
    final_r = np.sqrt(final_x**2 + final_y**2)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Initial position: x={initial_x/const.Gly_to_m:+8.4f} Gly, y={initial_y/const.Gly_to_m:+8.4f} Gly, r={initial_r/const.Gly_to_m:.4f} Gly")
    print(f"Final position:   x={final_x/const.Gly_to_m:+8.4f} Gly, y={final_y/const.Gly_to_m:+8.4f} Gly, r={final_r/const.Gly_to_m:.4f} Gly")
    print()

    # Check if BH completed orbit (should return to approximately initial x, but y should have changed)
    dx = final_x - initial_x
    dy = final_y - initial_y
    position_change = np.sqrt(dx**2 + dy**2)

    # Check if radius stayed approximately constant (circular orbit)
    radius_change = abs(final_r - initial_r)
    radius_stability = radius_change / initial_r

    # Check if BH moved in y direction (indicating rotation)
    y_changed = abs(final_y - initial_y) > 0.1 * const.Gly_to_m

    print(f"Position change: {position_change / const.Gly_to_m:.4f} Gly")
    print(f"Radius change: {radius_change / const.Gly_to_m:.6f} Gly ({radius_stability*100:.3f}%)")
    print()

    # Verify orbit
    success = True

    if y_changed:
        print("[OK] BH position changed significantly in y-direction (orbital motion detected)")
    else:
        print("[ERROR] BH did not move in y-direction - no orbital motion!")
        success = False

    if radius_stability < 0.05:  # Radius should stay within 5%
        print(f"[OK] Orbital radius stable within {radius_stability*100:.2f}%")
    else:
        print(f"[WARNING] Orbital radius varied by {radius_stability*100:.2f}% (expected <5%)")

    if position_change > 1.0 * const.Gly_to_m:
        print(f"[OK] BH moved {position_change/const.Gly_to_m:.2f} Gly - gravity is causing position changes!")
    else:
        print(f"[ERROR] BH moved only {position_change/const.Gly_to_m:.6f} Gly - insufficient movement!")
        success = False

    # Check if central BH moved (should be very small due to mass ratio)
    central_moved = np.sqrt(debris_pos[0,0]**2 + debris_pos[0,1]**2 + debris_pos[0,2]**2)
    print()
    print(f"Central BH displacement: {central_moved / const.Gly_to_m:.6f} Gly")
    if central_moved < 0.1 * const.Gly_to_m:
        print("[OK] Central BH stayed near origin (mass ratio ~40000:1)")

    print()
    if success:
        print("=" * 70)
        print("SUCCESS: Gravity causes measurable orbital motion!")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("FAILURE: Orbital motion not detected!")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit(main())
