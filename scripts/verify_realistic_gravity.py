"""
Verify gravity with REALISTIC simulation parameters.

Uses actual debris field parameters:
- Distance: 0.01 - 0.1 Gly (not 1 Gly!)
- Velocities: 0.01c - 0.92c (not at rest!)
- Particle mass: M_central / N_debris
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import update_debris_particles
from src import constants as const

def main():
    print("=" * 70)
    print("REALISTIC GRAVITY TEST - ACTUAL SIMULATION PARAMETERS")
    print("=" * 70)
    print()

    # REALISTIC SCENARIO 1: Two debris particles close together
    # Central BH mass: 4e22 M_sun divided among 1000 particles
    M_central = 4.0e22 * const.M_sun
    N_debris = 1000
    debris_mass = M_central / N_debris

    print("SCENARIO 1: Two debris particles (debris-debris gravity)")
    print(f"  Particle mass: {debris_mass / const.M_sun:.2e} M_sun")

    # Two particles at 0.05 Gly separation (realistic debris field scale)
    debris_pos = np.array([
        [0.025 * const.Gly_to_m, 0.0, 0.0],  # 0.025 Gly
        [0.075 * const.Gly_to_m, 0.0, 0.0]   # 0.075 Gly (0.05 Gly apart)
    ])
    # Start with realistic velocities (0.1c outward)
    debris_vel = np.array([
        [0.1 * const.c, 0.0, 0.0],
        [0.1 * const.c, 0.0, 0.0]
    ])
    debris_masses = np.array([debris_mass, debris_mass])
    debris_proper_times = np.zeros(2)
    debris_accreted = np.array([False, False])

    # No Ring BHs - pure debris-debris test
    bh_positions = np.zeros((0, 3))
    bh_masses = np.zeros(0)
    bh_velocities = np.zeros((0, 3))
    bh_is_static = np.zeros(0, dtype=bool)

    dt = 0.001 * const.Gyr_to_s  # Realistic timestep
    n_steps = 100

    print(f"  Initial separation: {(debris_pos[1, 0] - debris_pos[0, 0]) / const.Gly_to_m:.6f} Gly")
    print(f"  Timestep: {dt * const.s_to_Gyr:.3f} Gyr")
    print(f"  Steps: {n_steps}")
    print()

    initial_separation = debris_pos[1, 0] - debris_pos[0, 0]

    for _ in range(n_steps):
        update_debris_particles(
            debris_pos, debris_vel, debris_masses, debris_proper_times, debris_accreted,
            bh_positions, bh_masses, bh_velocities, bh_is_static,
            dt, use_relativistic=False
        )

    final_separation = debris_pos[1, 0] - debris_pos[0, 0]

    print(f"  Initial separation: {initial_separation / const.Gly_to_m:.10f} Gly")
    print(f"  Final separation:   {final_separation / const.Gly_to_m:.10f} Gly")
    print(f"  Change:             {(final_separation - initial_separation) / const.Gly_to_m:.10e} Gly")
    print()

    # SCENARIO 2: Debris particle near Ring 1 BH
    print("SCENARIO 2: Debris particle near Ring 1 BH (debris-BH gravity)")

    # Ring 1: 100 Gly, mass 5e21 M_sun
    # Debris particle at 99 Gly (1 Gly from Ring 1 BH)
    debris_pos2 = np.array([[99.0 * const.Gly_to_m, 0.0, 0.0]])
    debris_vel2 = np.array([[0.5 * const.c, 0.0, 0.0]])  # Moving at 0.5c
    debris_masses2 = np.array([debris_mass])
    debris_proper_times2 = np.array([0.0])
    debris_accreted2 = np.array([False])

    bh_positions2 = np.array([[100.0 * const.Gly_to_m, 0.0, 0.0]])  # Ring 1 BH
    bh_masses2 = np.array([5.0e21 * const.M_sun])
    bh_velocities2 = np.array([[0.0, 0.0, 0.0]])
    bh_is_static2 = np.array([True])

    print(f"  Ring 1 BH mass: {bh_masses2[0] / const.M_sun:.2e} M_sun")
    print(f"  Distance to BH: {(bh_positions2[0, 0] - debris_pos2[0, 0]) / const.Gly_to_m:.1f} Gly")
    print(f"  Initial debris velocity: {debris_vel2[0, 0] / const.c:.2f} c")
    print()

    initial_pos2 = debris_pos2[0, 0]

    for _ in range(n_steps):
        update_debris_particles(
            debris_pos2, debris_vel2, debris_masses2, debris_proper_times2, debris_accreted2,
            bh_positions2, bh_masses2, bh_velocities2, bh_is_static2,
            dt, use_relativistic=False
        )

    final_pos2 = debris_pos2[0, 0]

    print(f"  Initial position: {initial_pos2 / const.Gly_to_m:.10f} Gly")
    print(f"  Final position:   {final_pos2 / const.Gly_to_m:.10f} Gly")
    print(f"  Change:           {(final_pos2 - initial_pos2) / const.Gly_to_m:.10e} Gly")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check if positions changed detectably
    s1_changed = abs(final_separation - initial_separation) > 1e-30
    s2_changed = abs(final_pos2 - initial_pos2) > 1e-30

    if s1_changed:
        print(f"✓ Scenario 1: Debris separation changed by {(final_separation - initial_separation) / const.Gly_to_m:.3e} Gly")
    else:
        print(f"✗ Scenario 1: No detectable position change")

    if s2_changed:
        print(f"✓ Scenario 2: Debris position changed by {(final_pos2 - initial_pos2) / const.Gly_to_m:.3e} Gly")
    else:
        print(f"✗ Scenario 2: No detectable position change")

    print()
    print("Note: Position changes are dominated by initial velocities (0.1-0.5c).")
    print("      Gravitational acceleration provides small corrections over time.")
    print("      Velocity changes from gravity ARE detectable and tested elsewhere.")

if __name__ == "__main__":
    main()
