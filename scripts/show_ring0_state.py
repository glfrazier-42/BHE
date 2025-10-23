"""
Display Ring 0 black hole positions and velocities.

This script initializes a simulation with Ring 0 enabled and shows
the position and velocity vectors for each Ring 0 black hole.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulationParameters
from src.initialization import initialize_simulation
from src import constants as const


def main():
    # Load baseline config
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    # Check if Ring 0 is enabled
    if params.total_bh_count == 0 or (len(params.rings) > 0 and params.rings[0].ring_id != 0):
        print("WARNING: Ring 0 is disabled in baseline_config.yaml (count=0)")
        print("To see Ring 0 state, set ring_0.count to 4-8 in the config.")
        print()

        # Show what Ring 0 would look like if enabled
        print("If Ring 0 were enabled with 4 BHs:")
        r = 14.0 * const.Gly_to_m
        v = np.sqrt(const.G * params.M_central / r)

        print(f"  Radius: {r / const.Gly_to_m:.1f} Gly = {r:.3e} meters")
        print(f"  Keplerian velocity: {v / const.c:.4f}c = {v:.3e} m/s")
        print()
        print("Positions (meters):")
        for i in range(4):
            theta = 2.0 * np.pi * i / 4
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 0.0
            print(f"  BH {i}: [{x:+.3e}, {y:+.3e}, {z:+.3e}]")

        print()
        print("Velocities (m/s):")
        for i in range(4):
            theta = 2.0 * np.pi * i / 4
            vx = -v * np.sin(theta)
            vy = v * np.cos(theta)
            vz = 0.0
            print(f"  BH {i}: [{vx:+.3e}, {vy:+.3e}, {vz:+.3e}]")

        return

    # Initialize simulation
    state = initialize_simulation(params, seed=42)

    # Find Ring 0 BHs
    ring0_mask = state.bh_ring_ids == 0
    n_ring0 = np.sum(ring0_mask)

    if n_ring0 == 0:
        print("No Ring 0 black holes found.")
        return

    print("=" * 70)
    print(f"RING 0 BLACK HOLES: {n_ring0} BHs")
    print("=" * 70)
    print()

    # Get Ring 0 parameters
    ring0_config = params.rings[0]
    print(f"Configuration:")
    print(f"  Radius: {ring0_config.radius / const.Gly_to_m:.2f} Gly")
    print(f"  Mass per BH: {ring0_config.mass_per_bh / const.M_sun:.2e} M_sun")
    print(f"  Velocity mode: {params.rings[0].is_static and 'static' or 'dynamic'}")
    print(f"  Capture radius: {ring0_config.capture_radius / const.Gly_to_m:.2f} Gly")
    print()

    # Show each Ring 0 BH
    ring0_indices = np.where(ring0_mask)[0]

    for idx in ring0_indices:
        pos = state.bh_positions[idx]
        vel = state.bh_velocities[idx]
        mass = state.bh_masses[idx]

        # Calculate derived quantities
        r_magnitude = np.linalg.norm(pos)
        v_magnitude = np.linalg.norm(vel)

        # Check if velocity is perpendicular to position
        dot_product = np.dot(pos, vel)
        angle_deg = np.degrees(np.arccos(dot_product / (r_magnitude * v_magnitude + 1e-100)))

        print(f"BH {idx} (Ring 0, ID {idx - ring0_indices[0]}):")
        print(f"  Position [m]:     [{pos[0]:+.6e}, {pos[1]:+.6e}, {pos[2]:+.6e}]")
        print(f"  Position [Gly]:   [{pos[0]/const.Gly_to_m:+8.3f}, "
              f"{pos[1]/const.Gly_to_m:+8.3f}, {pos[2]/const.Gly_to_m:+8.3f}]")
        print(f"  |r| = {r_magnitude / const.Gly_to_m:.3f} Gly")
        print()
        print(f"  Velocity [m/s]:   [{vel[0]:+.6e}, {vel[1]:+.6e}, {vel[2]:+.6e}]")
        print(f"  Velocity [frac c]:[{vel[0]/const.c:+8.5f}, "
              f"{vel[1]/const.c:+8.5f}, {vel[2]/const.c:+8.5f}]")
        print(f"  |v| = {v_magnitude:.6e} m/s = {v_magnitude / const.c:.5f}c")
        print()
        print(f"  Mass: {mass / const.M_sun:.2e} M_sun")
        print(f"  Angle between r and v: {angle_deg:.2f}° (should be ~90° for circular orbit)")
        print()

    # Summary statistics
    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)

    velocities = state.bh_velocities[ring0_mask]
    v_magnitudes = np.linalg.norm(velocities, axis=1)

    print(f"Number of Ring 0 BHs: {n_ring0}")
    print(f"Velocity range: {np.min(v_magnitudes) / const.c:.5f}c to {np.max(v_magnitudes) / const.c:.5f}c")
    print(f"Mean velocity: {np.mean(v_magnitudes) / const.c:.5f}c")
    print()

    # Calculate total momentum
    total_momentum = np.sum(state.bh_masses[ring0_mask, np.newaxis] * velocities, axis=0)
    momentum_magnitude = np.linalg.norm(total_momentum)

    print(f"Total Ring 0 momentum: {momentum_magnitude:.3e} kg·m/s")
    print(f"  (should be ~0 for symmetric circular orbit)")
    print()

    # Calculate total angular momentum
    positions = state.bh_positions[ring0_mask]
    angular_momenta = np.cross(positions, state.bh_masses[ring0_mask, np.newaxis] * velocities)
    total_L = np.sum(angular_momenta, axis=0)
    L_magnitude = np.linalg.norm(total_L)

    print(f"Total Ring 0 angular momentum: {L_magnitude:.3e} kg·m²/s")
    print(f"  Direction: [{total_L[0]/L_magnitude:+.3f}, "
          f"{total_L[1]/L_magnitude:+.3f}, {total_L[2]/L_magnitude:+.3f}]")
    print(f"  (should point in +z or -z direction for planar orbit)")


if __name__ == "__main__":
    main()
