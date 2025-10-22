"""
Quick test of a small simulation.

Runs a tiny simulation (10 particles, 100 timesteps) to verify the evolution engine works.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulationParameters
from src.evolution import run_simulation
from src import constants as const


def main():
    print("=" * 70)
    print("SMALL SIMULATION TEST")
    print("=" * 70)
    print()

    # Load baseline config
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    # Override to small values for quick test
    params.debris_count = 10
    params.duration = 0.1 * const.Gyr_to_s  # 0.1 Gyr
    params.dt = 0.001 * const.Gyr_to_s  # 0.001 Gyr timestep

    print(f"Configuration:")
    print(f"  Debris particles: {params.debris_count}")
    print(f"  Duration: {params.duration * const.s_to_Gyr:.3f} Gyr")
    print(f"  Timestep: {params.dt * const.s_to_Gyr:.6f} Gyr")
    print(f"  Number of steps: {int(params.duration / params.dt)}")
    print(f"  Central BH mass: {params.M_central * const.kg_to_solar_mass:.2e} M_sun")
    print()

    # Run simulation
    state, stats = run_simulation(params, seed=42, show_progress=True)

    # Report results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Final time: {state.time * const.s_to_Gyr:.3f} Gyr")
    print(f"  Timesteps completed: {state.timestep_count}")
    print(f"  Active debris: {state.n_debris_active}/{state.n_debris}")
    print(f"  Accreted debris: {state.n_debris_accreted}/{state.n_debris}")
    print()

    # Show some particle details
    print("Sample debris particle states:")
    print(f"  {'ID':<4} {'Position (Gly)':<30} {'Velocity (frac c)':<30} {'Accreted':<10}")
    print(f"  {'-'*4} {'-'*30} {'-'*30} {'-'*10}")

    for i in range(min(5, state.n_debris)):
        pos_gly = state.debris_positions[i] / const.Gly_to_m
        vel_frac_c = state.debris_velocities[i] / const.c
        accreted = "Yes" if state.debris_accreted[i] else "No"

        print(f"  {i:<4} "
              f"[{pos_gly[0]:+7.2f}, {pos_gly[1]:+7.2f}, {pos_gly[2]:+7.2f}]  "
              f"[{vel_frac_c[0]:+7.4f}, {vel_frac_c[1]:+7.4f}, {vel_frac_c[2]:+7.4f}]  "
              f"{accreted:<10}")

    print()
    print("[OK] Small simulation completed successfully!")


if __name__ == "__main__":
    main()
