"""
Demonstrate runtime diagnostics for detecting timestep problems.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulationParameters
from src.initialization import initialize_simulation
from src.diagnostics import check_timestep_against_courant, calculate_total_energy
from src import constants as const

def main():
    print("=" * 70)
    print("RUNTIME DIAGNOSTICS DEMONSTRATION")
    print("=" * 70)
    print()

    # Load baseline config
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
    params = SimulationParameters.from_yaml(str(config_path))

    # Small particle count for fast test
    params.debris_count = 100

    print("Initializing simulation...")
    state = initialize_simulation(params, seed=42)

    print(f"Simulation configuration:")
    print(f"  Debris particles: {state.n_debris}")
    print(f"  Ring BHs: {state.n_bh}")
    print(f"  Timestep: {params.dt * const.s_to_Gyr:.6f} Gyr")
    print()

    # Calculate initial energy
    E_initial = calculate_total_energy(state, params)
    print(f"Initial total energy: {E_initial:.6e} J")
    print()

    # Check Courant condition
    print("=" * 70)
    print("COURANT CONDITION CHECK")
    print("=" * 70)
    courant_check = check_timestep_against_courant(state, params)

    print(f"Current timestep: {courant_check['current_dt'] * const.s_to_Gyr:.6f} Gyr")
    print(f"Courant limit:    {courant_check['recommended_dt'] * const.s_to_Gyr:.6f} Gyr")
    print(f"Ratio (current/limit): {courant_check['ratio']:.3f}")
    print()

    if courant_check['satisfies_courant']:
        print("[OK] Timestep satisfies Courant condition")
    else:
        print("[WARNING] Timestep VIOLATES Courant condition!")

    if courant_check['warning']:
        print(f"  {courant_check['warning']}")

    print()

    # Now demonstrate with an inappropriately large timestep
    print("=" * 70)
    print("TESTING WITH LARGE TIMESTEP (10x baseline)")
    print("=" * 70)

    params.dt = params.dt * 10.0
    courant_check_large = check_timestep_against_courant(state, params)

    print(f"Current timestep: {courant_check_large['current_dt'] * const.s_to_Gyr:.6f} Gyr")
    print(f"Courant limit:    {courant_check_large['recommended_dt'] * const.s_to_Gyr:.6f} Gyr")
    print(f"Ratio (current/limit): {courant_check_large['ratio']:.3f}")
    print()

    if courant_check_large['satisfies_courant']:
        print("[OK] Timestep satisfies Courant condition")
    else:
        print("[WARNING] Timestep VIOLATES Courant condition!")

    if courant_check_large['warning']:
        print(f"  {courant_check_large['warning']}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The Courant condition provides a runtime check for timestep stability.")
    print("During the simulation, check this periodically to detect problems early.")
    print()
    print("Recommended: Check Courant condition every 100-1000 timesteps.")
    print("             If violated, reduce timestep and restart from checkpoint.")

if __name__ == "__main__":
    main()
