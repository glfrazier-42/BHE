"""
Simple test of gravitational acceleration and position updates WITHOUT Numba.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants as const

def main():
    print("=" * 70)
    print("SIMPLE GRAVITY TEST (no Numba)")
    print("=" * 70)
    print()

    # Particle at 100 AU from 1e22 solar mass BH at origin
    AU_to_m = 1.496e11
    x = 100.0 * AU_to_m
    v = 0.0

    M_bh = 1.0e22 * const.M_sun
    dt = 86400.0  # 1 day

    print(f"Initial position: {x/AU_to_m} AU = {x} m")
    print(f"Initial velocity: {v} m/s")
    print(f"BH mass: {M_bh} kg = {M_bh/const.M_sun} M_sun")
    print(f"Timestep: {dt} s = {dt/86400} days")
    print()

    # Calculate gravitational acceleration
    r = x
    a = -const.G * M_bh / (r**2)  # Negative because toward origin

    print(f"Gravitational acceleration: {a} m/s^2")
    print()

    # Leapfrog integration for 1 timestep
    # Half-step velocity kick
    v_half = v + 0.5 * a * dt
    print(f"After half-step velocity kick: v_half = {v_half} m/s = {v_half/const.c} c")

    # Full-step position drift
    x_new = x + v_half * dt
    print(f"After position drift: x_new = {x_new} m = {x_new/AU_to_m} AU")
    print(f"Position change: {x_new - x} m = {(x_new - x)/AU_to_m} AU")

    # Recalculate acceleration at new position
    r_new = x_new
    a_new = -const.G * M_bh / (r_new**2)
    print(f"Acceleration at new position: {a_new} m/s^2")

    # Final half-step velocity kick
    v_new = v_half + 0.5 * a_new * dt
    print(f"After final velocity kick: v_new = {v_new} m/s = {v_new/const.c} c")
    print()

    # Summary
    print("=" * 70)
    print("RESULTS AFTER 1 TIMESTEP")
    print("=" * 70)
    print(f"Position changed: {x_new != x}")
    print(f"Velocity changed: {v_new != v}")
    print(f"Delta_x: {x_new - x} m")
    print(f"Delta_v: {v_new - v} m/s")

if __name__ == "__main__":
    main()
