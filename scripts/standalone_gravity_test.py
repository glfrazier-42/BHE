"""
Standalone gravity test - completely independent of the main codebase.
Tests the physics of a particle falling toward a massive object.

This uses solar system parameters: Earth falling toward the Sun.
"""

import numpy as np

# Physical constants
G = 6.67430e-11  # m³/(kg·s²)
M_sun = 1.98892e30  # kg
M_earth = 5.972e24  # kg
AU = 1.496e11  # meters
c = 299792458.0  # m/s

def calculate_acceleration(pos_particle, pos_attractor, mass_attractor):
    """
    Calculate gravitational acceleration on a particle.

    a = G * M / r^2

    Note: Particle mass does NOT affect acceleration (equivalence principle)
    """
    # Vector from particle to attractor
    r_vec = pos_attractor - pos_particle

    # Distance
    r_squared = np.sum(r_vec**2)
    r = np.sqrt(r_squared)

    # Acceleration magnitude
    a_mag = G * mass_attractor / r_squared

    # Acceleration vector (points toward attractor)
    a_vec = a_mag * (r_vec / r)

    return a_vec

def leapfrog_step(pos, vel, mass_attractor, pos_attractor, dt):
    """
    Single leapfrog integration step.

    1. Calculate acceleration at current position
    2. Half-step velocity kick: v_half = v + 0.5*a*dt
    3. Full-step position drift: pos_new = pos + v_half*dt
    4. Calculate acceleration at new position
    5. Final half-step velocity kick: v_new = v_half + 0.5*a_new*dt
    """
    # Step 1: Acceleration at current position
    a_old = calculate_acceleration(pos, pos_attractor, mass_attractor)

    # Step 2: Half-step velocity kick
    v_half = vel + 0.5 * a_old * dt

    # Step 3: Full-step position drift
    pos_new = pos + v_half * dt

    # Step 4: Acceleration at new position
    a_new = calculate_acceleration(pos_new, pos_attractor, mass_attractor)

    # Step 5: Final half-step velocity kick
    v_new = v_half + 0.5 * a_new * dt

    return pos_new, v_new

def main():
    print("=" * 70)
    print("STANDALONE GRAVITY TEST")
    print("=" * 70)
    print()

    # Setup: Earth at 1 AU, at rest, falling toward Sun at origin
    pos = np.array([1.0 * AU, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    mass_particle = M_earth

    pos_sun = np.array([0.0, 0.0, 0.0])
    mass_sun = M_sun

    # Timestep: 0.52 days = 45,000 seconds
    dt = 45000.0

    print(f"Initial conditions:")
    print(f"  Particle position: {pos[0]/AU:.6f} AU = {pos[0]:.3e} m")
    print(f"  Particle velocity: {vel[0]:.3e} m/s = {vel[0]/c:.3e} c")
    print(f"  Sun position: {pos_sun[0]/AU:.6f} AU")
    print(f"  Sun mass: {mass_sun:.3e} kg")
    print(f"  Timestep: {dt:.1f} s = {dt/86400:.3f} days")
    print()

    # Expected acceleration
    r = np.linalg.norm(pos - pos_sun)
    a_expected = G * mass_sun / (r**2)
    print(f"Expected acceleration: {a_expected:.6f} m/s²")
    print()

    # Test single step manually
    print("=" * 70)
    print("MANUAL CALCULATION FOR STEP 1")
    print("=" * 70)

    # Step 1: Acceleration at t=0
    a_0 = calculate_acceleration(pos, pos_sun, mass_sun)
    print(f"Acceleration at t=0: {a_0}")
    print(f"  Magnitude: {np.linalg.norm(a_0):.6f} m/s²")
    print(f"  Direction: toward Sun (negative x)")
    print()

    # Step 2: Half-step velocity kick
    v_half = vel + 0.5 * a_0 * dt
    print(f"After half-step velocity kick:")
    print(f"  v_half = {vel} + 0.5 * {a_0} * {dt}")
    print(f"  v_half = {v_half}")
    print(f"  v_half magnitude: {np.linalg.norm(v_half):.3e} m/s")
    print()

    # Step 3: Position drift
    pos_1 = pos + v_half * dt
    print(f"After position drift:")
    print(f"  pos_new = {pos} + {v_half} * {dt}")
    print(f"  pos_new = {pos_1}")
    print(f"  Position change: {pos_1[0] - pos[0]:.3e} m = {(pos_1[0] - pos[0])/AU:.6e} AU")
    print()

    # Step 4: Acceleration at new position
    a_1 = calculate_acceleration(pos_1, pos_sun, mass_sun)
    print(f"Acceleration at new position: {a_1}")
    print(f"  Magnitude: {np.linalg.norm(a_1):.6f} m/s²")
    print()

    # Step 5: Final velocity kick
    v_1 = v_half + 0.5 * a_1 * dt
    print(f"After final velocity kick:")
    print(f"  v_new = {v_half} + 0.5 * {a_1} * {dt}")
    print(f"  v_new = {v_1}")
    print(f"  v_new magnitude: {np.linalg.norm(v_1):.3e} m/s")
    print()

    print("=" * 70)
    print("RUNNING 5 LEAPFROG STEPS")
    print("=" * 70)
    print()

    # Reset
    pos = np.array([1.0 * AU, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])

    initial_pos = pos.copy()

    for step in range(5):
        pos, vel = leapfrog_step(pos, vel, mass_sun, pos_sun, dt)
        print(f"Step {step+1}:")
        print(f"  Position: {pos[0]/AU:.8f} AU (delta = {(pos[0]-initial_pos[0])/AU:.8e} AU)")
        print(f"  Velocity: {vel[0]:.6e} m/s (= {vel[0]/c:.6e} c)")

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Initial position: {initial_pos[0]/AU:.10f} AU")
    print(f"Final position:   {pos[0]/AU:.10f} AU")
    print(f"Position change:  {(pos[0]-initial_pos[0])/AU:.10f} AU")
    print(f"                  {pos[0]-initial_pos[0]:.6e} m")
    print()
    print(f"Final velocity:   {vel[0]:.6e} m/s")
    print(f"                  {vel[0]/c:.6e} c")
    print()

    # Checks
    print("=" * 70)
    print("CHECKS")
    print("=" * 70)

    if vel[0] < 0:
        print("[PASS] Velocity is negative (toward Sun)")
    else:
        print("[FAIL] Velocity should be negative!")

    if pos[0] < initial_pos[0]:
        print("[PASS] Position decreased (moved toward Sun)")
    else:
        print("[FAIL] Position should have decreased!")

    # Calculate expected position change for comparison
    # Using simplified formula: delta_x ~ 0.5 * a * t^2  (for small times)
    total_time = 5 * dt
    delta_x_approx = 0.5 * a_expected * total_time**2
    print()
    print(f"Approximate expected delta_x (using 0.5*a*t^2): {delta_x_approx:.6e} m")
    print(f"                                               {delta_x_approx/AU:.10f} AU")
    print(f"Actual delta_x: {pos[0]-initial_pos[0]:.6e} m")
    print(f"                {(pos[0]-initial_pos[0])/AU:.10f} AU")

if __name__ == "__main__":
    main()
