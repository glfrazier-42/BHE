"""
Physical and astronomical constants used throughout the simulation.

NATURAL UNITS SYSTEM:
- Distance: light-years (ly)
- Velocity: speed of light (c)
- Mass: solar masses (M_sun)
- Time: years (yr)

This means:
- c = 1.0 ly/yr (by definition)
- Positions are in ly
- Velocities are in units of c (dimensionless, 0 to 1)
- Masses are in M_sun
- Time is in yr
- G is in ly³/(M_sun·yr²)
"""

# Base natural units
c = 1.0  # Speed of light [ly/yr] - exactly 1 by definition
c_squared = 1.0  # c² [ly²/yr²]

# Gravitational constant in natural units
# Derived from G_SI = 6.674e-11 m³/(kg·s²)
# Conversion: G_natural = G_SI × M_sun × (yr)² / (ly)³
# Where: 1 ly = 9.461e15 m, 1 yr = 3.1557e7 s, 1 M_sun = 1.989e30 kg
# G_natural = 6.674e-11 × 1.989e30 × (3.1557e7)² / (9.461e15)³
# G_natural ≈ 1.561e-13 ly³/(M_sun·yr²)
G = 1.560994e-13  # Gravitational constant [ly³/(M_sun·yr²)]

# Convenient scale factors
Gly = 1.0e9  # Gigalight-year in ly

# For reference: SI equivalents (not used in calculations)
# 1 ly = 9.461e15 m
# 1 yr = 3.1557e7 s
# 1 M_sun = 1.989e30 kg
# c_SI = 299792458 m/s = 1 ly/yr
