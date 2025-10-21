"""
Physical and astronomical constants used throughout the simulation.

All values are in SI units unless otherwise noted.
"""

# Physical constants
c = 299_792_458  # Speed of light [m/s]
G = 6.674e-11  # Gravitational constant [m³/(kg·s²)]

# Derived constants
c_squared = c * c  # Speed of light squared [m²/s²]

# Astronomical constants
M_sun = 1.989e30  # Solar mass [kg]

# Unit conversions
# Length
m_to_ly = 1.0 / 9.461e15  # Meters to light-years
ly_to_m = 9.461e15  # Light-years to meters
Gly_to_m = 9.461e24  # Gigalight-years to meters
m_to_Gly = 1.0 / 9.461e24  # Meters to gigalight-years

# Time
s_to_yr = 1.0 / 31_557_600  # Seconds to years
yr_to_s = 31_557_600  # Years to seconds
Gyr_to_s = 3.1557e16  # Gigayears to seconds
s_to_Gyr = 1.0 / 3.1557e16  # Seconds to gigayears

# Mass
solar_mass_to_kg = M_sun  # Solar masses to kilograms
kg_to_solar_mass = 1.0 / M_sun  # Kilograms to solar masses
