"""
Configuration management for black hole explosion simulation.

This module handles loading and parsing YAML configuration files,
converting all parameters to SI units.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path

from . import constants as const


@dataclass
class RingConfig:
    """Configuration for a single ring of black holes."""

    ring_id: int
    count: int
    radius: float  # meters
    mass_per_bh: float  # kg
    is_static: bool
    orbital_velocity: float = 0.0  # m/s (0 for static rings)
    capture_radius: float = 0.0  # meters (only for Ring 0)

    def __repr__(self):
        """Human-readable representation."""
        radius_gly = self.radius * const.m_to_Gly
        mass_solar = self.mass_per_bh * const.kg_to_solar_mass
        vel_frac_c = self.orbital_velocity / const.c

        if self.is_static:
            return (f"Ring {self.ring_id}: {self.count} BHs at {radius_gly:.1f} Gly, "
                   f"{mass_solar:.2e} M☉ each (static)")
        else:
            return (f"Ring {self.ring_id}: {self.count} BHs at {radius_gly:.1f} Gly, "
                   f"{mass_solar:.2e} M☉ each, v={vel_frac_c:.2f}c")


@dataclass
class SimulationParameters:
    """
    Container for all simulation parameters.

    All internal values stored in SI units:
    - Distance: meters
    - Time: seconds
    - Mass: kilograms
    - Velocity: m/s
    """

    # Metadata
    simulation_name: str
    output_directory: str

    # Central black hole
    M_central: float  # kg

    # Ring configurations
    rings: List[RingConfig] = field(default_factory=list)

    # Debris field
    debris_count: int = 1000
    debris_r_min: float = 0.0  # meters
    debris_r_max: float = 0.0  # meters
    debris_v_min: float = 0.0  # m/s
    debris_v_max: float = 0.0  # m/s
    debris_distribution: str = "uniform"

    # Simulation control
    dt: float = 0.0  # seconds
    duration: float = 0.0  # seconds
    output_interval: float = 0.0  # seconds
    checkpoint_interval: float = 0.0  # seconds

    # Physics options
    force_method: str = "direct"  # "direct" or "barnes_hut"
    include_debris_gravity: bool = False
    use_relativistic_mass: bool = True

    # Barnes-Hut parameters (if applicable)
    barnes_hut_theta: float = 0.5
    barnes_hut_max_particles_per_leaf: int = 8
    barnes_hut_tree_rebuild_interval: int = 10

    # Diagnostics
    check_energy_conservation: bool = True
    check_momentum_conservation: bool = True
    log_level: str = "INFO"

    @property
    def debris_mass_per_particle(self) -> float:
        """Calculate mass per debris particle (M_central / particle_count)."""
        return self.M_central / self.debris_count

    @property
    def total_bh_count(self) -> int:
        """Total number of black holes across all rings."""
        return sum(ring.count for ring in self.rings)

    @classmethod
    def from_yaml(cls, filepath: str) -> 'SimulationParameters':
        """
        Load configuration from YAML file and convert to SI units.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            SimulationParameters object with all values in SI units

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        def to_float(value: Any) -> float:
            """Convert value to float, handling YAML quirks with scientific notation."""
            if isinstance(value, str):
                return float(value)
            return float(value)

        def to_int(value: Any) -> int:
            """Convert value to int."""
            if isinstance(value, str):
                return int(value)
            return int(value)

        def to_bool(value: Any) -> bool:
            """Convert value to bool."""
            if isinstance(value, str):
                return value.lower() in ('true', 'yes', '1')
            return bool(value)

        # Load YAML file
        config_path = Path(filepath)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract and convert central black hole mass
        M_central_solar = to_float(config['central_black_hole']['mass_solar_masses'])
        M_central = M_central_solar * const.M_sun

        # Parse ring configurations
        rings = []
        ring_names = ['ring_0', 'ring_1', 'ring_2', 'ring_3']

        for ring_id, ring_name in enumerate(ring_names):
            if ring_name not in config:
                continue

            ring_data = config[ring_name]
            count = to_int(ring_data['count'])

            # Skip if count is 0 (ring disabled)
            if count == 0:
                continue

            # Convert units
            radius = to_float(ring_data['radius_gly']) * const.Gly_to_m
            mass_per_bh = to_float(ring_data['mass_per_bh_solar_masses']) * const.M_sun
            is_static = to_bool(ring_data.get('is_static', False))

            # Orbital velocity (only for non-static rings)
            if not is_static and 'orbital_velocity_fraction_c' in ring_data:
                orbital_velocity = to_float(ring_data['orbital_velocity_fraction_c']) * const.c
            else:
                orbital_velocity = 0.0

            # Capture radius (only for Ring 0)
            if ring_id == 0 and 'capture_radius_gly' in ring_data:
                capture_radius = to_float(ring_data['capture_radius_gly']) * const.Gly_to_m
            else:
                capture_radius = 0.0

            rings.append(RingConfig(
                ring_id=ring_id,
                count=count,
                radius=radius,
                mass_per_bh=mass_per_bh,
                is_static=is_static,
                orbital_velocity=orbital_velocity,
                capture_radius=capture_radius
            ))

        # Parse debris field parameters
        debris_data = config['debris_field']
        debris_count = to_int(debris_data['particle_count'])
        debris_r_min = to_float(debris_data['position_min_gly']) * const.Gly_to_m
        debris_r_max = to_float(debris_data['position_max_gly']) * const.Gly_to_m
        debris_v_min = to_float(debris_data['velocity_min_fraction_c']) * const.c
        debris_v_max = to_float(debris_data['velocity_max_fraction_c']) * const.c
        debris_distribution = debris_data.get('distribution', 'uniform')

        # Parse simulation control
        sim_control = config['simulation_control']
        dt = to_float(sim_control['timestep_gyr']) * const.Gyr_to_s
        duration = to_float(sim_control['duration_gyr']) * const.Gyr_to_s
        output_interval = to_float(sim_control['output_interval_gyr']) * const.Gyr_to_s
        checkpoint_interval = to_float(sim_control['checkpoint_interval_gyr']) * const.Gyr_to_s

        # Parse physics options
        physics_opts = config['physics_options']
        force_method = physics_opts.get('force_method', 'direct')
        include_debris_gravity = to_bool(physics_opts.get('include_debris_gravity', False))
        use_relativistic_mass = to_bool(physics_opts.get('use_relativistic_mass', True))

        # Barnes-Hut parameters
        barnes_hut_theta = 0.5
        barnes_hut_max_particles_per_leaf = 8
        barnes_hut_tree_rebuild_interval = 10

        if 'barnes_hut' in physics_opts:
            bh_params = physics_opts['barnes_hut']
            barnes_hut_theta = to_float(bh_params.get('opening_angle_theta', 0.5))
            barnes_hut_max_particles_per_leaf = to_int(bh_params.get('max_particles_per_leaf', 8))
            barnes_hut_tree_rebuild_interval = to_int(bh_params.get('tree_rebuild_interval', 10))

        # Parse diagnostics
        diagnostics = config.get('diagnostics', {})
        check_energy = to_bool(diagnostics.get('check_energy_conservation', True))
        check_momentum = to_bool(diagnostics.get('check_momentum_conservation', True))
        log_level = diagnostics.get('log_level', 'INFO')

        # Create and return SimulationParameters object
        return cls(
            simulation_name=config['simulation_name'],
            output_directory=config['output_directory'],
            M_central=M_central,
            rings=rings,
            debris_count=debris_count,
            debris_r_min=debris_r_min,
            debris_r_max=debris_r_max,
            debris_v_min=debris_v_min,
            debris_v_max=debris_v_max,
            debris_distribution=debris_distribution,
            dt=dt,
            duration=duration,
            output_interval=output_interval,
            checkpoint_interval=checkpoint_interval,
            force_method=force_method,
            include_debris_gravity=include_debris_gravity,
            use_relativistic_mass=use_relativistic_mass,
            barnes_hut_theta=barnes_hut_theta,
            barnes_hut_max_particles_per_leaf=barnes_hut_max_particles_per_leaf,
            barnes_hut_tree_rebuild_interval=barnes_hut_tree_rebuild_interval,
            check_energy_conservation=check_energy,
            check_momentum_conservation=check_momentum,
            log_level=log_level
        )

    def __repr__(self):
        """Human-readable representation."""
        lines = [
            f"Simulation: {self.simulation_name}",
            f"Central BH: {self.M_central * const.kg_to_solar_mass:.2e} M☉",
            f"Rings: {len(self.rings)} configured",
        ]
        for ring in self.rings:
            lines.append(f"  {ring}")
        lines.extend([
            f"Debris: {self.debris_count} particles",
            f"Duration: {self.duration * const.s_to_Gyr:.2f} Gyr",
            f"Timestep: {self.dt * const.s_to_Gyr:.6f} Gyr",
            f"Force method: {self.force_method}",
            f"Debris gravity: {self.include_debris_gravity}",
        ])
        return "\n".join(lines)
