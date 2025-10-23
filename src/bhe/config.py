"""
Configuration management for black hole explosion simulation.

This module handles loading and parsing YAML configuration files,
converting all parameters to SI units.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path
import numpy as np

from bhe import constants as const


@dataclass
class RingConfig:
    """Configuration for a single ring of black holes."""

    ring_id: int
    count: int
    radius: float  # light-years
    mass_per_bh: float  # solar masses
    is_static: bool
    orbital_velocity: float = 0.0  # fraction of c (0 for static rings)
    capture_radius: float = 0.0  # light-years (only for Ring 0)

    def __repr__(self):
        """Human-readable representation."""
        radius_gly = self.radius / const.Gly  # ly → Gly
        mass_solar = self.mass_per_bh  # Already in M_sun
        vel_frac_c = self.orbital_velocity  # Already fraction of c

        if self.is_static:
            return (f"Ring {self.ring_id}: {self.count} BHs at {radius_gly:.1f} Gly, "
                   f"{mass_solar:.2e} M_sun each (static)")
        else:
            return (f"Ring {self.ring_id}: {self.count} BHs at {radius_gly:.1f} Gly, "
                   f"{mass_solar:.2e} M_sun each, v={vel_frac_c:.2f}c")


@dataclass
class SimulationParameters:
    """
    Container for all simulation parameters.

    All internal values stored in natural units:
    - Distance: light-years (ly)
    - Time: years (yr)
    - Mass: solar masses (M_sun)
    - Velocity: fraction of speed of light (c = 1.0 ly/yr)
    """

    # Metadata
    simulation_name: str
    output_directory: str

    # Central black hole
    M_central: float  # solar masses

    # Ring configurations
    rings: List[RingConfig] = field(default_factory=list)

    # Debris field
    debris_count: int = 1000
    debris_r_min: float = 0.0  # light-years
    debris_r_max: float = 0.0  # light-years
    debris_v_min: float = 0.0  # fraction of c
    debris_v_max: float = 0.0  # fraction of c
    debris_distribution: str = "uniform"

    # Simulation control
    dt: float = 0.0  # years
    duration: float = 0.0  # years
    output_interval: float = 0.0  # years
    checkpoint_interval: float = 0.0  # years

    # Physics options
    force_method: str = "direct"  # "direct" or "barnes_hut"
    use_newtonian_enhancements: bool = False  # Must be False; errors if True

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
        if self.debris_count == 0:
            return 0.0
        return self.M_central / self.debris_count

    @property
    def total_bh_count(self) -> int:
        """Total number of black holes across all rings."""
        return sum(ring.count for ring in self.rings)

    def validate(self) -> list:
        """
        Perform sanity checks on configuration parameters.

        Returns:
            List of warning/error messages. Empty list if all checks pass.
        """
        warnings = []

        # Calculate Schwarzschild radius in natural units
        r_schwarzschild = 2 * const.G * self.M_central / const.c_squared  # ly
        r_schwarzschild_gly = r_schwarzschild / const.Gly  # Gly

        # Check central mass is positive
        if self.M_central <= 0:
            warnings.append(f"ERROR: Central BH mass must be positive, got {self.M_central}")

        # Check debris parameters
        if self.debris_count <= 0:
            warnings.append(f"ERROR: debris_count must be positive, got {self.debris_count}")

        if self.debris_r_min >= self.debris_r_max:
            warnings.append(f"ERROR: debris_r_min ({self.debris_r_min}) must be < debris_r_max ({self.debris_r_max})")

        if self.debris_v_min >= self.debris_v_max:
            warnings.append(f"ERROR: debris_v_min must be < debris_v_max")

        if self.debris_v_max >= 1.0:
            warnings.append(f"ERROR: debris_v_max ({self.debris_v_max:.3f}c) must be < c")

        # Check timestep
        if self.dt <= 0:
            warnings.append(f"ERROR: timestep dt must be positive, got {self.dt}")

        if self.duration <= 0:
            warnings.append(f"ERROR: duration must be positive, got {self.duration}")

        if self.dt >= self.duration:
            warnings.append(f"WARNING: timestep ({self.dt}) >= duration ({self.duration})")

        # Check ring configurations
        for ring in self.rings:
            ring_name = f"Ring {ring.ring_id}"

            # Check mass
            if ring.mass_per_bh <= 0:
                warnings.append(f"ERROR: {ring_name} mass must be positive")

            # Check radius
            if ring.radius <= 0:
                warnings.append(f"ERROR: {ring_name} radius must be positive")

            # For Ring 0, check if inside Schwarzschild radius
            if ring.ring_id == 0 and ring.radius < r_schwarzschild:
                warnings.append(
                    f"ERROR: {ring_name} radius ({ring.radius/const.Gly:.2f} Gly) is INSIDE "
                    f"Schwarzschild radius ({r_schwarzschild_gly:.2f} Gly). No stable orbit possible!"
                )

            # Warn if Ring 0 is very close to Schwarzschild radius
            if ring.ring_id == 0 and ring.radius < 1.2 * r_schwarzschild:
                warnings.append(
                    f"WARNING: {ring_name} radius ({ring.radius/const.Gly:.2f} Gly) is very close to "
                    f"Schwarzschild radius ({r_schwarzschild_gly:.2f} Gly). Orbit may be unstable."
                )

            # Check velocity for non-static rings
            if not ring.is_static:
                if ring.orbital_velocity <= 0:
                    warnings.append(f"ERROR: {ring_name} orbital velocity must be positive")

                if ring.orbital_velocity >= 1.0:
                    warnings.append(
                        f"ERROR: {ring_name} orbital velocity ({ring.orbital_velocity:.3f}c) "
                        f"must be < c"
                    )

                # Calculate Keplerian velocity and warn if significantly different
                v_keplerian = np.sqrt(const.G * self.M_central / ring.radius)
                v_ratio = ring.orbital_velocity / v_keplerian

                if v_ratio < 0.5:
                    warnings.append(
                        f"WARNING: {ring_name} velocity ({ring.orbital_velocity:.3f}c) is much less than "
                        f"Keplerian ({v_keplerian:.3f}c). Orbit will spiral inward rapidly."
                    )
                elif v_ratio > 2.0:
                    warnings.append(
                        f"WARNING: {ring_name} velocity ({ring.orbital_velocity:.3f}c) is much greater than "
                        f"Keplerian ({v_keplerian:.3f}c). Orbit will spiral outward rapidly."
                    )
                elif abs(v_ratio - 1.0) > 0.1:
                    warnings.append(
                        f"INFO: {ring_name} velocity ({ring.orbital_velocity:.3f}c) differs from "
                        f"Keplerian ({v_keplerian:.3f}c) by {abs(v_ratio-1.0)*100:.1f}%. "
                        f"Orbit may not be stable."
                    )

            # Check capture radius for Ring 0
            if ring.ring_id == 0:
                if ring.capture_radius <= 0:
                    warnings.append(f"WARNING: {ring_name} capture_radius should be positive for accretion")

                if ring.capture_radius > ring.radius * 0.1:
                    warnings.append(
                        f"WARNING: {ring_name} capture_radius ({ring.capture_radius/const.Gly:.2f} Gly) "
                        f"is large compared to orbital radius ({ring.radius/const.Gly:.2f} Gly). "
                        f"May accrete too aggressively."
                    )

        # Check for overlapping rings (warn if rings are too close)
        for i, ring1 in enumerate(self.rings):
            for ring2 in self.rings[i+1:]:
                if abs(ring1.radius - ring2.radius) < 1.0 * const.Gly:
                    warnings.append(
                        f"WARNING: Ring {ring1.ring_id} and Ring {ring2.ring_id} are very close "
                        f"({abs(ring1.radius - ring2.radius)/const.Gly:.2f} Gly apart)"
                    )

        return warnings

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

        # Extract central black hole mass (already in M_sun)
        M_central = to_float(config['central_black_hole']['mass_solar_masses'])

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

            # Parse values in natural units (Gly → ly, M_sun, fraction of c)
            radius = to_float(ring_data['radius_gly']) * const.Gly  # Gly → ly
            mass_per_bh = to_float(ring_data['mass_per_bh_solar_masses'])  # Already in M_sun
            is_static = to_bool(ring_data.get('is_static', False))

            # Orbital velocity (only for non-static rings)
            if not is_static:
                # Check velocity mode (keplerian or manual)
                velocity_mode = ring_data.get('velocity_mode', 'manual')

                if velocity_mode == 'keplerian':
                    # Calculate Keplerian orbital velocity: v = sqrt(G * M / r) in units of c
                    orbital_velocity = np.sqrt(const.G * M_central / radius)
                elif velocity_mode == 'manual':
                    # Use user-specified velocity (already as fraction of c)
                    if 'orbital_velocity_fraction_c' in ring_data:
                        orbital_velocity = to_float(ring_data['orbital_velocity_fraction_c'])
                    else:
                        raise ValueError(f"{ring_name}: velocity_mode='manual' requires orbital_velocity_fraction_c")
                else:
                    raise ValueError(f"{ring_name}: velocity_mode must be 'keplerian' or 'manual', got '{velocity_mode}'")
            else:
                orbital_velocity = 0.0

            # Capture radius (only for Ring 0)
            if ring_id == 0 and 'capture_radius_gly' in ring_data:
                capture_radius = to_float(ring_data['capture_radius_gly']) * const.Gly  # Gly → ly
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
        debris_r_min = to_float(debris_data['position_min_gly']) * const.Gly  # Gly → ly
        debris_r_max = to_float(debris_data['position_max_gly']) * const.Gly  # Gly → ly
        debris_v_min = to_float(debris_data['velocity_min_fraction_c'])  # Already fraction of c
        debris_v_max = to_float(debris_data['velocity_max_fraction_c'])  # Already fraction of c
        debris_distribution = debris_data.get('distribution', 'uniform')

        # Parse simulation control
        sim_control = config['simulation_control']
        dt = to_float(sim_control['timestep_gyr']) * 1.0e9  # Gyr → yr
        duration = to_float(sim_control['duration_gyr']) * 1.0e9  # Gyr → yr
        output_interval = to_float(sim_control['output_interval_gyr']) * 1.0e9  # Gyr → yr
        checkpoint_interval = to_float(sim_control['checkpoint_interval_gyr']) * 1.0e9  # Gyr → yr

        # Parse physics options
        physics_opts = config['physics_options']
        force_method = physics_opts.get('force_method', 'direct')
        use_newtonian_enhancements = to_bool(physics_opts.get('use_newtonian_enhancements', False))

        # Validate: newtonian enhancements not supported
        if use_newtonian_enhancements:
            raise ValueError(
                "use_newtonian_enhancements=true is not supported. "
                "The simulation only supports Newtonian gravity with rest masses."
            )

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
            use_newtonian_enhancements=use_newtonian_enhancements,
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
            f"Central BH: {self.M_central:.2e} M_sun",
            f"Rings: {len(self.rings)} configured",
        ]
        for ring in self.rings:
            lines.append(f"  {ring}")
        lines.extend([
            f"Debris: {self.debris_count} particles",
            f"Duration: {self.duration / 1.0e9:.2f} Gyr",
            f"Timestep: {self.dt / 1.0e9:.6f} Gyr",
            f"Force method: {self.force_method}",
        ])
        return "\n".join(lines)
