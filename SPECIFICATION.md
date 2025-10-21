## Computational Considerations

### Performance Optimization with Numba

**Critical: Numba constraints must be followed in hot loops**

Numba-compatible code:
- ✅ NumPy arrays and operations
- ✅ Basic math operations (+, -, *, /, **, sqrt, etc.)
- ✅ for loops, if/else, while
- ✅ Recursion (for tree traversal)
- ✅ Function calls to other @jit functions
- ✅ Typed dictionaries (numba.typed.Dict)
- ❌ Python lists, dicts, sets in compiled functions (use typed versions)
- ❌ Object method calls
- ❌ String operations

**Expected performance gains:**
- Pure Python: ~1000 particles, 1000 timesteps = hours
- Python + NumPy (vectorized): ~10-50x faster = minutes
- Python + Numba: ~50-200x faster than pure Python = seconds to minutes
- Barnes-Hut + Numba: Enables 100k-1M particles

### Force Calculation Complexity

| Method | Complexity | N=1k | N=# Black Hole Explosion Simulation - Technical Specification

## Executive Summary

This document specifies a novel N-body simulation of an exploding supermassive black hole surrounded by orbiting black holes. The simulation aims to demonstrate that debris from the explosion can escape gravitational collapse through the "gravitational tugboat" effect of fast-moving orbiting black holes that accrete and drag debris outward.

## Implementation Language and Tools

**Primary Language**: Python 3.10+

**Core Dependencies**:
- **NumPy**: Array operations and vectorized calculations
- **Numba**: JIT compilation for performance-critical loops (near-C speeds)
- **PyYAML**: Configuration file parsing
- **Matplotlib**: Visualization
- **h5py**: Data storage for large simulation outputs

**Why Numba from the start?**
- With 1000-10000 particles and 50000+ timesteps, we hit O(N²) or O(N) operations billions of times
- Pure Python would take hours/days; Numba brings this to minutes/hours
- Numba constraints are minimal: use NumPy arrays, avoid Python objects in hot loops
- Better to design with Numba in mind than refactor later

## Configuration Schema (YAML)

All simulation parameters should be specified in YAML configuration files for reproducibility and easy parameter sweeps.

### Example Configuration: `baseline_config.yaml`

```yaml
# Baseline simulation: Test Ring 0 tugboat effect without debris gravity
simulation_name: "baseline_ring0_test"
output_directory: "./output/baseline"

# Central black hole that explodes
central_black_hole:
  mass_solar_masses: 4.0e22  # ~200% visible universe mass

# Ring 0: The Tugboats - close, fast-moving black holes (OPTIONAL)
ring_0:
  count: 0  # Baseline: no tugboats. Set to 4-8 for enhanced scenario
  radius_gly: 3.0  # Just outside event horizon (only used if count > 0)
  mass_per_bh_solar_masses: 1.0e21  # Only used if count > 0
  orbital_velocity_fraction_c: 0.8  # Only used if count > 0
  capture_radius_gly: 0.5  # Accretion radius (only used if count > 0)

# Ring 1: Inner shell - long-range gravitational structure
ring_1:
  count: 4
  radius_gly: 100.0
  mass_per_bh_solar_masses: 5.0e21
  is_static: true

# Ring 2: Middle shell - primary acceleration zone
ring_2:
  count: 6
  radius_gly: 150.0
  mass_per_bh_solar_masses: 1.0e22
  is_static: true

# Ring 3: Outer shell - outer gravitational boundary
ring_3:
  count: 8
  radius_gly: 200.0
  mass_per_bh_solar_masses: 1.5e22
  is_static: true

# Debris field parameters
debris_field:
  particle_count: 1000
  velocity_min_fraction_c: 0.01
  velocity_max_fraction_c: 0.92
  distribution: "uniform"  # uniform in solid angle
  # Note: mass_per_particle is calculated as M_central / particle_count

# Simulation control
simulation_control:
  timestep_gyr: 0.001  # May need smaller for stability
  duration_gyr: 50.0
  output_interval_gyr: 0.1  # How often to save state
  checkpoint_interval_gyr: 5.0  # Full checkpoint for restart

# Physics options
physics_options:
  force_method: "barnes_hut"  # "direct" or "barnes_hut"
  include_debris_gravity: false  # Phase 1: off, Phase 2: on
  use_relativistic_mass: true
  barnes_hut:
    opening_angle_theta: 0.5
    max_particles_per_leaf: 8
    tree_rebuild_interval: 10
  
# Logging and diagnostics
diagnostics:
  check_energy_conservation: true
  check_momentum_conservation: true
  log_level: "INFO"
```

### Parameter Sweep Configuration: `sweep_config.yaml`

```yaml
# Parameter sweep: Explore Ring 0 parameter space
sweep_name: "ring0_parameter_sweep"
base_config: "baseline_config.yaml"

# Parameters to sweep (will generate all combinations)
sweep_parameters:
  ring_0.radius_gly: [2.0, 3.0, 4.0, 5.0]
  ring_0.orbital_velocity_fraction_c: [0.7, 0.8, 0.9]
  ring_0.count: [4, 6, 8]
  ring_0.capture_radius_gly: [0.3, 0.5, 1.0]

# Total runs: 4 × 3 × 3 × 3 = 108 simulations
```

### The Core Hypothesis

A supermassive black hole explodes, releasing all its mass as debris. Normally, this debris would gravitationally collapse back on itself (since ejection velocities < c). However, a system of orbiting black holes in close, high-speed orbits provides the escape mechanism:

1. **Ring 0 (Inner Ring)**: Small number of black holes orbiting very close to the explosion center at extreme velocities (~0.7-0.9c)
2. These BHs immediately begin accreting nearby debris
3. Their high velocity + accumulated mass creates a powerful gravitational drag effect
4. They act as "tugboats," pulling debris streams outward and preventing collapse
5. As they move to larger radii, they decelerate and eventually reach stable positions at 100+ Gly

### Four-Ring Structure

**Ring 0 - The Tugboats (NEW)**
- **Location**: 2-5 Gly from center (just outside event horizon of central BH)
- **Number**: 4-8 black holes
- **Individual mass**: 10²⁰ - 10²¹ M☉ (much smaller than central BH)
- **Total ring mass**: 1-5% of central BH mass
- **Orbital velocity**: 0.7-0.9c
- **Role**: Accrete debris and drag it outward

**Ring 1 - Inner Shell**
- **Location**: 100 Gly
- **Number**: 4 BHs (simulation) representing more
- **Mass per BH**: 5×10²¹ M☉
- **Role**: Long-range gravitational structure

**Ring 2 - Middle Shell**
- **Location**: 150 Gly
- **Number**: 6 BHs (simulation) representing more
- **Mass per BH**: 10²² M☉ (previously discussed 10²³ M☉ variant)
- **Role**: Primary gravitational acceleration zone

**Ring 3 - Outer Shell**
- **Location**: 200 Gly
- **Number**: 8 BHs (simulation) representing more
- **Mass per BH**: 1.5×10²² M☉
- **Role**: Outer gravitational boundary

## Physics Implementation

### Constants

```
c = 299,792,458 m/s              // Speed of light
G = 6.674e-11 m³/(kg·s²)         // Gravitational constant
M_sun = 1.989e30 kg              // Solar mass
ly_to_m = 9.461e15 m             // Light-year to meters
Gly_to_m = 9.461e24 m            // Gigalight-year to meters
yr_to_s = 31,557,600 s           // Year to seconds
Gyr_to_s = 3.1557e16 s           // Gigayear to seconds
```

### Lorentz Factor and Relativistic Mass

```
γ(v) = 1 / sqrt(1 - v²/c²)
m_relativistic = γ(v) × m_rest
```

### Gravitational Force Calculation Methods

The simulation supports two modes for force calculation:

**1. Direct N-body** (for small N, validation, or Ring 0-3 forces)
```
For each particle i:
  For each particle/BH j:
    if i ≠ j:
      r_ij = position_j - position_i
      distance = |r_ij|
      m_j_eff = γ(v_j) × m_j_rest  (if use_relativistic)
      F_ij = G × m_j_eff / distance² × (r_ij / distance)
      F_i += F_ij
```

**2. Barnes-Hut Tree** (for large N debris field)
- See detailed Barnes-Hut section below
- O(N log N) complexity
- Controlled accuracy via opening angle θ

### Debris Field Representation

Rather than simulating every galaxy individually:
- Use **100,000 - 1,000,000 representative particles** for production runs
- Start with **1,000 particles** for validation and testing
- Each particle represents a portion of the exploded black hole mass
- **Mass per particle**: M_central / particle_count (automatically calculated)
- Sample uniformly over solid angle (θ, φ)
- Sample ejection velocities from range: 0.01c to 0.95c
- Initial position: All at origin (explosion center)
- Initial velocity: Radial direction determined by (θ, φ), magnitude from sampled distribution

### Barnes-Hut Tree Algorithm for Debris Field Gravity

For large particle counts (100k+), direct N² force calculation is infeasible. The Barnes-Hut algorithm reduces complexity to O(N log N) by using a hierarchical approximation.

#### Algorithm Overview

**Phase 1: Tree Construction**
```
1. Find bounding box containing all particles
2. Create root octree node (cube)
3. Recursively subdivide:
   - If node contains ≤ max_particles_per_leaf: make it a leaf
   - Else: divide into 8 octants (subcubes)
   - Assign particles to appropriate octants
   - Recurse on non-empty octants
4. For each node (bottom-up), compute aggregate properties:
   - Total mass: M_node = Σ m_i (for all particles in subtree)
   - Center of mass: COM_node = Σ(m_i × pos_i) / M_node
   - Aggregate velocity: v_node = Σ(m_i × vel_i) / M_node (for relativistic mass)
```

**Phase 2: Force Calculation (Tree Walk)**
```
For each particle i:
  Initialize force F_i = 0
  Start at root node
  Traverse tree recursively:
    For current node:
      r_vec = node.COM - particle_i.position
      distance = |r_vec|
      
      If distance == 0: 
        skip (particle is in this node)
      
      # Opening angle criterion
      theta = node.size / distance
      
      If theta < theta_threshold:
        # Node is sufficiently far: use aggregate
        m_eff = node.mass (or γ × node.mass if relativistic)
        F_i += G × m_eff / distance² × (r_vec / distance)
      Else if node is leaf:
        # Too close: direct calculation with leaf particles
        For each particle j in leaf:
          If i ≠ j:
            F_i += direct_force(particle_i, particle_j)
      Else:
        # Recurse into 8 child octants
        For each non-empty child:
          Add child to traversal stack
```

#### Data Structures

**OctreeNode** (Numba-compatible):
```python
# Store in parallel arrays for Numba compatibility
node_bounds_min: array[(N_nodes, 3), float64]  # Min corner of cube
node_bounds_max: array[(N_nodes, 3), float64]  # Max corner of cube
node_mass: array[N_nodes, float64]             # Total mass
node_com: array[(N_nodes, 3), float64]         # Center of mass
node_com_velocity: array[(N_nodes, 3), float64]  # Aggregate velocity
node_is_leaf: array[N_nodes, bool]             # Is this a leaf?
node_particle_start: array[N_nodes, int32]     # First particle index (if leaf)
node_particle_count: array[N_nodes, int32]     # Number of particles (if leaf)
node_children: array[(N_nodes, 8), int32]      # Indices of 8 children (-1 if empty)
```

#### Parameters

```yaml
barnes_hut:
  opening_angle_theta: 0.5        # Standard value: 0.5-0.7
  max_particles_per_leaf: 8       # Stop subdividing at this count
  tree_rebuild_interval: 10       # Rebuild tree every N timesteps
  use_for_debris_field: true      # Use Barnes-Hut for debris-debris forces
  use_for_bh_forces: false        # Always use direct calculation for BH forces
```

#### Validation Tests

1. **Two-body orbit**: Should match Kepler orbit exactly
2. **N-body vs Barnes-Hut**: Compare with direct calculation for N=100, vary θ
3. **Energy conservation**: Total energy should be conserved to within tolerance
4. **Plummer sphere**: Should maintain virial equilibrium
5. **Opening angle study**: Verify force accuracy vs θ (expect ~θ² error)

#### Performance Characteristics

| N particles | Direct O(N²) | Barnes-Hut O(N log N) | Speedup |
|-------------|--------------|----------------------|---------|
| 1,000       | 1 sec        | 0.5 sec              | 2×      |
| 10,000      | 100 sec      | 7 sec                | 14×     |
| 100,000     | ~3 hours     | 90 sec               | 120×    |
| 1,000,000   | ~300 hours   | 18 min               | 1000×   |

*Estimates assume Numba-compiled code on modern CPU*

### Ring 0 Orbital Mechanics

Initial configuration at t=0:
- Place Ring 0 BHs in circular orbit at radius R₀ (2-5 Gly)
- Orbital velocity for circular orbit: v_orbit = sqrt(G × M_central / R₀)
- Adjust to desired velocity (may exceed circular orbit velocity)

### Accretion Model

This is the complex part. Several approaches:

**Option 1: Capture Radius**
- Each Ring 0 BH has capture radius r_capture
- Any debris particle within r_capture gets accreted
- Accreted mass added to BH, momentum conserved

**Option 2: Gravitational Focusing**
- Calculate gravitational deflection of debris trajectories
- If trajectory intersects event horizon, particle is accreted
- More physically accurate but computationally expensive

**Option 3: Probabilistic Accretion**
- Probability of accretion ∝ (mass × cross_section) / distance²
- Monte Carlo approach at each timestep
- Simpler but less deterministic

**Recommendation**: Start with Option 1, refine to Option 2 if needed.

### Time Evolution Algorithm

```
1. Initialize system:
   - Place Ring 0 BHs in orbit
   - Place Rings 1-3 at static positions
   - Distribute debris particles at origin with sampled velocities
   - Set M_central_remaining = M_central (will decrease as debris escapes)

2. Main simulation loop (timestep dt):
   For each timestep:
     a. Update Ring 0 positions and velocities:
        - Calculate gravitational force from central mass (if any remains)
        - Calculate gravitational force from Rings 1-3
        - Calculate gravitational force from other Ring 0 BHs
        - Update position: x_new = x + v*dt + 0.5*a*dt²
        - Update velocity: v_new = v + a*dt
     
     b. Check for accretion:
        For each Ring 0 BH:
          For each debris particle:
            If distance < r_capture:
              - Add particle mass to BH
              - Add particle momentum to BH (conserve momentum)
              - Remove particle from debris list
              - Reduce M_central_remaining
     
     c. Update debris particle positions and velocities:
        For each debris particle:
          - Calculate force from Ring 0 BHs (using relativistic mass)
          - Calculate force from Rings 1-3
          - Calculate force from other debris particles (if feasible)
          - Update position and velocity
     
     d. Record data:
        - Time
        - Ring 0 BH positions, velocities, masses
        - Sample of debris particle positions, velocities
        - Energy, momentum (for conservation checks)
        - Redshift estimates for debris particles
     
     e. Check termination conditions:
        - Maximum simulation time reached (e.g., 50 Gyr)
        - All debris either accreted or escaped (distance > threshold)

3. Post-processing:
   - Calculate redshift vs. proper time for debris particles
   - Analyze final state of Ring 0 BHs
   - Determine fraction of debris that escaped vs. was accreted
   - Generate visualizations
```

## Data Structures

All data structures should be designed for Numba compatibility (NumPy arrays, not Python objects in hot loops).

### Python Implementation

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class SimulationParameters:
    """
    Container for all simulation parameters.
    Loaded from YAML configuration file.
    """
    # Central black hole
    M_central: float  # kg
    
    # Ring configurations (all in SI units after loading)
    ring_configs: list  # List of ring configurations
    
    # Debris field
    debris_count: int
    debris_v_min: float  # m/s
    debris_v_max: float  # m/s
    # Note: debris_mass_per_particle is derived as M_central / debris_count
    
    # Simulation control
    dt: float  # seconds
    duration: float  # seconds
    output_interval: float  # seconds
    
    # Physics options
    include_debris_gravity: bool
    use_relativistic_mass: bool
    
    @classmethod
    def from_yaml(cls, filepath: str):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert to SI units
        M_sun = 1.989e30  # kg
        Gly_to_m = 9.461e24  # m
        c = 299792458  # m/s
        Gyr_to_s = 3.1557e16  # s
        
        # Parse and convert units
        # ... (implementation details)
        
        return cls(...)

# NumPy arrays for simulation state (Numba-compatible)

