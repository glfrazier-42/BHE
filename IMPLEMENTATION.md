# Black Hole Explosion Simulation - Implementation Guide

This document provides practical guidance for implementing the simulation described in SPECIFICATION.md.

## Project Structure

Recommended directory layout:

```
black_hole_explosion/
├── README.md
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Poetry config (alternative)
├── SPECIFICATION.md         # Physics and algorithms
├── IMPLEMENTATION.md        # This file
├── CLAUDE.md               # Instructions for Claude Code
├── configs/
│   ├── baseline_config.yaml
│   ├── phase2_config.yaml
│   └── sweep_config.yaml
├── src/
│   ├── __init__.py
│   ├── constants.py         # Physical constants
│   ├── physics.py           # Numba-compiled physics functions
│   ├── config.py            # SimulationParameters class
│   ├── state.py             # SimulationState class
│   ├── initialization.py    # Initial condition setup
│   ├── evolution.py         # Time evolution engine
│   ├── accretion.py         # Accretion detection and handling
│   ├── output.py            # HDF5 data recording
│   ├── analysis.py          # Post-simulation analysis
│   └── visualization.py     # Plotting and animation
├── tests/
│   ├── test_physics.py
│   ├── test_initialization.py
│   └── test_evolution.py
├── scripts/
│   ├── run_simulation.py    # Main entry point
│   ├── run_sweep.py         # Parameter sweep
│   └── analyze_results.py   # Batch analysis
├── notebooks/
│   └── exploration.ipynb    # Interactive analysis
└── results/
    ├── baseline/
    ├── phase2/
    └── sweep/
```

## Dependencies

### requirements.txt

```
numpy>=1.24.0
numba>=0.57.0
pyyaml>=6.0
matplotlib>=3.7.0
h5py>=3.8.0
tqdm>=4.65.0
scipy>=1.10.0
pytest>=7.3.0
```

## Implementation Roadmap

### Stage 1: Core Physics Engine and Configuration
- [x] Set up Python project structure with pip requirements
- [x] Implement `SimulationParameters.from_yaml()` with unit conversion
- [x] Implement physical constants module
- [x] Implement and test Lorentz factor calculation (with Numba)
- [x] Implement and test relativistic mass calculation
- [x] Implement and test gravitational acceleration (with Numba)
- [x] Write unit tests for all physics functions
- [x] Verify Numba compilation is working correctly

**Deliverable**: `physics.py` module with JIT-compiled functions passing all tests ✅

### Stage 2: Initialization Module
- [x] Implement `SimulationState` class with NumPy arrays
- [x] Function to initialize Ring 0 in circular orbit
- [x] Function to initialize Rings 1-3 in static positions (2-arm spiral)
- [x] Function to sample debris particles uniformly over solid angle
- [x] Function to sample debris velocities from distribution
- [x] Validation: check initial energy, momentum, angular momentum
- [x] Save/load initial conditions to HDF5

**Deliverable**: `initialization.py` module that creates valid initial states ✅

### Stage 3: Time Evolution Engine (Direct N-body)
- [ ] Implement `update_debris_particles()` with Numba
- [ ] Implement Ring 0 BH evolution (orbital mechanics)
- [ ] Implement accretion detection (capture radius method)
- [ ] Implement momentum conservation in accretion events
- [ ] Implement main simulation loop with timesteps
- [ ] Add progress bar (tqdm) for long runs
- [ ] Test on small system (10 particles, 100 timesteps)

**Deliverable**: `evolution.py` module that evolves system state using direct N-body

### Stage 4: Data Recording and Checkpointing
- [ ] Implement time series recording to HDF5 (efficient for large data)
- [ ] Record: time, BH positions/velocities/masses, sample of debris states
- [ ] Implement checkpoint/restart functionality
- [ ] Implement energy and momentum conservation checks
- [ ] Log warnings when conservation violated by >1%
- [ ] Save configuration YAML with each output file for reproducibility

**Deliverable**: `output.py` module for data persistence

### Stage 5: Analysis and Visualization
- [ ] Calculate redshift from velocity: z = sqrt((1+β)/(1-β)) - 1
- [ ] Calculate proper time for each debris particle
- [ ] Plot: redshift vs. distance for debris particles
- [ ] Plot: proper time vs. redshift
- [ ] Plot: Ring 0 trajectories in 3D
- [ ] Plot: escape fraction vs. time
- [ ] Animate: 3D visualization of system evolution
- [ ] Generate summary statistics report

**Deliverable**: `analysis.py` and `visualization.py` modules

### Stage 6: Baseline Simulation (Phase 1)
- [ ] Create `baseline_config.yaml`
- [ ] Run baseline simulation: 1000 particles, 50 Gyr, no debris gravity
- [ ] Profile performance: identify bottlenecks
- [ ] Optimize if needed (larger Numba parallel loops, etc.)
- [ ] Analyze results: Does Ring 0 prevent collapse?
- [ ] Document findings in `results/baseline/README.md`

**Deliverable**: First complete simulation results with analysis

### Stage 7: Barnes-Hut Tree Implementation
- [ ] Implement octree construction with Numba-compatible arrays
- [ ] Implement tree traversal for force calculation
- [ ] Calculate aggregate mass and center of mass for nodes
- [ ] Implement opening angle criterion (theta check)
- [ ] Test with two-body problem (should match direct calculation)
- [ ] Test with Plummer sphere (check energy conservation)
- [ ] Benchmark: compare performance with direct N-body
- [ ] Validate: compare forces with direct calculation for varying theta

**Deliverable**: Barnes-Hut tree code enabling 100k+ particle simulations

### Stage 8: Add Debris Field Gravity (Phase 2)
- [ ] Enable Barnes-Hut for debris-debris interactions
- [ ] Test with small particle count first (1000 particles)
- [ ] Verify energy conservation with debris gravity enabled
- [ ] Run full simulation with debris gravity enabled
- [ ] Compare with baseline: deceleration → acceleration transition?
- [ ] Profile and optimize tree rebuild frequency

**Deliverable**: Phase 2 results showing debris field effects

### Stage 9: Parameter Exploration (Phase 3)
- [ ] Implement parameter sweep framework
- [ ] Load `sweep_config.yaml` and generate configurations
- [ ] Run simulations in parallel (multiprocessing or job array)
- [ ] Aggregate results across parameter space
- [ ] Generate comparative plots: success metrics vs. parameters
- [ ] Identify optimal parameter ranges

**Deliverable**: Comprehensive parameter study with recommendations

### Stage 10 (Optional): Advanced Features
- [ ] Adaptive timestep based on minimum approach distance
- [ ] GPU acceleration with CuPy or CUDA
- [ ] More sophisticated accretion model (gravitational focusing)
- [ ] Post-Newtonian corrections for Ring 0
- [ ] Time-dependent Ring 1-3 (orbits of BH galaxy arms)

**Deliverable**: Performance improvements and physics refinements

## Computational Considerations

### Timestep Selection (Critical for Stability)

The timestep must be small enough that particles don't "jump over" important regions:

```python
# Courant condition for Ring 0 (most stringent)
dt_ring0 = 0.1 * R_ring0 / v_ring0
# Example: R=3 Gly, v=0.8c → dt ~ 0.0004 Gyr

# Condition for debris near Ring 0
dt_debris = 0.1 * capture_radius / c
# Example: r_capture=0.5 Gly → dt ~ 0.00006 Gyr

# Use minimum of all constraints
dt = min(dt_ring0, dt_debris, dt_user_specified)
```

**Start conservatively small, increase if stable.**

### Memory Requirements

For 10,000 particles, 50 Gyr at 0.001 Gyr timesteps:
- Timesteps: 50,000
- Arrays: positions (3), velocities (3), proper_times (1) per particle
- Storage per timestep: ~10,000 * 7 * 8 bytes = 560 KB
- Total if saving every timestep: 560 KB * 50,000 = 28 GB

**Solution**: Only save every N timesteps (output_interval in config)
- Save every 0.1 Gyr (100 timesteps) → 280 MB
- Use HDF5 with compression → ~50-100 MB

For 100,000 particles (production runs):
- Saving every 0.1 Gyr → 2.8 GB uncompressed
- With HDF5 compression → ~500 MB - 1 GB

For 1,000,000 particles:
- Saving every 0.1 Gyr → 28 GB uncompressed
- With HDF5 compression → ~5-10 GB
- May need to save less frequently or use downsampling

### Parallelization Strategy

**Level 1: Numba parallel loops (easiest, built-in)**
```python
@jit(nopython=True, parallel=True)
def update_debris_particles(...):
    for i in prange(N):  # Automatic parallelization
        # Update particle i
```
Good for: 4-16 cores on single machine

**Level 2: Multiprocessing for parameter sweeps**
```python
from multiprocessing import Pool

def run_simulation(config_file):
    # Run one simulation
    ...

# Run multiple configs in parallel
with Pool(processes=8) as pool:
    results = pool.map(run_simulation, config_files)
```
Good for: Running 100s of parameter combinations

**Level 3 (future): GPU acceleration**
- Use CuPy (NumPy-like interface for CUDA)
- Or write custom CUDA kernels
- Can handle 100,000+ particles efficiently

### Expected Runtime Estimates

**Baseline simulation** (1,000 particles, 50 Gyr, dt=0.001 Gyr):
- Pure Python: ~10 hours
- Numba (single core): ~10 minutes
- Numba (8 cores): ~2 minutes

**Medium simulation** (10,000 particles, direct N-body):
- Numba (8 cores): ~20-30 minutes

**Large simulation** (100,000 particles, Barnes-Hut):
- Numba + Barnes-Hut: ~1-2 hours
- GPU acceleration: ~10-20 minutes

**Production simulation** (1,000,000 particles, Barnes-Hut):
- Numba + Barnes-Hut: ~10-20 hours
- GPU acceleration: ~2-3 hours

**Parameter sweep** (100 configurations, 1000 particles each):
- Sequential: 100 * 2 min = ~3 hours
- Parallel (8 cores): ~25 minutes

## Code Examples

### SimulationState Class

```python
import numpy as np

class SimulationState:
    """
    All arrays are NumPy arrays for Numba JIT compilation.
    """
    def __init__(self, params: SimulationParameters):
        # Black hole arrays (N_bh black holes)
        N_bh = params.total_bh_count
        self.bh_positions = np.zeros((N_bh, 3))  # meters
        self.bh_velocities = np.zeros((N_bh, 3))  # m/s
        self.bh_masses_rest = np.zeros(N_bh)  # kg
        self.bh_rings = np.zeros(N_bh, dtype=np.int32)  # ring ID
        self.bh_is_static = np.zeros(N_bh, dtype=np.bool_)
        self.bh_capture_radii = np.zeros(N_bh)  # meters (for Ring 0)

        # Debris particle arrays (N_debris particles)
        N_debris = params.debris_count
        self.debris_positions = np.zeros((N_debris, 3))  # meters
        self.debris_velocities = np.zeros((N_debris, 3))  # m/s
        self.debris_masses = np.zeros(N_debris)  # kg
        self.debris_proper_times = np.zeros(N_debris)  # seconds
        self.debris_is_accreted = np.zeros(N_debris, dtype=np.bool_)
        self.debris_accreted_by = np.full(N_debris, -1, dtype=np.int32)

        # Simulation metadata
        self.time = 0.0  # coordinate time in seconds
        self.timestep = 0
```

### Numba-Compatible Force Calculation

```python
from numba import jit, prange
import numpy as np

# Physical constants (global)
G = 6.674e-11  # m³/(kg·s²)
c = 299792458  # m/s
c_squared = c * c

@jit(nopython=True)
def lorentz_factor(velocity):
    """
    Calculate Lorentz factor γ for a 3D velocity vector.

    Args:
        velocity: 3D velocity array [vx, vy, vz] in m/s

    Returns:
        γ = 1/sqrt(1 - v²/c²)
    """
    v_squared = velocity[0]**2 + velocity[1]**2 + velocity[2]**2
    beta_squared = v_squared / c_squared

    # Clamp to avoid numerical issues near c
    if beta_squared >= 1.0:
        beta_squared = 0.9999

    return 1.0 / np.sqrt(1.0 - beta_squared)

@jit(nopython=True)
def calculate_acceleration(pos_i, bh_positions, bh_masses_rest,
                          bh_velocities, bh_is_static, use_relativistic):
    """
    Calculate gravitational acceleration on particle at pos_i.
    Compiled with Numba for speed.

    Args:
        pos_i: Position of particle [x, y, z]
        bh_positions: Array of BH positions (N_bh, 3)
        bh_masses_rest: Array of BH rest masses (N_bh,)
        bh_velocities: Array of BH velocities (N_bh, 3)
        bh_is_static: Array of static flags (N_bh,)
        use_relativistic: Boolean flag for relativistic mass

    Returns:
        acceleration: 3D acceleration vector [ax, ay, az]
    """
    accel = np.zeros(3)

    for j in range(len(bh_positions)):
        # Vector from particle to BH
        r_vec = bh_positions[j] - pos_i
        r_squared = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
        r = np.sqrt(r_squared)

        if r < 1e10:  # Minimum distance to avoid singularities (adjust as needed)
            continue

        # Effective mass (relativistic if moving)
        if use_relativistic and not bh_is_static[j]:
            gamma = lorentz_factor(bh_velocities[j])
            m_eff = gamma * bh_masses_rest[j]
        else:
            m_eff = bh_masses_rest[j]

        # Gravitational acceleration: a = G*M/r² * r_hat
        accel += G * m_eff / r_squared * (r_vec / r)

    return accel

@jit(nopython=True, parallel=True)
def update_debris_particles(debris_pos, debris_vel, debris_proper_times,
                           bh_positions, bh_masses_rest, bh_velocities,
                           bh_is_static, dt, use_relativistic):
    """
    Update all debris particles for one timestep.
    Parallelized with Numba for maximum speed.

    This is the computational bottleneck: O(N_debris * N_bh) operations.
    """
    N = len(debris_pos)

    for i in prange(N):
        # Calculate acceleration
        accel = calculate_acceleration(
            debris_pos[i], bh_positions, bh_masses_rest,
            bh_velocities, bh_is_static, use_relativistic
        )

        # Update velocity: v_new = v + a*dt
        debris_vel[i] += accel * dt

        # Update position: x_new = x + v*dt + 0.5*a*dt²
        debris_pos[i] += debris_vel[i] * dt + 0.5 * accel * dt * dt

        # Update proper time (time dilation)
        gamma = lorentz_factor(debris_vel[i])
        debris_proper_times[i] += dt / gamma
```

## Success Criteria

1. **Escape Validation**: >50% of debris particles escape to >100 Gly without being accreted
2. **Redshift Reproduction**: Debris particles exhibit z~6-14 at distances 50-100 Gly
3. **Proper Time**: Escaped particles have proper times >30 Gyr
4. **Energy Conservation**: Total energy (kinetic + potential + rest mass) conserved to within 1%
5. **Ring 0 Evolution**: Ring 0 BHs successfully migrate outward to ~100 Gly range (if Ring 0 is enabled)

## Parameters to Explore

### Phase 1: Baseline (No Debris Field Gravity, No Ring 0)
- Central BH mass: 4×10²² M☉ (approximately 200% visible universe)
- Ring 0: 0 BHs (disabled for baseline)
- Rings 1-3: As specified in config, static positions
- Debris: 1000 particles, velocities 0.01c to 0.92c
- Timestep: 0.001 Gyr (adjust for stability)
- Duration: 50 Gyr
- **Goal**: Understand baseline behavior without tugboats

### Phase 2: Add Ring 0 Tugboats
- Ring 0: 4-8 BHs, each 10²¹ M☉, at 2-5 Gly, velocity 0.7-0.9c
- **Goal**: Does Ring 0 prevent collapse? Do debris particles escape?

### Phase 3: Add Debris Field Gravity (Barnes-Hut)
- Include gravitational force from all debris particles using Barnes-Hut
- May need 10k-100k particles to see collective effects
- **Goal**: Does debris field create deceleration → acceleration transition?

### Phase 4: Parameter Sweep
Vary:
- Ring 0 radius: 2-5 Gly
- Ring 0 velocity: 0.7-0.9c
- Ring 0 count: 4-8 BHs
- Ring 0 mass: 10²⁰ - 10²¹ M☉
- Capture radius: 0.1-1 Gly
- Debris count: 1000-100000 particles
- Barnes-Hut opening angle: 0.3-0.7

## Development Notes

### Version Control
- Commit after each stage completion
- Tag major milestones (e.g., `v1.0-baseline`, `v2.0-barnes-hut`)
- Keep configuration files in version control
- Use `.gitignore` for results/ and large HDF5 files

### Testing Strategy
- Unit tests for all physics functions
- Integration tests for initialization and evolution
- Validation tests against known solutions (Kepler orbit, energy conservation)
- Performance regression tests (track runtime for standard problem)

### Documentation
- Docstrings for all public functions
- Comments explaining physics assumptions
- README.md with quick start guide
- Document any deviations from SPECIFICATION.md
