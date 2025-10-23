# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based N-body simulation exploring whether an exploding supermassive black hole surrounded by fast-moving orbiting black holes can produce observed galaxy redshifts through the "gravitational tugboat" effect.

## Key Architecture

**Core Hypothesis**: A supermassive black hole explodes, releasing debris that would normally collapse gravitationally. However, a system of orbiting black holes in close, high-speed orbits (~0.7-0.9c) acts as "tugboats" - accreting debris and dragging it outward to prevent collapse.

**Four-Ring Structure**:
- **Ring 0 (Tugboats)**: 2-5 Gly from center, 4-8 black holes at 0.7-0.9c velocity
- **Ring 1-3**: Static shells at 100, 150, 200 Gly providing gravitational structure

**Performance-Critical Design**: Uses Numba JIT compilation for near-C performance in force calculations. All hot loops must be Numba-compatible (NumPy arrays, no Python objects).

## Development Commands

**Environment Setup**:
```bash
pip install -r requirements.txt
```

**Key Dependencies** (from SPECIFICATION.md):
- NumPy (array operations)
- Numba (JIT compilation for performance)
- PyYAML (configuration files)
- Matplotlib (visualization)
- h5py (data storage)
- tqdm (progress bars)

**Testing**: 
- Use pytest for unit tests (tests/ directory)
- Focus on physics function validation and Numba compilation

## Configuration System

All simulation parameters are specified in YAML files in the configs/ directory:
- `baseline_config.yaml`: Phase 1 simulation without debris field gravity
- `phase2_config.yaml`: Phase 2 with debris-debris interactions
- `sweep_config.yaml`: Parameter exploration runs

Configuration includes:
- Central black hole mass (~4×10²² solar masses)
- Ring configurations (positions, masses, velocities)
- Debris field parameters (1000-10000 particles)
- Simulation control (timesteps, duration, output intervals)
- Physics options (relativistic mass, gravity interactions)

## Code Organization

**src/ Structure**:
- `physics.py`: Numba-compiled force calculations and relativistic mechanics
- `config.py`: SimulationParameters class with YAML loading
- `state.py`: SimulationState with NumPy arrays for Numba compatibility
- `initialization.py`: Setup of initial conditions (ring positions, debris sampling)
- `evolution.py`: Main time evolution engine
- `accretion.py`: Debris capture by black holes
- `output.py`: HDF5 data recording and checkpointing
- `analysis.py`: Post-simulation redshift and proper time calculations
- `visualization.py`: 3D plots and animations

**scripts/ Entry Points**:
- `run_simulation.py`: Main simulation runner
- `run_sweep.py`: Parameter sweep execution
- `analyze_results.py`: Batch analysis of results

## Critical Performance Considerations

**Numba Constraints**: All computational hot loops must be Numba-compatible:
- ✅ NumPy arrays, basic math, for loops
- ❌ Python objects, lists, dicts in compiled functions

**Timestep Selection**: Critical for stability
- Must satisfy Courant condition for Ring 0 BHs
- Typical values: 0.0001-0.001 Gyr
- Too large causes numerical instability

**Memory Management**: 
- 10k particles × 50k timesteps = ~28GB if saving all data
- Use output_interval to save subset of timesteps
- HDF5 compression reduces storage significantly

## Implementation Phases

**Phase 1**: Core physics engine without debris-debris gravity
**Phase 2**: Add gravitational interactions between debris particles  
**Phase 3**: Parameter exploration across Ring 0 configurations

## Success Criteria

- >50% debris escape to >100 Gly without accretion
- Redshift reproduction: z~6-14 at 50-100 Gly distances
- Energy conservation within 1%
- Ring 0 migration to ~100 Gly range

## Physics Implementation Notes

**Units**: All internal calculations in SI units (meters, seconds, kg)
**Relativistic Effects**: Lorentz factor γ for moving black holes
**Force Calculation**: Standard Newtonian gravity with relativistic mass corrections
**Accretion Model**: Capture radius approach (debris within r_capture gets accreted)