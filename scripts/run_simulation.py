"""
Main simulation runner script.

Usage:
    python scripts/run_simulation.py configs/baseline_config.yaml

This script:
1. Loads configuration from YAML file
2. Initializes simulation state
3. Runs the simulation with progress bar
4. Saves results to HDF5 file
5. Generates plots and summary report
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path so we can import bhe package
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bhe.config import SimulationParameters
from bhe.initialization import initialize_simulation
from bhe.evolution import evolve_system
from bhe.output import SimulationRecorder
from bhe.visualization import (
    plot_redshift_vs_distance,
    plot_proper_time_vs_redshift,
    plot_ring0_trajectories_3d,
    plot_escape_fraction_vs_time,
    generate_summary_report
)
from bhe.analysis import analyze_simulation


def main():
    parser = argparse.ArgumentParser(
        description='Run black hole explosion simulation'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HDF5 file path (default: auto from config)'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable profiling'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    params = SimulationParameters.from_yaml(args.config)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate from config
        output_dir = Path(params.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{params.simulation_name}.h5")

    print(f"Output will be saved to: {output_path}")
    print()

    # Print configuration summary
    print("=" * 70)
    print(f"SIMULATION: {params.simulation_name}")
    print("=" * 70)
    print(f"Central BH mass: {params.M_central:.2e} M_sun")
    print(f"Debris particles: {params.debris_count}")
    # Count Ring 0 BHs (ring_id == 0)
    ring0_count = sum(r.count for r in params.rings if r.ring_id == 0)
    print(f"Ring 0 BHs: {ring0_count}")
    print(f"Total BHs: {sum(r.count for r in params.rings)}")

    # Import constants for unit conversions
    from bhe import constants as const
    yr_to_Gyr = 1.0e9  # Convert years to gigayears
    print(f"Duration: {params.duration / yr_to_Gyr:.1f} Gyr")
    print(f"Timestep: {params.dt / yr_to_Gyr:.4f} Gyr")
    print(f"Output interval: {params.output_interval / yr_to_Gyr:.2f} Gyr")
    print(f"Newtonian enhancements: {params.use_newtonian_enhancements}")
    print("=" * 70)
    print()

    # Initialize simulation
    print("Initializing simulation state...")
    state = initialize_simulation(params)
    print(f"Initialized {state.n_debris} debris particles")
    print(f"Initialized {state.n_bh} black holes")
    print(f"Total particles: {state.n_total}")
    print()

    # Run simulation
    print("Starting simulation...")
    start_time = time.time()

    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()

    # Calculate number of timesteps
    n_steps = int(params.duration / params.dt)

    with SimulationRecorder(output_path, params, state) as recorder:
        evolve_system(state, params, n_steps, show_progress=True, recorder=recorder)

    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print("\n" + "=" * 70)
        print("PROFILING RESULTS (Top 20 functions)")
        print("=" * 70)
        stats.print_stats(20)

    elapsed_time = time.time() - start_time

    print()
    print("=" * 70)
    print(f"Simulation completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("=" * 70)
    print()

    # Analyze results
    print("Analyzing results...")
    results = analyze_simulation(output_path)

    print()
    print("=" * 70)
    print("QUICK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total debris particles: {results['n_debris_total']}")
    print(f"Accreted particles: {results['n_debris_accreted']} ({100.0 * results['n_debris_accreted']/results['n_debris_total']:.1f}%)")
    print(f"Escaped particles (>100 Gly): {results['n_debris_escaped']} ({100.0 * results['escape_fraction']:.1f}%)")
    if results['n_debris_escaped'] > 0:
        print(f"Mean redshift (escaped): {results['redshift_mean']:.4f}")
        print(f"Mean proper time (escaped): {results['proper_time_mean_gyr']:.2f} Gyr")
    print(f"Energy conservation error: {100.0 * results['energy_conservation_error']:.4f}%")
    print(f"Momentum conservation error: {100.0 * results['momentum_conservation_error']:.4f}%")
    print("=" * 70)
    print()

    # Generate plots and report
    if not args.skip_plots:
        print("Generating plots and report...")
        output_dir = Path(output_path).parent

        # Generate plots
        plot_files = {
            'redshift_distance': output_dir / 'redshift_vs_distance.png',
            'proper_time_redshift': output_dir / 'proper_time_vs_redshift.png',
            'ring0_trajectories': output_dir / 'ring0_trajectories_3d.png',
            'escape_fraction': output_dir / 'escape_fraction_vs_time.png'
        }

        try:
            plot_redshift_vs_distance(output_path, str(plot_files['redshift_distance']))
            print(f"  [OK] {plot_files['redshift_distance'].name}")
        except Exception as e:
            print(f"  [ERROR] redshift_vs_distance: {e}")

        try:
            plot_proper_time_vs_redshift(output_path, str(plot_files['proper_time_redshift']))
            print(f"  [OK] {plot_files['proper_time_redshift'].name}")
        except Exception as e:
            print(f"  [ERROR] proper_time_vs_redshift: {e}")

        try:
            plot_ring0_trajectories_3d(output_path, str(plot_files['ring0_trajectories']))
            print(f"  [OK] {plot_files['ring0_trajectories'].name}")
        except Exception as e:
            print(f"  [ERROR] ring0_trajectories_3d: {e}")

        try:
            plot_escape_fraction_vs_time(output_path, str(plot_files['escape_fraction']))
            print(f"  [OK] {plot_files['escape_fraction'].name}")
        except Exception as e:
            print(f"  [ERROR] escape_fraction_vs_time: {e}")

        # Generate summary report
        report_path = output_dir / 'summary_report.txt'
        try:
            generate_summary_report(output_path, str(report_path))
            print(f"  [OK] {report_path.name}")
        except Exception as e:
            print(f"  [ERROR] summary_report: {e}")

        print()
        print(f"All outputs saved to: {output_dir}")
    else:
        print("Skipping plot generation (--skip-plots)")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
