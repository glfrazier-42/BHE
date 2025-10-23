"""
Visualization functions for black hole explosion simulation.

This module provides functions to create plots and animations from simulation
HDF5 files:
- Redshift vs distance scatter plots
- Proper time vs redshift plots
- Ring 0 trajectory 3D plots
- Escape fraction vs time plots
- 3D system evolution animations
- Summary report generation

All plots are saved as PNG files with publication-quality settings (300 DPI).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Optional
import h5py

from bhe import constants as const
from bhe.analysis import (
    get_final_debris_state,
    get_ring0_trajectories,
    calculate_escape_fraction_vs_time,
    analyze_simulation
)


# Set publication-quality plot defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def plot_redshift_vs_distance(hdf5_filepath: str, output_path: str):
    """
    Create scatter plot of redshift vs distance for final debris state.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file
        output_path: Path to save PNG plot

    Creates a scatter plot with:
    - X-axis: Distance from origin (Gly)
    - Y-axis: Redshift z
    - Color: Blue for escaped particles, red for accreted
    """
    # Load final debris state
    state = get_final_debris_state(hdf5_filepath)

    # Convert to convenient units (distances already in ly, convert to Gly)
    distances_gly = state['distances'] / const.Gly
    redshifts = state['redshifts']
    accreted = state['accreted']

    # Separate escaped and accreted
    escaped_mask = ~accreted
    accreted_mask = accreted

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot escaped particles
    if np.any(escaped_mask):
        ax.scatter(distances_gly[escaped_mask], redshifts[escaped_mask],
                  c='blue', s=20, alpha=0.6, label='Escaped', edgecolors='none')

    # Plot accreted particles
    if np.any(accreted_mask):
        ax.scatter(distances_gly[accreted_mask], redshifts[accreted_mask],
                  c='red', s=20, alpha=0.6, label='Accreted', edgecolors='none')

    ax.set_xlabel('Distance from Center (Gly)')
    ax.set_ylabel('Redshift z')
    ax.set_title('Redshift vs Distance (Final State)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_proper_time_vs_redshift(hdf5_filepath: str, output_path: str):
    """
    Create scatter plot of proper time vs redshift for final debris state.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file
        output_path: Path to save PNG plot

    Creates a scatter plot with:
    - X-axis: Redshift z
    - Y-axis: Proper time (Gyr)
    - Color: Blue for escaped, red for accreted
    """
    # Load final debris state
    state = get_final_debris_state(hdf5_filepath)

    # Convert to convenient units
    redshifts = state['redshifts']
    proper_times_gyr = state['proper_times'] / 1.0e9  # yr to Gyr
    accreted = state['accreted']

    # Separate escaped and accreted
    escaped_mask = ~accreted
    accreted_mask = accreted

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot escaped particles
    if np.any(escaped_mask):
        ax.scatter(redshifts[escaped_mask], proper_times_gyr[escaped_mask],
                  c='blue', s=20, alpha=0.6, label='Escaped', edgecolors='none')

    # Plot accreted particles
    if np.any(accreted_mask):
        ax.scatter(redshifts[accreted_mask], proper_times_gyr[accreted_mask],
                  c='red', s=20, alpha=0.6, label='Accreted', edgecolors='none')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Proper Time (Gyr)')
    ax.set_title('Proper Time vs Redshift (Final State)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ring0_trajectories_3d(hdf5_filepath: str, output_path: str):
    """
    Create 3D plot of Ring 0 BH trajectories.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file
        output_path: Path to save PNG plot

    Creates a 3D plot showing:
    - Ring 0 BH trajectories as lines
    - Start positions as green markers
    - End positions as red markers
    - Coordinate axes with scale in Gly
    """
    # Load Ring 0 trajectories
    trajectories = get_ring0_trajectories(hdf5_filepath)

    if trajectories is None or trajectories['n_ring0'] == 0:
        # Create empty plot with message
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.text(0.5, 0.5, 0.5, 'No Ring 0 BHs in simulation',
               ha='center', va='center', fontsize=14)
        ax.set_xlabel('X (Gly)')
        ax.set_ylabel('Y (Gly)')
        ax.set_zlabel('Z (Gly)')
        ax.set_title('Ring 0 Trajectories')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    # Convert positions to Gly (positions already in ly)
    positions_gly = trajectories['positions'] / const.Gly
    n_timesteps, n_bh, _ = positions_gly.shape

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories for each BH
    for i in range(n_bh):
        x = positions_gly[:, i, 0]
        y = positions_gly[:, i, 1]
        z = positions_gly[:, i, 2]

        # Plot trajectory line
        ax.plot(x, y, z, '-', alpha=0.6, linewidth=1.5, label=f'BH {i}')

        # Mark start position (green)
        ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=100,
                  marker='o', edgecolors='black', linewidths=1)

        # Mark end position (red)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=100,
                  marker='s', edgecolors='black', linewidths=1)

    ax.set_xlabel('X (Gly)')
    ax.set_ylabel('Y (Gly)')
    ax.set_zlabel('Z (Gly)')
    ax.set_title('Ring 0 Black Hole Trajectories')

    # Add legend only if not too many BHs
    if n_bh <= 8:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Set equal aspect ratio
    # Find max range across all dimensions
    max_range = np.array([
        positions_gly[:, :, 0].max() - positions_gly[:, :, 0].min(),
        positions_gly[:, :, 1].max() - positions_gly[:, :, 1].min(),
        positions_gly[:, :, 2].max() - positions_gly[:, :, 2].min()
    ]).max() / 2.0

    mid_x = (positions_gly[:, :, 0].max() + positions_gly[:, :, 0].min()) / 2.0
    mid_y = (positions_gly[:, :, 1].max() + positions_gly[:, :, 1].min()) / 2.0
    mid_z = (positions_gly[:, :, 2].max() + positions_gly[:, :, 2].min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_escape_fraction_vs_time(hdf5_filepath: str, output_path: str,
                                  distance_threshold: float = 100.0):
    """
    Create line plot of escape fraction vs time.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file
        output_path: Path to save PNG plot
        distance_threshold: Escape distance threshold in Gly (default: 100 Gly)

    Creates a line plot with:
    - X-axis: Time (Gyr)
    - Y-axis: Fraction of debris beyond threshold
    """
    # Calculate escape fraction vs time
    times_gyr, escape_fractions = calculate_escape_fraction_vs_time(
        hdf5_filepath, distance_threshold
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot escape fraction
    ax.plot(times_gyr, escape_fractions, 'b-', linewidth=2)

    ax.set_xlabel('Time (Gyr)')
    ax.set_ylabel('Escape Fraction')
    ax.set_title(f'Fraction of Debris Beyond {distance_threshold:.0f} Gly vs Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def animate_system_evolution(hdf5_filepath: str, output_path: str,
                            frame_skip: int = 10, fps: int = 10,
                            distance_limit_gly: float = 200.0):
    """
    Create 3D animation of system evolution over time.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file
        output_path: Path to save animation (GIF or MP4)
        frame_skip: Use every Nth frame to reduce file size (default: 10)
        fps: Frames per second in output animation (default: 10)
        distance_limit_gly: Plot limits in Gly (default: 200 Gly)

    Saves animation showing debris and BH positions over time.
    """
    with h5py.File(hdf5_filepath, 'r') as f:
        times = f['timeseries/time'][::frame_skip]
        all_positions = f['timeseries/positions'][::frame_skip]
        all_accreted = f['timeseries/accreted'][::frame_skip]

        # Get metadata to filter debris and BHs
        from bhe.state import DEBRIS, BLACK_HOLE
        particle_type = f['metadata/particle_type'][:]
        debris_mask = (particle_type == DEBRIS)
        bh_mask = (particle_type == BLACK_HOLE)

    # Extract debris and BH positions
    debris_positions = all_positions[:, debris_mask, :]
    debris_accreted = all_accreted[:, debris_mask]
    bh_positions = all_positions[:, bh_mask, :]

    # Convert to Gly (positions already in ly, times in yr)
    times_gyr = times / 1.0e9
    debris_pos_gly = debris_positions / const.Gly
    bh_pos_gly = bh_positions / const.Gly

    n_frames = len(times)

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set fixed axis limits
    lim = distance_limit_gly
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X (Gly)')
    ax.set_ylabel('Y (Gly)')
    ax.set_zlabel('Z (Gly)')

    # Initialize scatter plots
    debris_scatter = ax.scatter([], [], [], c='blue', s=2, alpha=0.5, label='Debris')
    bh_scatter = ax.scatter([], [], [], c='red', s=100, marker='o',
                          edgecolors='black', linewidths=1, label='Black Holes')

    title = ax.set_title('')
    ax.legend()

    def update(frame):
        """Update function for animation."""
        # Get data for this frame
        debris_pos = debris_pos_gly[frame]
        accreted = debris_accreted[frame]
        bh_pos = bh_pos_gly[frame]

        # Update debris (only show non-accreted)
        active = ~accreted
        if np.any(active):
            debris_scatter._offsets3d = (
                debris_pos[active, 0],
                debris_pos[active, 1],
                debris_pos[active, 2]
            )
        else:
            debris_scatter._offsets3d = ([], [], [])

        # Update BHs
        if len(bh_pos) > 0:
            bh_scatter._offsets3d = (
                bh_pos[:, 0],
                bh_pos[:, 1],
                bh_pos[:, 2]
            )
        else:
            bh_scatter._offsets3d = ([], [], [])

        # Update title
        title.set_text(f'Time: {times_gyr[frame]:.2f} Gyr')

        return debris_scatter, bh_scatter, title

    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    # Save animation
    if output_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    else:
        # Default to GIF if extension not recognized
        anim.save(output_path + '.gif', writer=PillowWriter(fps=fps))

    plt.close()


def generate_summary_report(hdf5_filepath: str, output_path: str):
    """
    Generate text summary report of simulation results.

    Args:
        hdf5_filepath: Path to HDF5 simulation output file
        output_path: Path to save text report

    Creates a formatted text file with:
    - Simulation parameters
    - Final statistics (escape %, accretion %)
    - Redshift distribution
    - Proper time distribution
    - Energy/momentum conservation metrics
    """
    # Analyze simulation
    results = analyze_simulation(hdf5_filepath)

    # Create report
    lines = []
    lines.append("=" * 70)
    lines.append("BLACK HOLE EXPLOSION SIMULATION - SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Simulation info
    lines.append("SIMULATION INFORMATION")
    lines.append("-" * 70)
    lines.append(f"HDF5 File: {Path(hdf5_filepath).name}")
    lines.append(f"Final Time: {results['final_time_gyr']:.2f} Gyr")
    lines.append("")

    # Particle statistics
    lines.append("PARTICLE STATISTICS")
    lines.append("-" * 70)
    lines.append(f"Total Debris Particles: {results['n_debris_total']}")
    lines.append(f"Accreted Particles: {results['n_debris_accreted']} "
                f"({100.0 * results['n_debris_accreted']/results['n_debris_total']:.1f}%)")
    lines.append(f"Escaped Particles (>100 Gly): {results['n_debris_escaped']} "
                f"({100.0 * results['escape_fraction']:.1f}%)")
    lines.append("")

    # Redshift statistics
    lines.append("REDSHIFT DISTRIBUTION (Escaped Particles)")
    lines.append("-" * 70)
    if results['n_debris_escaped'] > 0:
        lines.append(f"Mean Redshift: {results['redshift_mean']:.4f}")
        lines.append(f"Std Dev Redshift: {results['redshift_std']:.4f}")
    else:
        lines.append("No escaped particles")
    lines.append("")

    # Proper time statistics
    lines.append("PROPER TIME DISTRIBUTION (Escaped Particles)")
    lines.append("-" * 70)
    if results['n_debris_escaped'] > 0:
        lines.append(f"Mean Proper Time: {results['proper_time_mean_gyr']:.2f} Gyr")
        lines.append(f"Std Dev Proper Time: {results['proper_time_std_gyr']:.2f} Gyr")
    else:
        lines.append("No escaped particles")
    lines.append("")

    # Conservation metrics
    lines.append("CONSERVATION METRICS")
    lines.append("-" * 70)
    lines.append(f"Energy Conservation Error: {100.0 * results['energy_conservation_error']:.4f}%")
    lines.append(f"Momentum Conservation Error: {100.0 * results['momentum_conservation_error']:.4f}%")

    if results['energy_conservation_error'] > 0.01:
        lines.append("WARNING: Energy conservation violated by >1%")
    if results['momentum_conservation_error'] > 0.01:
        lines.append("WARNING: Momentum conservation violated by >1%")

    lines.append("")
    lines.append("=" * 70)

    # Write report
    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text, encoding='ascii')
