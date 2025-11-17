import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import from sims
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sims.arbitrary_parameterized import Arbitrary_Shape_Parameterized, sample_star_shape, create_mesh_from_star_shape

def plot_boundary(points, ax, color='black', linewidth=2, label='Boundary'):
    """
    Plot the boundary of the star shape.
    
    Parameters:
    points: Array of (x, y) coordinates defining the boundary
    ax: matplotlib axes object
    color: color of the boundary line
    linewidth: width of the boundary line
    label: label for the boundary
    """
    # Close the polygon by adding the first point at the end
    closed_points = np.vstack([points, points[0]])
    
    ax.plot(closed_points[:, 0], closed_points[:, 1], 
            color=color, linewidth=linewidth, label=label)

def plot_triangular_mesh(mesh, ax, color='gray', linewidth=0.5, alpha=0.3):
    """
    Plot the triangular mesh cells.
    
    Parameters:
    mesh: pygmsh mesh object
    ax: matplotlib axes object
    color: color of the mesh lines
    linewidth: width of the mesh lines
    alpha: transparency of the mesh lines
    """
    tri_conn = mesh.get_cells_type("triangle")
    pts2d = mesh.points[:, :2]
    
    # Plot each triangle
    for tri in tri_conn:
        if len(tri) >= 3:
            # Get the three vertices of the triangle
            p1 = pts2d[tri[0]]
            p2 = pts2d[tri[1]]
            p3 = pts2d[tri[2]]
            
            # Plot the three edges of the triangle
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)
            ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)
            ax.plot([p3[0], p1[0]], [p3[1], p1[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)

# Hard-coded Fourier coefficients for star shape
# 5-pointed star: c0=3.0, c1=1.0, c6=0.5 (M=5, so c6 is c_{M+1})
# This gives us 2M+1 = 11 coefficients
FOURIER_COEFFICIENTS = np.array([3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

# Hard-coded simulation parameters
SIMULATION_PARAMS = {
    'N': 2000,                    # Number of particles
    'dt': 0.01,                   # Time step
    'n_tot': 100,                 # Total time steps
    'e': 1.0,                     # Energy parameter
    'mu': 1.0,                    # Mass parameter
    'alpha': 1.0,                 # Alpha parameter
    'T_x0': 1.0,                  # Initial x-temperature
    'T_y0': 1.0,                  # Initial y-temperature
    'num_boundary_points': 100    # Number of boundary points
}

def plot_velocity_field(positions, velocities, ax, scale=0.1):
    """
    Plot velocity vectors as arrows.
    
    Parameters:
    positions: array of particle positions
    velocities: array of particle velocities
    ax: matplotlib axes object
    scale: scaling factor for arrow length
    """
    if len(positions) == 0:
        return
    
    # Sample particles for visualization (too many arrows can be cluttered)
    if len(positions) > 500:
        indices = np.random.choice(len(positions), 500, replace=False)
        pos_sample = positions[indices]
        vel_sample = velocities[indices]
    else:
        pos_sample = positions
        vel_sample = velocities
    
    ax.quiver(pos_sample[:, 0], pos_sample[:, 1], 
              vel_sample[:, 0], vel_sample[:, 1],
              angles='xy', scale_units='xy', scale=1.0/scale, 
              alpha=0.6, width=0.003, color='red')

def main():
    print("=== Arbitrary Parameterized Shape 2D DSMC Simulation ===")
    print("This simulation uses Fourier coefficients to define star-shaped boundaries.")
    
    # Use hard-coded Fourier coefficients
    fourier_coefficients = FOURIER_COEFFICIENTS
    M = (len(fourier_coefficients) - 1) // 2
    print(f"\nUsing {len(fourier_coefficients)} Fourier coefficients (M={M} modes)")
    print(f"Fourier coefficients: {fourier_coefficients}")
    
    # Use hard-coded simulation parameters
    params = SIMULATION_PARAMS
    
    # Extract parameters
    N = params['N']
    dt = params['dt']
    n_tot = params['n_tot']
    e = params['e']
    mu = params['mu']
    alpha = params['alpha']
    T_x0 = params['T_x0']
    T_y0 = params['T_y0']
    num_boundary_points = params['num_boundary_points']
    
    print(f"\nSimulation parameters:")
    print(f"  Number of particles: {N}")
    print(f"  Time step: {dt}")
    print(f"  Total time steps: {n_tot}")
    print(f"  Energy parameter: {e}")
    print(f"  Mass parameter: {mu}")
    print(f"  Alpha parameter: {alpha}")
    print(f"  Initial temperatures: T_x0={T_x0}, T_y0={T_y0}")
    print(f"  Boundary points: {num_boundary_points}")
    
    # Generate boundary points for visualization
    boundary_points = sample_star_shape(fourier_coefficients, num_boundary_points)
    mesh = create_mesh_from_star_shape(boundary_points)
    
    # Placeholder positions (will be re-initialized inside Arbitrary_Shape_Parameterized)
    positions = np.zeros((N, 2))
    
    print(f"\nRunning simulation with {N} particles for {n_tot} time steps...")
    print(f"Boundary shape: {num_boundary_points} points, {M} Fourier modes")
    
    # Run the simulation
    positions, velocities, temperature_history = Arbitrary_Shape_Parameterized(
        N=N,
        fourier_coefficients=fourier_coefficients,
        num_boundary_points=num_boundary_points,
        positions=positions,
        T_x0=T_x0,
        T_y0=T_y0,
        dt=dt,
        n_tot=n_tot,
        e=e,
        mu=mu,
        alpha=alpha,
    )
    
    print("Simulation completed!")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Final particle positions with mesh
    ax1 = plt.subplot(2, 3, 1)
    if len(positions) > 0:
        ax1.scatter(positions[:, 0], positions[:, 1], s=2, alpha=0.6, c='blue', label='Particles')
    
    # Plot mesh
    plot_triangular_mesh(mesh, ax1, color='gray', linewidth=0.5, alpha=0.3)
    
    # Plot boundary
    plot_boundary(boundary_points, ax1, color='red', linewidth=2, label='Boundary')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Final Particle Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Velocity field
    ax2 = plt.subplot(2, 3, 2)
    if len(positions) > 0:
        # Color particles by speed
        speeds = np.linalg.norm(velocities, axis=1)
        scatter = ax2.scatter(positions[:, 0], positions[:, 1], 
                             c=speeds, s=2, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=ax2, label='Speed')
        
        # Plot velocity vectors
        plot_velocity_field(positions, velocities, ax2, scale=0.1)
    
    plot_boundary(boundary_points, ax2, color='red', linewidth=2)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Velocity Field (colored by speed)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Temperature history
    ax3 = plt.subplot(2, 3, 3)
    time_steps = np.arange(n_tot)
    ax3.plot(time_steps, temperature_history, 'r-', linewidth=2, label='Temperature')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Temperature Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Velocity distribution (x-component)
    ax4 = plt.subplot(2, 3, 4)
    if len(velocities) > 0:
        ax4.hist(velocities[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(np.mean(velocities[:, 0]), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(velocities[:, 0]):.3f}')
        ax4.set_xlabel('Velocity X-component')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Velocity X Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Velocity distribution (y-component)
    ax5 = plt.subplot(2, 3, 5)
    if len(velocities) > 0:
        ax5.hist(velocities[:, 1], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(np.mean(velocities[:, 1]), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(velocities[:, 1]):.3f}')
        ax5.set_xlabel('Velocity Y-component')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Velocity Y Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Speed distribution
    ax6 = plt.subplot(2, 3, 6)
    if len(velocities) > 0:
        speeds = np.linalg.norm(velocities, axis=1)
        ax6.hist(speeds, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax6.axvline(np.mean(speeds), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(speeds):.3f}')
        ax6.set_xlabel('Speed')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Speed Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n=== Simulation Statistics ===")
    print(f"Final temperature: {temperature_history[-1]:.4f}")
    print(f"Average temperature: {np.mean(temperature_history):.4f}")
    print(f"Temperature std: {np.std(temperature_history):.4f}")
    print(f"Final particle count: {len(positions)}")
    
    if len(velocities) > 0:
        speeds = np.linalg.norm(velocities, axis=1)
        print(f"\nVelocity Statistics:")
        print(f"Mean speed: {np.mean(speeds):.4f}")
        print(f"Std speed: {np.std(speeds):.4f}")
        print(f"Mean vx: {np.mean(velocities[:, 0]):.4f}")
        print(f"Mean vy: {np.mean(velocities[:, 1]):.4f}")

def create_animated_visualization():
    """
    Create an animated visualization of the simulation.
    Note: This requires modifying the simulation to store intermediate states.
    """
    print("\nAnimated visualization requires storing intermediate states.")
    print("This feature can be added by modifying the simulation function.")
    print("For now, please use the static visualization in main().")

if __name__ == "__main__":
    main()

