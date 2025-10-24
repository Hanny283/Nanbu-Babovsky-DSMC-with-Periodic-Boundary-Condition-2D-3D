import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys
import os

# Add the parent directory to the path to import from sims
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sims.arbitrary_shape2d import Arbitrary_Shape_2D
from helpers import create_arbitrary_shape_mesh_2d

def create_star_shape(center_x=0, center_y=0, outer_radius=5, inner_radius=2.5, num_points=5):
    """Create a star shape with specified parameters."""
    points = []
    
    for i in range(2 * num_points):
        angle = i * np.pi / num_points
        
        if i % 2 == 0:
            # Outer points
            x = center_x + outer_radius * np.cos(angle)
            y = center_y + outer_radius * np.sin(angle)
        else:
            # Inner points
            x = center_x + inner_radius * np.cos(angle)
            y = center_y + inner_radius * np.sin(angle)
        
        points.append((x, y))
    
    return points

def create_hexagon_shape(center_x=0, center_y=0, radius=4):
    """Create a regular hexagon shape."""
    points = []
    
    for i in range(6):
        angle = i * np.pi / 3
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append((x, y))
    
    return points

def create_square_shape(center_x=0, center_y=0, side_length=4):
    """Create a square shape."""
    half_side = side_length / 2
    points = [
        (center_x - half_side, center_y - half_side),  # Bottom left
        (center_x + half_side, center_y - half_side),  # Bottom right
        (center_x + half_side, center_y + half_side),  # Top right
        (center_x - half_side, center_y + half_side)   # Top left
    ]
    return points

def plot_boundary(points, ax, color='black', linewidth=2):
    """Plot the boundary of the arbitrary shape."""
    # Close the polygon by adding the first point at the end
    closed_points = points + [points[0]]
    
    x_coords = [p[0] for p in closed_points]
    y_coords = [p[1] for p in closed_points]
    
    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, label='Boundary')

def plot_mesh_boundary(mesh, ax, color='red', linewidth=2, alpha=0.8):
    """
    Plot the boundary of the mesh generated from arbitrary points.
    """
    # Get boundary edges from mesh
    boundary_edges = []
    for cell in mesh.cells:
        if cell.type == "line":
            boundary_edges.append(cell.data)
    
    if boundary_edges:
        # Plot boundary edges
        for edge_indices in boundary_edges:
            for edge in edge_indices:
                if len(edge) >= 2:
                    p1 = mesh.points[edge[0]]
                    p2 = mesh.points[edge[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=color, linewidth=linewidth, alpha=alpha)

def plot_triangular_mesh(mesh, ax, color='black', linewidth=0.8, alpha=0.6):
    """
    Plot the triangular mesh cells.
    """
    # Get triangle cells
    triangle_cells = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
            break
    
    if triangle_cells is not None:
        # Plot each triangle
        for triangle in triangle_cells:
            if len(triangle) >= 3:
                # Get the three vertices of the triangle
                p1 = mesh.points[triangle[0]]
                p2 = mesh.points[triangle[1]]
                p3 = mesh.points[triangle[2]]
                
                # Plot the three edges of the triangle
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color=color, linewidth=linewidth, alpha=alpha)
                ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 
                       color=color, linewidth=linewidth, alpha=alpha)
                ax.plot([p3[0], p1[0]], [p3[1], p1[1]], 
                       color=color, linewidth=linewidth, alpha=alpha)

def plot_mesh_and_boundary(points, ax, show_mesh=True, show_boundary=True, 
                          mesh_color='black', boundary_color='red', 
                          mesh_linewidth=0.8, boundary_linewidth=2, alpha=0.6):
    """
    Plot both the mesh and boundary for arbitrary points.
    """
    # Create mesh from points
    mesh = create_arbitrary_shape_mesh_2d(100, points)  # N=100 is just for mesh generation
    
    if show_mesh:
        plot_triangular_mesh(mesh, ax, color=mesh_color, linewidth=mesh_linewidth, alpha=alpha)
    
    if show_boundary:
        plot_mesh_boundary(mesh, ax, color=boundary_color, linewidth=boundary_linewidth, alpha=0.8)
    
    # Also plot the original boundary points for reference
    plot_boundary(points, ax, color='black', linewidth=1)

def main():
    print("=== Arbitrary Shape 2D DSMC Simulation Demo ===")
    
    # Simulation parameters (reduced for demo)
    N = 500  # Number of particles
    dt = 0.01  # Time step
    n_tot = 50  # Total number of time steps
    e = 1.0  # Energy parameter
    mu = 1.0  # Mass parameter
    alpha = 1.0  # Alpha parameter for VHS cross section
    T_x0 = 1.0  # Initial x-temperature
    T_y0 = 1.0  # Initial y-temperature
    buckets_x = 8  # Number of buckets in x direction
    buckets_y = 8  # Number of buckets in y direction
    
    # Test different shapes
    shapes_to_test = [
        ("Star", create_star_shape(outer_radius=4, inner_radius=2, num_points=5)),
        ("Hexagon", create_hexagon_shape(radius=3)),
        ("Square", create_square_shape(side_length=6))
    ]
    
    for shape_name, over_points in shapes_to_test:
        print(f"\nTesting {shape_name} shape...")
        
        # Placeholder positions (will be re-initialized inside Arbitrary_Shape_2D)
        positions = np.zeros((N, 2))
        radius = 1.0  # Not used for arbitrary shapes, but required by function signature
        
        print(f"Running simulation with {N} particles for {n_tot} time steps...")
        print(f"Boundary shape: {len(over_points)} vertices")
        
        try:
            # Run the simulation
            positions, velocities, temperature_history = Arbitrary_Shape_2D(
                N=N,
                points=over_points,
                positions=positions,
                radius=radius,
                T_x0=T_x0,
                T_y0=T_y0,
                dt=dt,
                n_tot=n_tot,
                e=e,
                mu=mu,
                alpha=alpha,
                buckets_x=buckets_x,
                buckets_y=buckets_y,
            )
            
            print(f"{shape_name} simulation completed successfully!")
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Final particle positions with mesh visualization
            ax1.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.6, c='blue', label='Particles')
            
            # Plot mesh and boundary
            plot_mesh_and_boundary(over_points, ax1, show_mesh=True, show_boundary=True)
            
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title(f'{shape_name} - Final Particle Positions with Mesh')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Plot 2: Temperature history
            time_steps = np.arange(n_tot)
            ax2.plot(time_steps, temperature_history, 'r-', linewidth=2, label='Temperature')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Temperature')
            ax2.set_title(f'{shape_name} - Temperature Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Print some statistics
            print(f"\n{shape_name} Simulation Statistics:")
            print(f"Final temperature: {temperature_history[-1]:.4f}")
            print(f"Average temperature: {np.mean(temperature_history):.4f}")
            print(f"Temperature std: {np.std(temperature_history):.4f}")
            print(f"Final particle count: {len(positions)}")
            
        except Exception as e:
            print(f"Error running {shape_name} simulation: {e}")
            continue

if __name__ == "__main__":
    main()
