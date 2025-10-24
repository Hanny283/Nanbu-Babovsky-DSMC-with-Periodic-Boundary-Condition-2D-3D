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
    """
    Create a star shape with specified parameters.
    
    Parameters:
    center_x, center_y: center of the star
    outer_radius: radius of the outer points
    inner_radius: radius of the inner points
    num_points: number of points on the star
    
    Returns:
    List of (x, y) tuples defining the star boundary
    """
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
    """
    Create a regular hexagon shape.
    
    Parameters:
    center_x, center_y: center of the hexagon
    radius: radius of the hexagon
    
    Returns:
    List of (x, y) tuples defining the hexagon boundary
    """
    points = []
    
    for i in range(6):
        angle = i * np.pi / 3
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append((x, y))
    
    return points

def create_custom_shape():
    """
    Create a custom arbitrary shape - a house-like shape.
    
    Returns:
    List of (x, y) tuples defining the custom boundary
    """
    points = [
        (-3, -2),  # Bottom left
        (3, -2),   # Bottom right
        (3, 1),    # Top right
        (1, 3),    # Peak right
        (0, 4),    # Peak center
        (-1, 3),   # Peak left
        (-3, 1)    # Top left
    ]
    return points

def create_square_shape(center_x=0, center_y=0, side_length=4):
    """
    Create a square shape.
    
    Parameters:
    center_x, center_y: center of the square
    side_length: length of each side
    
    Returns:
    List of (x, y) tuples defining the square boundary
    """
    half_side = side_length / 2
    points = [
        (center_x - half_side, center_y - half_side),  # Bottom left
        (center_x + half_side, center_y - half_side),  # Bottom right
        (center_x + half_side, center_y + half_side),  # Top right
        (center_x - half_side, center_y + half_side)   # Top left
    ]
    return points

def create_triangle_shape(center_x=0, center_y=0, side_length=4):
    """
    Create an equilateral triangle shape.
    
    Parameters:
    center_x, center_y: center of the triangle
    side_length: length of each side
    
    Returns:
    List of (x, y) tuples defining the triangle boundary
    """
    height = side_length * np.sqrt(3) / 2
    points = [
        (center_x, center_y + 2*height/3),           # Top vertex
        (center_x - side_length/2, center_y - height/3),  # Bottom left
        (center_x + side_length/2, center_y - height/3)   # Bottom right
    ]
    return points

def plot_boundary(points, ax, color='black', linewidth=2):
    """
    Plot the boundary of the arbitrary shape.
    
    Parameters:
    points: List of (x, y) tuples defining the boundary
    ax: matplotlib axes object
    color: color of the boundary line
    linewidth: width of the boundary line
    """
    # Close the polygon by adding the first point at the end
    closed_points = points + [points[0]]
    
    x_coords = [p[0] for p in closed_points]
    y_coords = [p[1] for p in closed_points]
    
    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, label='Boundary')

def plot_mesh_boundary(mesh, ax, color='red', linewidth=2, alpha=0.8):
    """
    Plot the boundary of the mesh generated from arbitrary points.
    
    Parameters:
    mesh: pygmsh mesh object
    ax: matplotlib axes object
    color: color of the boundary line
    linewidth: width of the boundary line
    alpha: transparency of the boundary line
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
    
    Parameters:
    mesh: pygmsh mesh object
    ax: matplotlib axes object
    color: color of the mesh lines
    linewidth: width of the mesh lines
    alpha: transparency of the mesh lines
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
    
    Parameters:
    points: List of (x, y) tuples defining the boundary
    ax: matplotlib axes object
    show_mesh: whether to show triangular mesh
    show_boundary: whether to show mesh boundary
    mesh_color: color of the mesh lines
    boundary_color: color of the boundary lines
    mesh_linewidth: width of the mesh lines
    boundary_linewidth: width of the boundary lines
    alpha: transparency
    """
    # Create mesh from points
    mesh = create_arbitrary_shape_mesh_2d(100, points)  # N=100 is just for mesh generation
    
    if show_mesh:
        plot_triangular_mesh(mesh, ax, color=mesh_color, linewidth=mesh_linewidth, alpha=alpha)
    
    if show_boundary:
        plot_mesh_boundary(mesh, ax, color=boundary_color, linewidth=boundary_linewidth, alpha=0.8)
    
    # Also plot the original boundary points for reference
    plot_boundary(points, ax, color='black', linewidth=1)

def get_user_shape_choice():
    """
    Get user input for shape choice.
    Returns the boundary points for the selected shape.
    """
    print("\nAvailable shapes:")
    print("1. star - 5-pointed star")
    print("2. hexagon - regular hexagon")
    print("3. square - square shape")
    print("4. triangle - equilateral triangle")
    print("5. house - house-like shape")
    print("6. custom - enter your own points")
    
    while True:
        choice = input("\nEnter your choice (1-6) or shape name: ").strip().lower()
        
        if choice in ['1', 'star']:
            outer_radius = float(input("Enter outer radius (default 5): ") or "5")
            inner_radius = float(input("Enter inner radius (default 2.5): ") or "2.5")
            num_points = int(input("Enter number of points (default 5): ") or "5")
            return create_star_shape(outer_radius=outer_radius, inner_radius=inner_radius, num_points=num_points)
            
        elif choice in ['2', 'hexagon']:
            radius = float(input("Enter radius (default 4): ") or "4")
            return create_hexagon_shape(radius=radius)
            
        elif choice in ['3', 'square']:
            side_length = float(input("Enter side length (default 4): ") or "4")
            return create_square_shape(side_length=side_length)
            
        elif choice in ['4', 'triangle']:
            side_length = float(input("Enter side length (default 4): ") or "4")
            return create_triangle_shape(side_length=side_length)
            
        elif choice in ['5', 'house']:
            return create_custom_shape()
            
        elif choice in ['6', 'custom']:
            return get_custom_points()
            
        else:
            print("Invalid choice. Please try again.")

def get_custom_points():
    """
    Get custom boundary points from user input.
    """
    print("\nEnter boundary points. Format: x1,y1 x2,y2 x3,y3 ...")
    print("Example: 0,0 2,0 2,2 0,2")
    print("Press Enter when done.")
    
    while True:
        try:
            points_input = input("Enter points: ").strip()
            if not points_input:
                print("Please enter at least 3 points.")
                continue
                
            # Parse the input
            point_strings = points_input.split()
            points = []
            
            for point_str in point_strings:
                x, y = map(float, point_str.split(','))
                points.append((x, y))
            
            if len(points) < 3:
                print("Please enter at least 3 points to form a polygon.")
                continue
                
            print(f"Created polygon with {len(points)} vertices:")
            for i, (x, y) in enumerate(points):
                print(f"  Point {i+1}: ({x}, {y})")
                
            confirm = input("Is this correct? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return points
            else:
                continue
                
        except ValueError:
            print("Invalid format. Please use format: x1,y1 x2,y2 x3,y3 ...")
        except Exception as e:
            print(f"Error: {e}")

def get_simulation_parameters():
    """
    Get simulation parameters from user input.
    Returns a dictionary with simulation parameters.
    """
    print("\nSimulation Parameters:")
    print("Press Enter to use default values.")
    
    N = int(input("Number of particles (default 2000): ") or "2000")
    dt = float(input("Time step (default 0.01): ") or "0.01")
    n_tot = int(input("Total time steps (default 100): ") or "100")
    e = float(input("Energy parameter (default 1.0): ") or "1.0")
    mu = float(input("Mass parameter (default 1.0): ") or "1.0")
    alpha = float(input("Alpha parameter (default 1.0): ") or "1.0")
    T_x0 = float(input("Initial x-temperature (default 1.0): ") or "1.0")
    T_y0 = float(input("Initial y-temperature (default 1.0): ") or "1.0")
    buckets_x = int(input("Number of buckets in x direction (default 10): ") or "10")
    buckets_y = int(input("Number of buckets in y direction (default 10): ") or "10")
    
    return {
        'N': N, 'dt': dt, 'n_tot': n_tot, 'e': e, 'mu': mu, 'alpha': alpha,
        'T_x0': T_x0, 'T_y0': T_y0, 'buckets_x': buckets_x, 'buckets_y': buckets_y
    }

def main():
    print("=== Arbitrary Shape 2D DSMC Simulation ===")
    
    # Get user's shape choice
    points = get_user_shape_choice()
    
    # Get simulation parameters
    params = get_simulation_parameters()
    
    # Extract parameters
    N = params['N']
    dt = params['dt']
    n_tot = params['n_tot']
    e = params['e']
    mu = params['mu']
    alpha = params['alpha']
    T_x0 = params['T_x0']
    T_y0 = params['T_y0']
    buckets_x = params['buckets_x']
    buckets_y = params['buckets_y']
    
    # Placeholder positions (will be re-initialized inside Arbitrary_Shape_2D)
    positions = np.zeros((N, 2))
    radius = 1.0  # Not used for arbitrary shapes, but required by function signature
    
    print(f"Running simulation with {N} particles for {n_tot} time steps...")
    print(f"Boundary shape: {len(points)} vertices")
    
    # Run the simulation
    positions, velocities, temperature_history = Arbitrary_Shape_2D(
        N=N,
        points=points,
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
    
    print("Simulation completed!")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Final particle positions with mesh visualization
    ax1.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.6, c='blue', label='Particles')
    
    # Plot mesh and boundary
    plot_mesh_and_boundary(points, ax1, show_mesh=True, show_boundary=True)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Final Particle Positions with Mesh')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Temperature history
    time_steps = np.arange(n_tot)
    ax2.plot(time_steps, temperature_history, 'r-', linewidth=2, label='Temperature')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Temperature Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nSimulation Statistics:")
    print(f"Final temperature: {temperature_history[-1]:.4f}")
    print(f"Average temperature: {np.mean(temperature_history):.4f}")
    print(f"Temperature std: {np.std(temperature_history):.4f}")
    print(f"Final particle count: {len(positions)}")

def create_animated_visualization():
    """
    Create an animated visualization of the simulation.
    Note: This is more computationally intensive and requires storing intermediate states.
    """
    # Simulation parameters (reduced for animation)
    N = 500
    dt = 0.01
    n_tot = 50
    e = 1.0
    mu = 1.0
    alpha = 1.0
    T_x0 = 1.0
    T_y0 = 1.0
    buckets_x = 8
    buckets_y = 8
    
    # Create boundary points
    points = create_star_shape(outer_radius=4, inner_radius=2, num_points=6)
    
    # Store positions at each time step for animation
    positions_history = []
    velocities_history = []
    temperature_history = []
    
    print("Running simulation for animation...")
    
    # Run simulation step by step to store history
    positions = np.zeros((N, 2))
    radius = 1.0
    
    # Initialize particles
    from helpers import create_arbitrary_shape_mesh_2d, assign_positions_arbitrary_2d, sample_velocities_from_maxwellian_2d
    
    mesh = create_arbitrary_shape_mesh_2d(N, points)
    positions = assign_positions_arbitrary_2d(N, mesh)
    velocities = sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)
    
    # Store initial state
    positions_history.append(positions.copy())
    velocities_history.append(velocities.copy())
    temperature_history.append(np.sum(velocities**2) / velocities.shape[0])
    
    # Run simulation steps
    for n in range(n_tot):
        positions, velocities, temp_history = Arbitrary_Shape_2D(
            N=N, points=points, positions=positions, radius=radius,
            T_x0=T_x0, T_y0=T_y0, dt=dt, n_tot=1, e=e, mu=mu, alpha=alpha,
            buckets_x=buckets_x, buckets_y=buckets_y
        )
        
        positions_history.append(positions.copy())
        velocities_history.append(velocities.copy())
        temperature_history.append(temp_history[0])
        
        if (n + 1) % 10 == 0:
            print(f"Completed {n + 1}/{n_tot} time steps")
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame):
        ax.clear()
        
        # Plot particles
        pos = positions_history[frame]
        ax.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.6, c='blue')
        
        # Plot boundary
        plot_boundary(points, ax)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Particle Simulation - Time Step {frame}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set consistent axis limits
        all_positions = np.vstack(positions_history)
        margin = 1.0
        ax.set_xlim(all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin)
        ax.set_ylim(all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(positions_history), 
                        interval=100, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == "__main__":
    # Run the main simulation
    main()
    
    # Uncomment the line below to run the animated version
    # anim = create_animated_visualization()
