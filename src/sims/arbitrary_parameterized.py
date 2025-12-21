import pygmsh 
import helpers as hf
import numpy as np
import periodic_bc as pb
import time
import cell_class as ct
import sys
import os
from scipy.spatial import cKDTree

# Add root directory to path to import edge_class
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
from edge_class import Edge

def sample_star_shape(c, N):
    """
    Generate N evenly spaced (x, y) points along the star-shaped boundary
    defined by equation (5): r(t;c) = c0 + Σ [c_m cos(mt) + c_{m+M} sin(mt)].
    
    Parameters
    ----------
    c : array-like of shape (2M+1,)
        Fourier coefficients [c0, c1..c_M, c_{M+1}..c_{2M}]
    N : int
        Number of sample points along the curve
    
    Returns
    -------
    points : ndarray of shape (N, 2)
        Array of (x, y) coordinates representing the boundary.
    """
    c = np.asarray(c)
    M = (len(c) - 1) // 2  # number of Fourier modes
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Compute r(t)
    r = c[0] * np.ones_like(t)
    for m in range(1, M + 1):
        r += c[m] * np.cos(m * t) + c[m + M] * np.sin(m * t)

    # Convert to Cartesian coordinates
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])


def create_mesh_from_star_shape(pts, mesh_size=0.1):
    """
    Create a mesh from star shape boundary points.
    
    Parameters
    ----------
    pts : array-like
        Boundary points
    mesh_size : float, optional
        Characteristic mesh size. Larger values = fewer, larger cells.
        Default is 0.1.
    
    Returns
    -------
    mesh : pygmsh mesh object
    """
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(pts, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
    return mesh


def create_cell_list_and_adjacency_lists(mesh):
    """
    Create cell list and edge-to-cells mapping for all triangle cells in the mesh.
    Optimized version using edge-based lookup for O(N) complexity instead of O(N^2).
    
    Parameters
    ----------
    mesh : pygmsh mesh object
        Mesh with triangle cell type
        
    Returns
    -------
    cells : list
        List of cell_triangle objects
    edge_to_cells : dict
        Dictionary mapping Edge objects to lists of cell objects containing each edge.
        Interior edges map to 2 cells, boundary edges map to 1 cell.
    """
    tri_conn = mesh.get_cells_type("triangle")
    pts2d = mesh.points[:, :2]  # Extract 2D coordinates


    cells = []
    for tri in tri_conn:
        verts = pts2d[tri][:, :2]   # shape (3, 2): the triangle's three (x,y) vertices
        cells.append(ct.cell_triangle(verts))
    
    # Build edge-to-cells mapping: each edge maps to list of cell objects containing it
    # Note: edges must be created using vertex coordinates (not indices) to match cell.edges
    edge_to_cells = {}
    for i, tri in enumerate(tri_conn):
        # Get the cell's vertices (as coordinate pairs, not indices)
        verts = cells[i].vertices
        # Get all 3 edges of the triangle using vertex coordinates (matching cell.edges format)
        edges = [
            Edge(verts[0], verts[1]),  # AB edge
            Edge(verts[1], verts[2]),  # BC edge
            Edge(verts[2], verts[0])   # CA edge
        ]

        for edge in edges:
            if edge not in edge_to_cells:
                edge_to_cells[edge] = []
            edge_to_cells[edge].append(cells[i])
    
    print(f"    Edge-to-cells mapping complete", flush=True)
    return cells, edge_to_cells

def find_nearest_centroid_cell_vectorized(positions, cells):
    """
    Find the nearest centroid cell for multiple positions using vectorized operations.
    Much faster than calling find_nearest_centroid_cell for each position.
    
    Time complexity: O(N × M) where N = number of positions, M = number of cells.
    This is optimal for typical mesh sizes (hundreds of cells).
    
    For very large meshes (thousands of cells), consider using scipy.spatial.cKDTree
    which has O(N × log(M)) query time, but with O(M × log(M)) build overhead.
    KD-trees typically become faster when M > ~1000 cells.
    
    Parameters
    ----------
    positions : array-like of shape (N, 2)
        Particle positions
    cells : list
        List of cell_triangle objects
        
    Returns
    -------
    nearest_cells : list
        List of nearest cell for each position
    """
    if len(positions) == 0:
        return []
    
    positions = np.asarray(positions)
    # Get all cell centers as array
    cell_centers = np.array([cell.center for cell in cells])  # shape: (n_cells, 2)
    
    # Compute distances: (N, n_cells) array
    # positions[:, None, :] is (N, 1, 2), cell_centers[None, :, :] is (1, n_cells, 2)
    # Result is (N, n_cells) - distance from each position to each cell center
    dists_sq = np.sum((positions[:, None, :] - cell_centers[None, :, :])**2, axis=2)
    
    # Find index of nearest cell for each position
    nearest_indices = np.argmin(dists_sq, axis=1)
    
    # Return list of nearest cells
    return [cells[idx] for idx in nearest_indices]

def find_nearest_centroid_cell_kdtree(positions, cells):
    """
    Find the nearest centroid cell for multiple positions using KD-tree.
    
    Parameters
    ----------
    positions : array-like of shape (N, 2)
        Particle positions
    cells : list
        List of cell_triangle objects
        
    Returns
    -------
    nearest_cells : list
        List of nearest cell for each position
    """
    if len(positions) == 0:
        return []
    
    positions = np.asarray(positions)
    # Get all cell centers as array
    cell_centers = np.array([cell.center for cell in cells])  # shape: (n_cells, 2)
    
    tree = cKDTree(cell_centers)
    
    # Query all positions at once: O(N × log(M))
    # Returns (distances, indices) where indices[i] is the index of the nearest cell for positions[i]
    distances, nearest_indices = tree.query(positions)
    
    return [cells[idx] for idx in nearest_indices]

def find_nearest_centroid_cell(position, cells):
    """
    Find the cell with the nearest centroid to the given position.
    (Kept for backward compatibility, but use vectorized version for multiple positions)
    
    Parameters
    ----------
    position : array-like of shape (2,)
        Particle position (x, y)
    cells : list
        List of cell_triangle objects
        
    Returns
    -------
    nearest_cell : cell_triangle
        Cell with nearest centroid
    """
    position = np.asarray(position)
    min_dist_sq = float('inf')
    nearest_cell = None
    
    for cell in cells:
        dist_sq = np.sum((position - cell.center)**2)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_cell = cell
    
    return nearest_cell

def get_boundary_edges(points):
    """
    Get the boundary edges of the arbitrary shape.
    Returns a list of edge segments [(x1,y1,x2,y2), ...]
    """
    edges = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]  # Wrap around to close the shape
        edges.append((p1[0], p1[1], p2[0], p2[1]))
    return edges

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    Parameters:
    px, py : float
        x and y coordinates of the point
    x1, y1 : float
        x and y coordinates of the start of the line segment
    x2, y2 : float
        x and y coordinates of the end of the line segment

    Returns:
    Calculate the distance from a point to a line segment.
    Returns the distance and the closest point on the line segment.
    """
    # Vector from line start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # Vector from line start to point
    px_vec = px - x1
    py_vec = py - y1
    
    # Project point onto line
    line_length_sq = dx * dx + dy * dy
    if line_length_sq < 1e-12:  # Degenerate line
        return np.sqrt(px_vec * px_vec + py_vec * py_vec), (x1, y1)
    
    t = (px_vec * dx + py_vec * dy) / line_length_sq
    t = np.clip(t, 0, 1)  # Clamp to line segment
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    return dist, (closest_x, closest_y)

def get_edge_normal(x1, y1, x2, y2):
    """
    Get the outward normal vector for an edge.
    Returns normalized normal vector pointing outward from the shape.
    """
    # Edge vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Normal vector (perpendicular to edge)
    # For outward normal, we rotate 90 degrees clockwise: (dx, dy) -> (dy, -dx)
    normal_x = dy
    normal_y = -dx
    
    # Normalize
    length = np.sqrt(normal_x**2 + normal_y**2)
    if length > 1e-12:
        normal_x /= length
        normal_y /= length
    
    return normal_x, normal_y

def point_in_polygon(point_x, point_y, polygon_points):
    """
    Parameters:
    point_x, point_y : float
        x and y coordinates of the point
    polygon_points : list of (x, y) tuples
        List of polygon points

    Returns:
    Ray casting algorithm to determine if point is inside polygon.
    Returns True if point is inside polygon, False otherwise.
    """
    num_vertices = len(polygon_points)
    is_inside = False
    
    previous_vertex_index = num_vertices - 1
    for current_vertex_index in range(num_vertices):
        current_x, current_y = polygon_points[current_vertex_index]
        previous_x, previous_y = polygon_points[previous_vertex_index]
        
        if ((current_y > point_y) != (previous_y > point_y)) and (point_x < (previous_x - current_x) * (point_y - current_y) / (previous_y - current_y) + current_x):
            is_inside = not is_inside
        previous_vertex_index = current_vertex_index
    
    return is_inside

def triangle_to_follow(position, cell, edge_to_cells):
    """
    Find the triangle to follow for a particle using barycentric coordinates.
    
    Barycentric coordinates:
    - a corresponds to vertex A (vertices[0])
    - b corresponds to vertex B (vertices[1])
    - c corresponds to vertex C (vertices[2])
    
    When a barycentric coordinate is negative, the point is outside on the 
    opposite side of the edge opposite to that vertex:
    - a < 0: point is outside edge BC (opposite to vertex A) -> use edges[1]
    - b < 0: point is outside edge CA (opposite to vertex B) -> use edges[2]
    - c < 0: point is outside edge AB (opposite to vertex C) -> use edges[0]
    
    Parameters
    ----------
    position : array-like of shape (2,)
        Particle position (x, y)
    cell : cell_triangle
        Current cell to check
    edge_to_cells : dict
        Dictionary mapping Edge objects to lists of cell objects
        
    Returns
    -------
    cell_triangle
        The adjacent cell across the edge the particle is outside of, or the current cell if inside
    """
    x, y = position[0], position[1]
    x1, y1 = cell.vertices[0]  # vertex A
    x2, y2 = cell.vertices[1]  # vertex B
    x3, y3 = cell.vertices[2]  # vertex C

    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-12:
        # Degenerate triangle, return current cell
        return cell
        
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b
    
    # Map negative barycentric coordinates to the correct edge
    # edges[0] = AB, edges[1] = BC, edges[2] = CA
    if a < 0:
        # Point is outside on the side opposite to vertex A, so it's outside edge BC
        edge = cell.edges[1]  # BC edge
    elif b < 0:
        # Point is outside on the side opposite to vertex B, so it's outside edge CA
        edge = cell.edges[2]  # CA edge
    elif c < 0:
        # Point is outside on the side opposite to vertex C, so it's outside edge AB
        edge = cell.edges[0]  # AB edge
    else:
        # All barycentric coordinates are non-negative, point is inside the triangle
        return cell
    
    # Get cells sharing this edge and return the one that's not the current cell
    candidate_cells = edge_to_cells.get(edge, [])
    for candidate in candidate_cells:
        if candidate is not cell:
            return candidate
    
    # If no adjacent cell found (boundary edge), return current cell
    return cell

def find_containing_cell(position, start_cell, edge_to_cells):
    """
    Iteratively follow triangles using triangle_to_follow until we find a cell where 
    cell.is_inside(position) is satisfied. This is guaranteed to find the correct cell
    for any position within the mesh domain.
    
    Parameters
    ----------
    position : array-like of shape (2,)
        Particle position (x, y)
    start_cell : cell_triangle
        Initial cell to start the search from (typically the nearest centroid cell)
    edge_to_cells : dict
        Dictionary mapping Edge objects to lists of cell objects
        
    Returns
    -------
    cell_triangle
        The cell that contains the position (guaranteed to find it)
    """
    current_cell = start_cell
    
    while True:
        # Check if current cell contains the position
        if current_cell.is_inside(position[0], position[1]):
            return current_cell
        
        # Get the next cell to follow
        next_cell = triangle_to_follow(position, current_cell, edge_to_cells)
        
        # If we're stuck (next_cell is the same as current_cell), we've reached a boundary
        # In this case, the point must be on the boundary, so return the current cell
        if next_cell is current_cell:
            return current_cell
        
        current_cell = next_cell

def reflecting_BC_arbitrary_shape(velocities, positions, boundary_points):
    """
    Apply reflecting boundary condition for arbitrary 2D shape.
    
    Parameters:
    velocities: array of shape (N, 2) - particle velocities
    positions: array of shape (N, 2) - particle positions
    boundary_points: list of (x, y) tuples defining the shape boundary
                    (should be actual B-spline boundary points, not control points)
    
    Returns:
    velocities, positions: updated arrays after reflection
    """
    # Get boundary edges
    edges = get_boundary_edges(boundary_points)
    
    # Check which particles are outside the domain
    outside_mask = np.zeros(len(positions), dtype=bool)
    for i, pos in enumerate(positions):
        # Use point-in-polygon test to determine if particle is outside
        if not point_in_polygon(pos[0], pos[1], boundary_points):
            outside_mask[i] = True
    
    if not np.any(outside_mask):
        return velocities, positions
    
    # Process particles that are outside
    for idx in np.where(outside_mask)[0]:
        pos = positions[idx]
        vel = velocities[idx]
        
        # Find the closest edge and reflect
        min_dist = float('inf')
        closest_edge = None
        closest_point = None
        
        for edge in edges:
            x1, y1, x2, y2 = edge
            dist, closest_pt = point_to_line_distance(pos[0], pos[1], x1, y1, x2, y2)
            
            if dist < min_dist:
                min_dist = dist
                closest_edge = edge
                closest_point = closest_pt
        
        if closest_edge is not None:
            x1, y1, x2, y2 = closest_edge
            
            # Get normal vector to the edge
            normal_x, normal_y = get_edge_normal(x1, y1, x2, y2)
            
            # Project particle back to boundary
            positions[idx] = np.array(closest_point)
            
            # Reflect velocity: v' = v - 2(v·n)n
            vel_dot_normal = vel[0] * normal_x + vel[1] * normal_y
            velocities[idx][0] = vel[0] - 2 * vel_dot_normal * normal_x
            velocities[idx][1] = vel[1] - 2 * vel_dot_normal * normal_y
    
    return velocities, positions


def Arbitrary_Shape_Parameterized(N, fourier_coefficients, num_boundary_points, T_x0, T_y0, dt, n_tot, e, mu, alpha, mesh_size=0.1):
    """
    Run DSMC simulation on arbitrary parameterized shape.
    
    Workflow:
    1. Create mesh from fourier coefficients
    2. Generate positions (area-weighted) and velocities
    3. Create cell list and edge-to-cells mapping
    4. Run algorithm (collide + update positions)
    5. Apply boundary condition
    6. Rebin particles that violate cell invariant
    
    Parameters
    ----------
    N : int
        Number of particles
    fourier_coefficients : array-like
        Fourier coefficients defining the shape
    num_boundary_points : int
        Number of points to sample along boundary
    T_x0, T_y0 : float
        Initial temperatures in x and y directions
    dt : float
        Time step
    n_tot : int
        Total number of time steps
    e : float
        Parameter for collision rate
    mu : float
        Viscosity parameter (unused currently)
    alpha : float
        VHS model parameter (unused currently)
    mesh_size : float, optional
        Characteristic mesh size. Larger values = fewer, larger cells.
        Default is 0.1. Recommended: 0.05 (fine) to 0.5 (coarse).
    """
    

    boundary_points = sample_star_shape(fourier_coefficients, num_boundary_points)
    mesh = hf.create_arbitrary_shape_mesh_2d(N, boundary_points, mesh_size=mesh_size)
    
    positions = hf.assign_positions_arbitrary_2d(N, mesh)
    velocities = hf.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)
    
    cell_list, edge_to_cells = create_cell_list_and_adjacency_lists(mesh)

    for position, velocity in zip(positions, velocities):
        for cell in cell_list:
            if cell.is_inside(position[0], position[1]):
                cell.add_particle(position, velocity)
                break
    
    temperature_history = np.zeros(n_tot)
    
    total_start_time = time.time()
    
    for n in range(n_tot):
        step_start_time = time.time()
        
        # Step 4: Run collision algorithm
        collision_start = time.time()
        for cell in cell_list: 
            num_collisions = cell.num_collisions(dt, e)

            indices_particles = np.arange(len(cell.particle_positions))

            indices_particles_to_collide = np.random.permutation(indices_particles)[:num_collisions]

            indices_i = indices_particles_to_collide[:num_collisions // 2]
            indices_j = indices_particles_to_collide[num_collisions // 2:]

            indices_pairs = np.column_stack((indices_i, indices_j))
            
            if len(indices_i) > 0 and len(indices_j) > 0:

                v_rel = cell.particle_velocities[indices_i] - cell.particle_velocities[indices_j]
                v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)

                sigma_ij = hf.ArraySigma_VHS(v_rel_mag).reshape(-1)

                Upper_bound_cross_sections = cell.upper_bound_cross_section()
                
                u_rand = np.random.rand(len(indices_i)) * Upper_bound_cross_sections

                accept_condition = (u_rand < sigma_ij)

                indices_i = indices_i[accept_condition]
                indices_j = indices_j[accept_condition]

                if len(indices_i) > 0:
                    cell.collide_and_update_particles(dt, indices_i, indices_j)
        collision_time = time.time() - collision_start

        bc_start = time.time()
        total_kinetic_energy = 0.0
        total_particles = 0
        
        for cell in cell_list:
            if len(cell.particle_positions) > 0:
                cell.particle_velocities, cell.particle_positions = reflecting_BC_arbitrary_shape(
                    cell.particle_velocities, cell.particle_positions, boundary_points
                )
                # Track temperature (sum of squared velocities)
                total_kinetic_energy += np.sum(cell.particle_velocities**2)
                total_particles += len(cell.particle_velocities)
        
        # Track temperature
        temperature_history[n] = total_kinetic_energy / total_particles if total_particles > 0 else 0.0
        bc_time = time.time() - bc_start

        # Step 6: Rebin particles that violate cell invariant
        rebin_start = time.time()
        particles_to_move = []  # List of (old_cell, particle_idx, position, velocity)
        
        # Loop through cells and detect particles that don't respect the invariant
        for cell in cell_list:
            indices_to_remove = []
            for i, position in enumerate(cell.particle_positions):
                if not cell.is_inside(position[0], position[1]):
                    # Particle violates invariant - mark for removal and rebinning
                    particles_to_move.append((cell, i, position, cell.particle_velocities[i]))
                    indices_to_remove.append(i)
            
            # Remove violating particles from this cell (in reverse order to preserve indices)
            for i in reversed(indices_to_remove):
                cell.remove_particle(i)
        
        # Now reassign the violating particles to their correct cells
        if len(particles_to_move) > 0:
            # Extract all positions for nearest centroid lookup
            positions_to_rebin = np.array([position for _, _, position, _ in particles_to_move])
            
            # Find nearest centroid cells for all particles at once using KD-tree
            nearest_cells = find_nearest_centroid_cell_kdtree(positions_to_rebin, cell_list)
            
            # Now iterate through and find containing cells using triangle following
            for (old_cell, _, position, velocity), nearest_cell in zip(particles_to_move, nearest_cells):
                # Use triangle_to_follow to iteratively find the containing cell
                containing_cell = find_containing_cell(position, nearest_cell, edge_to_cells)
                
                # Add particle to the containing cell
                containing_cell.add_particle(position, velocity)
        
        rebin_time = time.time() - rebin_start
        
        step_time = time.time() - step_start_time
        
        # Print progress every 10 steps or on last step
        if (n + 1) % 10 == 0 or (n + 1) == n_tot:
            print(f"{n+1}/{n_tot} ", end="", flush=True)
            if (n + 1) == n_tot or (n + 1) % 50 == 0:
                num_moved = len(particles_to_move)
                print(f"\n  Step {n+1} timing: collisions={collision_time:.3f}s, "
                      f"BC={bc_time:.3f}s, rebin={rebin_time:.3f}s, total={step_time:.3f}s")
                print(f"  Rebinned {num_moved} particles", flush=True)
    
    # Reconstruct final global arrays from cells
    all_positions = []
    all_velocities = []
    for cell in cell_list:
        if len(cell.particle_positions) > 0:
            all_positions.append(cell.particle_positions)
            all_velocities.append(cell.particle_velocities)
    
    if all_positions:
        final_positions = np.vstack(all_positions)
        final_velocities = np.vstack(all_velocities)
    else:
        final_positions = np.empty((0, 2))
        final_velocities = np.empty((0, 2))
    
    total_time = time.time() - total_start_time
    print(f"\nSimulation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per step: {total_time/n_tot:.3f} seconds")
    
    return final_positions, final_velocities, temperature_history, cell_list, boundary_points






