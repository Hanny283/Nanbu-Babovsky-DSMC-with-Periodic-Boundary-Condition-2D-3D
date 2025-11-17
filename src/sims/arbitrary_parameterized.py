import pygmsh 
import helpers as hf
import numpy as np
import periodic_bc as pb
import time


class cell_triangle:
    def __init__(self, vertices):
        """
        vertices: a list or array of 3 coordinate pairs [(x1, y1), (x2, y2), (x3, y3)]
        """
        self.vertices = np.array(vertices)
        self.center = np.mean(self.vertices, axis=0)  # centroid
        self.particle_positions = np.empty((0, 2))
        self.particle_velocities = np.empty((0, 2))
        self.rho_cell = len(self.particle_positions) / self.area()

        # make bounding box
        minx = min(self.vertices[:,0])
        maxx = max(self.vertices[:,0])
        miny = min(self.vertices[:,1])
        maxy = max(self.vertices[:,1])
        self.bounding_box = (minx, maxx, miny, maxy)

    
        
    def area(self):
        # Use a robust 2D shoelace/cross-product formula for triangle area
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]
        return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def is_inside(self, x, y):
        """
        Return True if point (x, y) lies inside or on the edges of the triangle.
        Uses barycentric coordinates.
        """
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]

        denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
        b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
        c = 1 - a - b

        return (a >= 0) and (b >= 0) and (c >= 0)

    def add_particle(self, position, velocity):
        position = np.asarray(position).reshape(1, 2)
        velocity = np.asarray(velocity).reshape(1, 2)
        self.particle_positions = np.vstack((self.particle_positions, position))
        self.particle_velocities = np.vstack((self.particle_velocities, velocity))

    def remove_particle(self, index):
        self.particle_positions = np.delete(self.particle_positions, index, axis=0)
        self.particle_velocities = np.delete(self.particle_velocities, index, axis=0)

    def get_particle_positions(self):
        return self.particle_positions

    def get_particle_velocities(self):
        return self.particle_velocities
    
    def upper_bound_cross_section(self):
        if len(self.particle_velocities) == 0:
            return 0.0
        
        v_mean = self.particle_velocities.mean(axis= 0)
        delta_v = np.linalg.norm(self.particle_velocities - v_mean, axis=1).max()
        return 2.0 * delta_v

    def num_collisions(self, dt, e):
        num_particles = len(self.particle_positions)
        if num_particles < 2:
            return 0
        ub_sigma = self.upper_bound_cross_section()
        expected_pairs = (num_particles * self.rho_cell * ub_sigma * dt) / (2 * e) if e != 0 else 0.0
        expected_pairs_int = hf.Iround(expected_pairs)
        max_pairs = num_particles // 2
        return int(min(expected_pairs_int, max_pairs))

    def collide_and_update_particles(self, dt, indices_i, indices_j):
        v_i = self.particle_velocities[indices_i]
        v_j = self.particle_velocities[indices_j]

        v_rel = v_i - v_j                      
        v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)  

        theta = np.random.uniform(0.0, 2.0*np.pi, size=len(indices_i))
        omega = np.column_stack((np.cos(theta), np.sin(theta))) 

        v_cm = 0.5 * (v_i + v_j)

        v_i_prime = v_cm + 0.5 * v_rel_mag * omega
        v_j_prime = v_cm - 0.5 * v_rel_mag * omega

        self.particle_velocities[indices_i] = v_i_prime
        self.particle_velocities[indices_j] = v_j_prime

        self.particle_positions = self.particle_positions + self.particle_velocities * dt

        return self.particle_positions, self.particle_velocities

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


def create_mesh_from_star_shape(pts):
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(pts, mesh_size=0.1)
        mesh = geom.generate_mesh()
    return mesh


def create_cell_list_and_adjacency_list(mesh):
    """
    Create an adjacency dictionary for all triangle cells in the mesh.
    
    Parameters
    ----------
    mesh : pygmsh mesh object
        Mesh with triangle cell type
        
    Returns
    -------
    adjacency_dict : dict
        Dictionary mapping cell_triangle objects to lists of adjacent cell_triangle objects.
        Each cell has 2-3 adjacent cells (2 for boundary cells, 3 for interior cells).
    """
    # Extract triangle connectivity and points from mesh
    tri_conn = mesh.get_cells_type("triangle")
    pts2d = mesh.points[:, :2]  # Extract 2D coordinates
    
    # Create cell_triangle objects for each triangle
    cells = []
    for tri in tri_conn:
        verts = pts2d[tri][:, :2]   # shape (3, 2): the triangle's three (x,y) vertices
        cells.append(cell_triangle(verts))
    
    # Create adjacency dictionary
    adjacency_dict = {}
    
    # For each cell, find adjacent cells (cells that share exactly 2 vertices)
    for i, cell in enumerate(cells):
        adjacent_cells = []
        # Get vertex indices for current triangle
        tri_i_vertices = set(tri_conn[i])
        
        # Check all other cells
        for j, other_cell in enumerate(cells):
            if i != j:  # Don't compare with itself
                # Get vertex indices for other triangle
                tri_j_vertices = set(tri_conn[j])
                
                # Check if they share exactly 2 vertices
                shared_vertices = tri_i_vertices & tri_j_vertices
                if len(shared_vertices) == 2:
                    adjacent_cells.append(other_cell)
        
        # Map cell to its list of adjacent cells
        adjacency_dict[cell] = adjacent_cells
    
    return cells, adjacency_dict

def find_nearest_centroid_cell_vectorized(positions, cells):
    """
    Find the nearest centroid cell for multiple positions using vectorized operations.
    Much faster than calling find_nearest_centroid_cell for each position.
    
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
    Ray casting algorithm to determine if point is inside polygon.
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


def Arbitrary_Shape_Parameterized(N, fourier_coefficients, num_boundary_points, positions, T_x0, T_y0, dt, n_tot, e, mu, alpha):

    boundary_points = sample_star_shape(fourier_coefficients, num_boundary_points)
    mesh = create_mesh_from_star_shape(boundary_points)
    
    cells, adjacency_dict = create_cell_list_and_adjacency_list(mesh)

    positions = hf.assign_positions_arbitrary_2d(N, mesh)
    velocities = hf.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)

    for position, velocity in zip(positions, velocities):
        for cell in cells:
            if cell.is_inside(position[0], position[1]):
                cell.add_particle(position, velocity)
                break

    temperature_history = np.zeros(n_tot)
    
    print(f"Starting simulation with {len(cells)} cells and {N} particles...")
    print(f"Progress: ", end="", flush=True)
    
    total_start_time = time.time()
    
    for n in range(n_tot):
        step_start_time = time.time()
        
        # Collision step
        collision_start = time.time()
        for cell in cells: 
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

        # Reconstruct global arrays from cells after collisions
        reconstruct_start = time.time()
        all_positions = []
        all_velocities = []
        for cell in cells:
            if len(cell.particle_positions) > 0:
                all_positions.append(cell.particle_positions)
                all_velocities.append(cell.particle_velocities)
        
        if all_positions:
            positions = np.vstack(all_positions)
            velocities = np.vstack(all_velocities)
        else:
            positions = np.empty((0, 2))
            velocities = np.empty((0, 2))
        reconstruct_time = time.time() - reconstruct_start

        # Apply reflecting boundary condition on global arrays
        bc_start = time.time()
        velocities, positions = reflecting_BC_arbitrary_shape(velocities, positions, boundary_points)
        bc_time = time.time() - bc_start

        # Track temperature
        temperature_history[n] = np.sum(velocities**2) / velocities.shape[0] if len(velocities) > 0 else 0.0

        # Re-binning particles into cells
        rebin_start = time.time()
        # Clear all cells first (necessary because particles have moved)
        for cell in cells:
            cell.particle_positions = np.empty((0, 2))
            cell.particle_velocities = np.empty((0, 2))
        
        # Re-assign particles to cells using adjacency_dict for efficiency
        # Use vectorized nearest cell finding
        fallback_count = 0
        if len(positions) > 0:
            nearest_cells = find_nearest_centroid_cell_vectorized(positions, cells)
            
            # Now assign particles using the nearest cells
            for i, (position, velocity) in enumerate(zip(positions, velocities)):
                nearest_cell = nearest_cells[i]
                
                # Get candidate cells: nearest cell + its adjacent cells
                candidate_cells = [nearest_cell] + adjacency_dict.get(nearest_cell, [])
                
                # Check only candidate cells to find which one contains the particle
                assigned = False
                for cell in candidate_cells:
                    if cell.is_inside(position[0], position[1]):
                        cell.add_particle(position, velocity)
                        assigned = True
                        break
                
                # If not found in candidate cells, fall back to checking all cells
                if not assigned:
                    fallback_count += 1
                    for cell in cells:
                        if cell.is_inside(position[0], position[1]):
                            cell.add_particle(position, velocity)
                            break
        rebin_time = time.time() - rebin_start
        
        step_time = time.time() - step_start_time
        
        # Print progress every 10 steps or on last step
        if (n + 1) % 10 == 0 or (n + 1) == n_tot:
            print(f"{n+1}/{n_tot} ", end="", flush=True)
            if (n + 1) == n_tot or (n + 1) % 50 == 0:
                print(f"\n  Step {n+1} timing: collisions={collision_time:.3f}s, "
                      f"reconstruct={reconstruct_time:.3f}s, BC={bc_time:.3f}s, "
                      f"rebin={rebin_time:.3f}s, total={step_time:.3f}s")
                if fallback_count > 0:
                    print(f"  Warning: {fallback_count} particles needed fallback search")
    
    total_time = time.time() - total_start_time
    print(f"\nSimulation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per step: {total_time/n_tot:.3f} seconds")
    
    return positions, velocities, temperature_history






