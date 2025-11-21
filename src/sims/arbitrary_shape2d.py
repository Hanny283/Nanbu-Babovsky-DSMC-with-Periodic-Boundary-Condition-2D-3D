import numpy as np 
import helpers as hf
import periodic_bc as pb
import pygmsh
import cell_class as ct

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

def Arbitrary_Shape_2D(N, points, positions, radius, T_x0, T_y0, dt, n_tot, e, mu, alpha, buckets_x, buckets_y):

    """
    Arbitrary Shape 2D Simulation

    Parameters:
    N: int
    The number of particles
    points: list of tuples
    The points of the arbitrary shape in 2D
    positions: list of tuples
    The positions of the particles
    radius: float
    The radius of the arbitrary shape
    T_x0: float
    The x-component of the temperature
    T_y0: float
    The y-component of the temperature
    dt: float
    The time step
    n_tot: int
    The total number of time steps
    e: float
    The energy parameter
    mu: float
    The mass parameter
    alpha: float
    The alpha parameter for VHS cross section
    buckets_x: int
    The number of buckets in the x direction
    buckets_y: int
    The number of buckets in the y direction
    """

    mesh = hf.create_arbitrary_shape_mesh_2d(N, points)
    positions = hf.assign_positions_arbitrary_2d(N, mesh)
    velocities = hf.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)
    tri_conn = mesh.get_cells_type("triangle")
    pts2d = mesh.points[:, :]
    
    # Extract actual boundary points from the mesh (B-spline boundary, not control points)
    actual_boundary_points = hf.extract_boundary_points_from_mesh(mesh)

    length = max(pts2d[:, 0]) - min(pts2d[:, 0])

    height = max(pts2d[:, 1]) - min(pts2d[:, 1])

    #spatial discretization
    cell_width = length / buckets_x
    cell_height = height / buckets_y

    bucket_dict = {(i, j): set() for i in range(buckets_x) for j in range(buckets_y)}

    cells = []
    for tri in tri_conn:
        verts = pts2d[tri][:, :2]   # shape (3, 2): the triangle's three (x,y) vertices, ensure only 2D
        cells.append(ct.cell_triangle(verts))

    for position, velocity in zip(positions, velocities):
        for cell in cells:
            if cell.is_inside(position[0], position[1]):
                cell.add_particle(position, velocity)
                break

    for bucket in bucket_dict:
        for cell in cells:
            if cell.bounding_box[0] <= bucket[0] * cell_width and cell.bounding_box[1] >= bucket[0] * cell_width and cell.bounding_box[2] <= bucket[1] * cell_height and cell.bounding_box[3] >= bucket[1] * cell_height:
                bucket_dict[bucket].add(cell)
                break

    temperature_history = np.zeros(n_tot)

    for n in range(n_tot):
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

        # Reconstruct global arrays from cells after collisions
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

        # Apply reflecting boundary condition on global arrays using actual B-spline boundary
        velocities, positions = reflecting_BC_arbitrary_shape(velocities, positions, actual_boundary_points)

        # Track temperature
        temperature_history[n] = np.sum(velocities**2) / velocities.shape[0] if len(velocities) > 0 else 0.0

        # Re-binning particles into cells
        # Clear all cells first
        for cell in cells:
            cell.particle_positions = np.empty((0, 2))
            cell.particle_velocities = np.empty((0, 2))
        
        # Re-assign particles to cells
        for position, velocity in zip(positions, velocities):
            for cell in cells:
                if cell.is_inside(position[0], position[1]):
                    cell.add_particle(position, velocity)
                    break
    
    return positions, velocities, temperature_history
