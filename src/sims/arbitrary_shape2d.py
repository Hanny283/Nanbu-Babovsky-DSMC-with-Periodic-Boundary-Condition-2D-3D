import numpy as np 
import helpers as hf
import periodic_bc as pb
import pygmsh

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

def point_in_polygon(px, py, polygon_points):
    """
    Ray casting algorithm to determine if point is inside polygon.
    """
    n = len(polygon_points)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]
        
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside

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

    length = max(pts2d[:, 0]) - min(pts2d[:, 0])

    height = max(pts2d[:, 1]) - min(pts2d[:, 1])

    #spatial discretization
    cell_width = length / buckets_x
    cell_height = height / buckets_y

    bucket_dict = {(i, j): set() for i in range(buckets_x) for j in range(buckets_y)}

    cells = []
    for tri in tri_conn:
        verts = pts2d[tri][:, :2]   # shape (3, 2): the triangle's three (x,y) vertices, ensure only 2D
        cells.append(cell_triangle(verts))

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

        # Apply reflecting boundary condition on global arrays
        velocities, positions = reflecting_BC_arbitrary_shape(velocities, positions, points)

        # Track temperature
        temperature_history[n] = np.sum(velocities**2) / velocities.shape[0]

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
