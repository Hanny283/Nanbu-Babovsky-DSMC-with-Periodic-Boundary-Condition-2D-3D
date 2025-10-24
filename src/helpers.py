import numpy as np 
import pygmsh 


def create_arbitrary_shape_mesh_2d(N, points):
    """
    Create an arbitrary shape in 2D using B-spline curves for smooth boundaries.
    """

    with pygmsh.occ.Geometry() as geom:

        # 1) Add gmsh points
        gmpts = [geom.add_point([x, y]) for (x, y) in points]

        # 2) Create B-spline curve connecting all points
        # For a closed B-spline, we need to repeat the first point at the end
        closed_points = gmpts + [gmpts[0]]  # Close the curve
        
        # Create B-spline curve with the closed point list
        bspline_curve = geom.add_bspline(closed_points)

        # 3) Close the boundary: use a curve loop, then a plane surface
        loop = geom.add_curve_loop([bspline_curve])
        surf = geom.add_plane_surface(loop)

        # tag the boundary as a Physical Group for later lookup
        boundary_pg = geom.add_physical([bspline_curve], label="boundary")
        domain_pg   = geom.add_physical([surf], label="domain")

        # 4) Generate mesh (triangles by default)
        mesh = geom.generate_mesh()

    return mesh

def assign_positions_arbitrary_2d(N, mesh):

    # 1) Extract triangle data from mesh
    triangle_indices = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_indices = cell.data
            break
    if triangle_indices is None:
        raise ValueError("No triangles found.")

    mesh_points = mesh.points  # x,y coordinates of all mesh points

    # 2) Calculate triangle areas using cross product
    vertex_a = mesh_points[triangle_indices[:, 0]]  # First vertex of each triangle
    vertex_b = mesh_points[triangle_indices[:, 1]]  # Second vertex of each triangle
    vertex_c = mesh_points[triangle_indices[:, 2]]  # Third vertex of each triangle
    
    # Calculate areas using 2D cross product formula
    triangle_areas = 0.5 * np.abs((vertex_b[:, 0] - vertex_a[:, 0]) * (vertex_c[:, 1] - vertex_a[:, 1]) - 
                                  (vertex_c[:, 0] - vertex_a[:, 0]) * (vertex_b[:, 1] - vertex_a[:, 1]))

    # 3) Choose triangles by area-weighted sampling
    selected_triangle_indices = np.random.choice(len(triangle_indices), size=N, p=triangle_areas / triangle_areas.sum())

    # 4) Generate random barycentric coordinates (Marsaglia trick)
    sqrt_random_1 = np.sqrt(np.random.rand(N, 1))
    random_2 = np.random.rand(N, 1)
    barycentric_u = 1 - sqrt_random_1
    barycentric_v = sqrt_random_1 * (1 - random_2)
    barycentric_w = sqrt_random_1 * random_2

    # 5) Get vertices of selected triangles and compute final positions
    selected_vertex_a = vertex_a[selected_triangle_indices]
    selected_vertex_b = vertex_b[selected_triangle_indices]
    selected_vertex_c = vertex_c[selected_triangle_indices]
    particle_positions = barycentric_u * selected_vertex_a + barycentric_v * selected_vertex_b + barycentric_w * selected_vertex_c
    
    # Ensure we only return 2D positions
    particle_positions = particle_positions[:, :2]
    
    return particle_positions


def sample_velocities_from_maxwellian_2d(T_x0, T_y0, N):
    """
    Sample from a 2D Maxwellian distribution.

    Parameters
    ----------
    T_x0 : float
    The x-component of the temperature.

    T_y0 : float
    The y-component of the temperature.

    N : int 
    The number of velocities to draw
    Returns a numpy array of shape (N, 2) containing the sampled velocities.
    """

    v_x_samples = np.random.normal(0, np.sqrt(T_x0), N)
    v_y_samples = np.random.normal(0, np.sqrt(T_y0), N)

    return np.column_stack((v_x_samples, v_y_samples))

def sample_velocities_from_maxwellian_3d(T_x0, T_y0, T_z0, N):
    """
    Sample from a 3D Maxwellian distribution.

    Parameters
    ----------
    T_x0 : float
    The x-component of the temperature.

    T_y0 : float
    The y-component of the temperature.

    T_z0 : float
    The z-component of the temperature.

    N : int 
    The number of velocities to draw
    Returns a numpy array of shape (N, 3) containing the sampled velocities.
    """

    v_x_samples = np.random.normal(0, np.sqrt(T_x0), N)
    v_y_samples = np.random.normal(0, np.sqrt(T_y0), N)
    v_z_samples = np.random.normal(0, np.sqrt(T_z0), N)

    return np.column_stack((v_x_samples, v_y_samples, v_z_samples))
    

def assign_positions_2d(N, Lx, Ly):
    """
    Assign positions to particles on a 2-dimensional spatial domain.

    Parameters
    ----------
    N : int
        Number of particles
    Lx : float 
        The length of the x spatial domain
    Ly : float 
        The length of the y spatial domain

    Returns
    -------
    positions : numpy array of shape (N, 2)
        The positions of the particles.
    """
    positions_x = np.random.uniform(0, Lx, N)
    positions_y = np.random.uniform(0, Ly, N)
    positions = np.column_stack((positions_x, positions_y))
    return positions

def assign_positions_circular(N, R):
    """
    Assign positions to particles on a circular domain.
    """
    positions_x = R * np.cos(2 * np.pi * np.random.rand(N))
    positions_y = R * np.sin(2 * np.pi * np.random.rand(N))
    positions = np.column_stack((positions_x, positions_y))
    return positions

def assign_positions_3d(N, Lx, Ly, Lz):
    """
    Assign positions to particles on a 3-dimensional spatial domain.

    Parameters
    ----------
    N : int
        Number of particles
    Lx : float 
        The length of the x spatial domain
    Ly : float 
        The length of the y spatial domain
    Lz : float 
        The length of the z spatial domain

    Returns
    -------
    positions : numpy array of shape (N, 3)
        The positions of the particles.
    """
    positions_x = np.random.uniform(0, Lx, N)
    positions_y = np.random.uniform(0, Ly, N)
    positions_z = np.random.uniform(0, Lz, N)
    positions = np.column_stack((positions_x, positions_y, positions_z))
    return positions

def assign_positions_circular(N, R):
    """
    Assign positions to particles on a circular domain.
    """
    positions_x = R * np.cos(2 * np.pi * np.random.rand(N))
    positions_y = R * np.sin(2 * np.pi * np.random.rand(N))
    positions = np.column_stack((positions_x, positions_y))
    return positions


def assign_positions_spherical(N, R):
    """
    Assign positions to particles uniformly distributed inside a sphere.
    """
    # Generate random points in unit sphere using rejection sampling
    positions = np.zeros((N, 3))
    count = 0
    while count < N:
        # Generate random point in cube [-1, 1]^3
        x = 2 * np.random.rand() - 1
        y = 2 * np.random.rand() - 1
        z = 2 * np.random.rand() - 1
        
        # Check if point is inside unit sphere
        if x*x + y*y + z*z <= 1:
            positions[count] = [x, y, z]
            count += 1
    
    # Scale to radius R
    positions *= R
    return positions


def collide_particles_2d(velocities, indices_i, indices_j):
    """
    Collides M particles with velocities given by the arraysv_i and v_j.

    Parameters
    ----------
    velocities : numpy array of shape (N, 2)
    indices_i : numpy array of shape (M,)
    indices_j : numpy array of shape (M,)
    """
    v_i = velocities[indices_i]
    v_j = velocities[indices_j]

    v_rel = v_i - v_j                      
    v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)  

    theta = np.random.uniform(0.0, 2.0*np.pi, size=len(indices_i))
    omega = np.column_stack((np.cos(theta), np.sin(theta))) 

    v_cm = 0.5 * (v_i + v_j)

    v_i_prime = v_cm + 0.5 * v_rel_mag * omega
    v_j_prime = v_cm - 0.5 * v_rel_mag * omega

    velocities[indices_i] = v_i_prime
    velocities[indices_j] = v_j_prime

    return velocities

def collide_particles_3d(velocities, indices_i, indices_j):
    """
    Collides M particles with velocities given by the arrays v_i and v_j.

    Parameters
    ----------
    velocities : numpy array of shape (N, 3)
    indices_i : numpy array of shape (M,)
    indices_j : numpy array of shape (M,)
    """
    v_i = velocities[indices_i]
    v_j = velocities[indices_j]

    v_rel = v_i - v_j                      
    v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)  
    
    theta = np.arccos(2*np.random.rand( len(indices_i) ) - 1)
    phi = np.random.uniform(0.0, 2.0*np.pi, size=len(indices_i))

    omega = np.column_stack((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)))

    v_cm = 0.5 * (v_i + v_j)

    v_i_prime = v_cm + 0.5 * v_rel_mag * omega
    v_j_prime = v_cm - 0.5 * v_rel_mag * omega

    velocities[indices_i] = v_i_prime
    velocities[indices_j] = v_j_prime

    return velocities

def compute_upper_bound_cross_section(velocities):
    if velocities.size == 0:
        return 0.0

    v_mean = velocities.mean(axis= 0)
    delta_v = np.linalg.norm(velocities - v_mean, axis=1).max()
    return 2.0 * delta_v

def Iround(x):
    """
    Vectorized probabilistic rounding of an array of floats.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array of rounded integers.
    """
    lower = np.floor(x).astype(int)
    prob = x - lower
    random_numbers = np.random.rand(*x.shape)
    return lower + np.where(random_numbers < prob, 1, 0)

def sample_particle_indices_to_collide_grid(Nc, cell_velocities, rng=None):
    """
    Generalized version for 2-D (and N-D) grids.

    Parameters
    ----------
    Nc : array-like of int, shape = grid shape (e.g., (nx, ny))
        Nc[idx] is the number of collision pairs to draw in that cell.
    cell_velocities : array-like, shape = grid shape + (object,)
        Each cell entry cell_velocities[idx] must be a 1-D array-like
        of per-particle velocities for that cell (length = #particles).
        Works well if this is a numpy array with dtype=object, or a nested list.
    rng : None | np.random.Generator | int
        Random generator or seed.

    Returns
    -------
    sampled_indices : np.ndarray(dtype=object, shape=Nc.shape)
        sampled_indices[idx] is a 1-D int array of length 2*Nc[idx]
        (empty array if Nc[idx] == 0).
    """
    Nc = np.asarray(Nc, dtype=int)
    grid_shape = Nc.shape

    # Basic shape check: the first Nc.ndim axes must match
    if tuple(np.shape(cell_velocities)[:Nc.ndim]) != grid_shape:
        raise ValueError(
            f"Shape mismatch: Nc.shape={grid_shape} vs cell_velocities.shape[:{Nc.ndim}]="
            f"{np.shape(cell_velocities)[:Nc.ndim]}"
        )

    # Robust RNG handling
    if isinstance(rng, (int, np.integer)) or rng is None:
        rng = np.random.default_rng(rng)

    sampled = np.empty(grid_shape, dtype=object)

    for idx in np.ndindex(*grid_shape):
        k = int(Nc[idx])
        if k < 0:
            raise ValueError(f"k must be nonnegative at cell {idx}, got {k}")

        # Get the per-cell particle list; ensure it's an array
        g = np.asarray(cell_velocities[idx])
        n_particles = int(g.shape[0])

        if 2 * k > n_particles:
            raise ValueError(
                f"Cannot sample {2*k} from cell {idx} which has {n_particles} particles"
            )

        if k == 0:
            sampled[idx] = np.empty(0, dtype=int)
        else:
            sampled[idx] = rng.choice(n_particles, size=2 * k, replace=False).astype(int)

    return sampled

def pair_particle_indices_2d(sampled_indices):
    """
    input: list of [i,j] pairs of particle indices in a cell
    """
    shuffled = sampled_indices[:]           # copy so original isn't changed
    np.random.shuffle(shuffled)   # shuffle order

    # group into consecutive pairs
    return np.array([[shuffled[i], shuffled[i+1]] for i in range(0, len(shuffled), 2)])

def pair_particle_indices_3d(sampled_indices):
    """
    input: list of particle indices in a cell, returns pairs for collision
    """
    shuffled = sampled_indices[:]           # copy so original isn't changed
    np.random.shuffle(shuffled)   # shuffle order

    # group into consecutive pairs
    return np.array([[shuffled[i], shuffled[i+1]] for i in range(0, len(shuffled), 2)])

def update_positions_2d(positions, velocities, dt):
    """
    Update particle positions using velocities and time step.
    
    Parameters
    ----------
    positions : numpy array of shape (N, 2)
        Current particle positions
    velocities : numpy array of shape (N, 2)
        Current particle velocities
    dt : float
        Time step
        
    Returns
    -------
    positions : numpy array of shape (N, 2)
        Updated particle positions
    """
    positions = positions + velocities * dt
    return positions

def update_positions_3d(positions, velocities, dt):
    """
    Update particle positions using velocities and time step.
    """
    positions = positions + velocities * dt
    return positions


def ArraySigma_VHS(v):
    Constant = 1.0
    alpha = 1
    return Constant * np.power(v, alpha)