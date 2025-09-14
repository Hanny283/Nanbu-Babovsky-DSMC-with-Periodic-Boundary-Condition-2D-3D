import numpy as np 


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
    

def assign_positions_2d(velocities, Lx, Ly):
    """
    Assign positions to particles on a 1-dimensional spatial domain.

    Parameters
    ----------
    velocities : numpy array of particles velocities
    L: float 
    The length of the spatial domain

    Returns
    -------
    positions : numpy array of shape (N, 2)
        The positions of the particles.
    """
    num_particles = len(velocities)
    positions = np.random.uniform(0, Lx, num_particles)
    positions = np.random.uniform(0, Ly, num_particles)
    positions = np.column_stack((positions_x, positions_y))
    return positions

def assign_positions_3d(velocities, Lx, Ly, Lz):
    """
    Assign positions to particles on a 1-dimensional spatial domain.

    Parameters
    ----------
    velocities : numpy array of particles velocities
    L: float 
    The length of the spatial domain

    Returns
    -------
    positions : numpy array of shape (N, 2)
        The positions of the particles.
    """
    num_particles = len(velocities)
    positions = np.random.uniform(0, Lx, num_particles)
    positions = np.random.uniform(0, Ly, num_particles)
    positions = np.random.uniform(0, Lz, num_particles)
    positions = np.column_stack((positions_x, positions_y, positions_z))
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
    Collides M particles with velocities given by the arraysv_i and v_j.

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

