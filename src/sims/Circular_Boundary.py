import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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







def Circular_Boundary(N, R, positions, radius, T_x0, T_y0, dt, n_tot, e, mu, alpha, buckets_x, buckets_y):

    positions = hf.assign_positions_circular(N, R)
    velocities = hf.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)


    #spatial discretization
    cell_width = radius / buckets_x
    cell_height = radius / buckets_y

    bucket_dict = {(i, j): set() for i in range(buckets_x) for j in range(buckets_y)}



    with pygmsh.geo.Geometry() as geom:
        circle = geom.add_circle([0.0, 0.0, 0.0], radius=1.0, mesh_size=0.1)
        mesh = geom.generate_mesh()

    tri_conn = mesh.get_cells_type("triangle")
    # 2D coords (drop z); points shape is (n_pts, 3)
    pts2d = mesh.points[:, :2]

    cells = []
    for tri in tri_conn:
        verts = pts2d[tri]          # shape (3, 2): the triangle's three (x,y) vertices
        cells.append(cell_triangle(verts))

    for position, velocity in zip(positions, velocities):
        for cell in cells:
            if cell.is_inside(position[0], position[1]):
                cell.add_particle(position, velocity)
                break

    for bucket in bucket_dict:
        for cell in cells:
            if (cell.bounding_box[0] < bucket[0] * cell_width + cell_width and 
                cell.bounding_box[1] > bucket[0] * cell_width and 
                cell.bounding_box[2] < bucket[1] * cell_height + cell_height and 
                cell.bounding_box[3] > bucket[1] * cell_height):
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

        # Apply circular reflecting BC on global arrays
        velocities, positions = pb.reflecting_BC_circular(velocities, positions, R)

        # Track temperature
        temperature_history[n] = np.sum(velocities**2) / velocities.shape[0]

        # Re-binning (if desired; left as-is)
        for cell in cells:
            for particle in cell.particle_positions:
                if not(cell.is_inside(particle[0], particle[1])):
                    cell.remove_particle(particle)
                    x_coord = particle[0]
                    y_coord = particle[1]
                    ix = np.floor(x_coord / cell_width).astype(int)
                    iy = np.floor(y_coord / cell_height).astype(int)
                    corresponding_bucket = (ix, iy)

                    for triangle in bucket_dict[corresponding_bucket]:
                        if triangle.is_inside(particle[0], particle[1]):
                            triangle.add_particle(particle, cell.particle_velocities[particle])
                            break
    return positions, velocities, temperature_history
        