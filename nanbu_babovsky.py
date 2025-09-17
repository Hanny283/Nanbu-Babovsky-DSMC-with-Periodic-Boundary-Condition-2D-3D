import numpy as np
import helpers as hf 
import periodic_bc as pb


def Nanbu_Babovsky_2D_Periodic(N, dt, n_tot, e , mu, alpha, Lx, Ly, ncx, ncy, S, dx, T_x0=1.0, T_y0=1.0):
    positions = hf.assign_positions_2d(N, Lx, Ly)
    velocities = hf.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)

    #spatial discretization
    cell_width = Lx / ncx
    cell_height = Ly / ncy

    ix = np.floor(positions[:,0] / cell_width).astype(int)
    iy = np.floor(positions[:,1] / cell_height).astype(int)

    # assign particles to cells
    particle_cell_indices = np.column_stack((ix, iy))

    particle_cell_indices[particle_cell_indices[:,0] == ncx, 0] = ncx - 1
    particle_cell_indices[particle_cell_indices[:,1] == ncy, 1] = ncy - 1


    #count number of particles in each cell 
    particles_per_cell = np.zeros((ncx, ncy), dtype=int)
    np.add.at(particles_per_cell, (ix, iy), 1)

    cell_velocities = cells = [
    velocities[(particle_cell_indices[:,0] == i) & (particle_cell_indices[:,1] == j)]
    for i in range(ncx) for j in range(ncy)
    ]

    cell_velocities = np.array(cells, dtype=object).reshape(ncx, ncy)


    cglobal_indices = [
    np.flatnonzero((particle_cell_indices[:,0] == i) & (particle_cell_indices[:,1] == j))
    for i in range(ncx) for j in range(ncy)
    ]

    cglobal_indices = np.array(cglobal_indices, dtype=object).reshape(ncx, ncy)

    temperature_history = np.zeros(n_tot)


    for n in range(n_tot):
        
        #compute upper bound cross section for each cell 

        upper_bound_cross_sections = np.array([
        hf.compute_upper_bound_cross_section(cell) if len(cell) else 0.0
        for cell in cell_velocities.ravel()
        ]).reshape(ncx, ncy)


        #physical number density per cell
        rho_cell = particles_per_cell / (cell_width * cell_height)

        # Expected pairs to collide per cell

        Nc = np.minimum(
            hf.Iround((particles_per_cell * rho_cell * upper_bound_cross_sections * dt)/ (2*e)), 
            particles_per_cell // 2
            )


        #sample and pair indices per cell 

        sampled_indices = hf.sample_particle_indices_to_collide_grid(Nc, cell_velocities) 


        indices_i_global = []
        indices_j_global = []

        for cx in range(ncx):
            for cy in range(ncy):
                idx_local = sampled_indices[cx,cy]

                if len(idx_local) == 0 :
                    continue

                pairs_loc = hf.pair_particle_indices_2d(idx_local)
                indices_particles_i = pairs_loc[:,0]
                indices_particles_j = pairs_loc[:,1]

                idx_map = cglobal_indices[cx,cy]
                indices_i_global.append(idx_map[indices_particles_i])
                indices_j_global.append(idx_map[indices_particles_j])

        indices_ij = (np.concatenate(indices_i_global) if indices_i_global else np.array([], dtype=int)).ravel()
        indices_kl = (np.concatenate(indices_j_global) if indices_j_global else np.array([], dtype=int)).ravel()

        if len(indices_ij) > 0 and len(indices_kl) > 0:
            indices_ij = np.asarray(indices_ij).reshape(-1)
            indices_kl = np.asarray(indices_kl).reshape(-1)

            v_rel = velocities[indices_ij] - velocities[indices_kl]
            v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)

            sigma_ijkl = hf.ArraySigma_VHS(v_rel_mag).reshape(-1)

            Sigma_pairs_of_pairs = upper_bound_cross_sections[particle_cell_indices[indices_ij][:, 0], particle_cell_indices[indices_ij][:, 1]]

            u_rand = np.random.rand(len(indices_ij)) * Sigma_pairs_of_pairs

            accept_condition  = (u_rand < sigma_ijkl)

            indices_ij = indices_ij[accept_condition]
            indices_kl = indices_kl[accept_condition]

            if len(indices_ij) > 0:
                velocities = hf.collide_particles_2d(velocities, indices_ij, indices_kl)

            positions = hf.update_positions_2d(positions, velocities, dt)

            positions = pb.periodic_BC_2d(positions, Lx, Ly)


            #Re-bin particles

            particle_cell_indices[particle_cell_indices[:,0] == ncx, 0] = ncx - 1
            particle_cell_indices[particle_cell_indices[:,1] == ncy, 1] = ncy - 1


            #count number of particles in each cell 
            particles_per_cell = np.zeros((ncx, ncy), dtype=int)
            np.add.at(particles_per_cell, (ix, iy), 1)

            cell_velocities = cells = [
            velocities[(particle_cell_indices[:,0] == i) & (particle_cell_indices[:,1] == j)]
            for i in range(ncx) for j in range(ncy)
            ]

            cell_velocities = np.array(cells, dtype=object).reshape(ncx, ncy)


            cglobal_indices = [
            np.flatnonzero((particle_cell_indices[:,0] == i) & (particle_cell_indices[:,1] == j))
            for i in range(ncx) for j in range(ncy)
            ]

            cglobal_indices = np.array(cglobal_indices, dtype=object).reshape(ncx, ncy)

        temperature_history[n]  = np.sum(velocities**2) / velocities.shape[0]

    return positions, velocities, temperature_history




