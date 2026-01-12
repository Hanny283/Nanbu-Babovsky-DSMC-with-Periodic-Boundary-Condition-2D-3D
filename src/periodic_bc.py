import numpy as np
import helpers as hf


def periodic_BC_3d (positions, Lx, Ly, Lz):
    positions[:,0] = np.mod(positions[:,0], Lx)
    positions[:,1] = np.mod(positions[:,1], Ly)
    positions[:,2] = np.mod(positions[:,2], Lz)
    return positions

def reflecting_BC_2d (velocities, positions, Lx, Ly):

    m_left  = positions[:, 0] < 0
    m_right = positions[:, 0] > Lx
    m_bottom = positions[:, 1] < 0
    m_top    = positions[:, 1] > Ly

    positions[m_left,  0] = 0  - np.abs(positions[m_left,  0] - 0)  * np.sign(velocities[m_left,  0])
    positions[m_right, 0] = Lx - np.abs(positions[m_right, 0] - Lx) * np.sign(velocities[m_right, 0])
    positions[m_bottom,1] = 0  - np.abs(positions[m_bottom,1] - 0)  * np.sign(velocities[m_bottom,1])
    positions[m_top,   1] = Ly - np.abs(positions[m_top,   1] - Ly) * np.sign(velocities[m_top,   1])

    
    velocities[m_left,  0] *= -1
    velocities[m_right, 0] *= -1
    velocities[m_bottom,1] *= -1
    velocities[m_top,   1] *= -1

    return velocities, positions



def reflecting_BC_3d ( velocities, positions, Lx, Ly, Lz):
    m_left  = positions[:, 0] < 0
    m_right = positions[:, 0] > Lx
    m_bottom = positions[:, 1] < 0
    m_top    = positions[:, 1] > Ly
    m_front  = positions[:, 2] < 0
    m_back   = positions[:, 2] > Lz
    
    positions[m_left,  0] = 0  - np.abs(positions[m_left,  0] - 0)  * np.sign(velocities[m_left,  0])
    positions[m_right, 0] = Lx - np.abs(positions[m_right, 0] - Lx) * np.sign(velocities[m_right, 0])
    positions[m_bottom,1] = 0  - np.abs(positions[m_bottom,1] - 0)  * np.sign(velocities[m_bottom,1])
    positions[m_top,   1] = Ly - np.abs(positions[m_top,   1] - Ly) * np.sign(velocities[m_top,   1])
    positions[m_front, 2] = 0  - np.abs(positions[m_front, 2] - 0)  * np.sign(velocities[m_front, 2])
    positions[m_back,  2] = Lz - np.abs(positions[m_back,  2] - Lz) * np.sign(velocities[m_back,  2])
    
    velocities[m_left,  0] *= -1
    velocities[m_right, 0] *= -1
    velocities[m_bottom,1] *= -1
    velocities[m_top,   1] *= -1
    velocities[m_front, 2] *= -1
    velocities[m_back,  2] *= -1
    
    return velocities, positions


def reflecting_BC_spherical(velocities, positions, R):
    pos_x, pos_y, pos_z = positions[:, 0], positions[:, 1], positions[:, 2]
    vel_x, vel_y, vel_z = velocities[:, 0], velocities[:, 1], velocities[:, 2]

    radius_squared = pos_x * pos_x + pos_y * pos_y + pos_z * pos_z
    outside_mask = radius_squared > R * R
    
    if not np.any(outside_mask):
        return velocities, positions

    # Gather particles that ended outside the sphere
    pos_x_out = pos_x[outside_mask]
    pos_y_out = pos_y[outside_mask]
    pos_z_out = pos_z[outside_mask]
    vel_x_out = vel_x[outside_mask]
    vel_y_out = vel_y[outside_mask]
    vel_z_out = vel_z[outside_mask]

    # Solve for backtracked time-to-collision tau >= 0:
    # |p - v * tau|^2 = R^2 -> (v·v) tau^2 - 2(p·v) tau + (p·p - R^2) = 0
    speed_sq = vel_x_out * vel_x_out + vel_y_out * vel_y_out + vel_z_out * vel_z_out
    pos_dot_vel = pos_x_out * vel_x_out + pos_y_out * vel_y_out + pos_z_out * vel_z_out
    pos_sq_minus_R2 = (pos_x_out * pos_x_out + pos_y_out * pos_y_out + pos_z_out * pos_z_out) - R * R

    # Numerical guards
    tiny = 1e-300
    speed_sq_safe = np.where(speed_sq < tiny, 1.0, speed_sq)  # avoid division; will zero tau below
    discriminant = pos_dot_vel * pos_dot_vel - speed_sq * pos_sq_minus_R2
    discriminant = np.maximum(discriminant, 0.0)
    sqrt_discriminant = np.sqrt(discriminant)

    # Smallest nonnegative root
    tau = (pos_dot_vel - sqrt_discriminant) / speed_sq_safe
    # If velocity is nearly zero, take tau = 0 (snap to boundary along radial later)
    tau = np.where(speed_sq < tiny, 0.0, tau)
    tau = np.maximum(tau, 0.0)

    # Contact point on the sphere
    contact_x = pos_x_out - vel_x_out * tau
    contact_y = pos_y_out - vel_y_out * tau
    contact_z = pos_z_out - vel_z_out * tau

    # Outward unit normal at contact
    normal_x = contact_x / R
    normal_y = contact_y / R
    normal_z = contact_z / R

    # Specular reflection: v' = v - 2 (v·n) n
    vel_dot_normal = vel_x_out * normal_x + vel_y_out * normal_y + vel_z_out * normal_z
    refl_vel_x = vel_x_out - 2.0 * vel_dot_normal * normal_x
    refl_vel_y = vel_y_out - 2.0 * vel_dot_normal * normal_y
    refl_vel_z = vel_z_out - 2.0 * vel_dot_normal * normal_z


    # Advance from contact for remaining time tau with reflected velocity
    new_pos_x = contact_x + refl_vel_x * tau
    new_pos_y = contact_y + refl_vel_y * tau
    new_pos_z = contact_z + refl_vel_z * tau

    # Numerical clamping: if still slightly outside, project back to sphere
    new_r2 = new_pos_x * new_pos_x + new_pos_y * new_pos_y + new_pos_z * new_pos_z
    outside_again = new_r2 > (R * R)
    if np.any(outside_again):
        scale = (R / np.sqrt(new_r2[outside_again]))
        new_pos_x[outside_again] *= scale
        new_pos_y[outside_again] *= scale
        new_pos_z[outside_again] *= scale

    # Write back
    positions[outside_mask, 0] = new_pos_x
    positions[outside_mask, 1] = new_pos_y
    positions[outside_mask, 2] = new_pos_z
    velocities[outside_mask, 0] = refl_vel_x
    velocities[outside_mask, 1] = refl_vel_y
    velocities[outside_mask, 2] = refl_vel_z

    return velocities, positions

def maxwell_bc_2d(positions, velocities, Lx, Ly, alpha, T_x0, T_y0, rng=np.random):
    x, y = positions[:,0], positions[:,1]
    vx, vy = velocities[:,0], velocities[:,1]

    def handle_side(mask, wall_pos, normal_axis, tangent_axis, into_sign):
        """
        mask: particles that ended past this wall after free flight
        wall_pos: 0 or Lx/Ly
        normal_axis: 0 for x walls, 1 for y walls
        tangent_axis: 1 or 0 (the other axis)
        into_sign: +1 if the normal component must be >0 after emission (into the domain),
                   -1 if it must be <0 after emission.
        """
        if not np.any(mask):
            return

        idx = np.flatnonzero(mask)
        # Pre-collision components
        v_n_old = velocities[idx, normal_axis]
        v_t_old = velocities[idx, tangent_axis]
        pos_n = positions[idx, normal_axis]
        pos_t = positions[idx, tangent_axis]

        # Overshoot distance beyond the wall along the normal
        if into_sign == +1:  # wall at 0 (left/bottom)
            overshoot = -(pos_n - wall_pos)  # positive value
        else:                 # wall at L (right/top)
            overshoot = (pos_n - wall_pos)   # positive value

        # Time from collision to end of step (remaining time)
        dt_rem = overshoot / np.abs(v_n_old)

        # Position at the instant of collision (rewind along old velocity)
        pos_t_coll = pos_t - v_t_old * dt_rem

        # Decide Maxwell vs specular per particle
        do_maxwell = rng.random(idx.size) < alpha
        idx_M = idx[do_maxwell]
        idx_S = idx[~do_maxwell]

        # --- Specular: flip normal, keep tangential; then advance for dt_rem from the wall ---
        if idx_S.size:
            v_n_new = -velocities[idx_S, normal_axis]
            v_t_new = velocities[idx_S, tangent_axis]

            # New positions start at wall, then advance for dt_rem with new velocity
            positions[idx_S, normal_axis] = wall_pos + into_sign * np.abs(v_n_new) * dt_rem[~do_maxwell]
            positions[idx_S, tangent_axis] = pos_t_coll[~do_maxwell] + v_t_new * dt_rem[~do_maxwell]

            velocities[idx_S, normal_axis] = v_n_new
            velocities[idx_S, tangent_axis] = v_t_new

        # --- Maxwell: sample from wall Maxwellian, enforce into-domain sign on normal ---
        if idx_M.size:
            new_v = hf.sample_velocities_from_maxwellian_2d(T_x0, T_y0, idx_M.size)
            # normal/tangent mapping
            v_n_draw = new_v[:, 0] if normal_axis == 0 else new_v[:, 1]
            v_t_draw = new_v[:, 1] if normal_axis == 0 else new_v[:, 0]

            v_n_new = into_sign * np.abs(v_n_draw)   # half-Maxwellian into the domain
            v_t_new = v_t_draw

            # Advance from the wall for dt_rem with new velocity
            positions[idx_M, normal_axis] = wall_pos + v_n_new * dt_rem[do_maxwell]
            positions[idx_M, tangent_axis] = pos_t_coll[do_maxwell] + v_t_new * dt_rem[do_maxwell]

            velocities[idx_M, normal_axis] = v_n_new
            velocities[idx_M, tangent_axis] = v_t_new

    # Masks: after streaming, who is outside?
    left   = x < 0
    right  = x > Lx
    bottom = y < 0
    top    = y > Ly

    handle_side(left,   wall_pos=0.0, normal_axis=0, tangent_axis=1, into_sign=+1)
    handle_side(right,  wall_pos=Lx,  normal_axis=0, tangent_axis=1, into_sign=-1)
    handle_side(bottom, wall_pos=0.0, normal_axis=1, tangent_axis=0, into_sign=+1)
    handle_side(top,    wall_pos=Ly,  normal_axis=1, tangent_axis=0, into_sign=-1)

    return velocities, positions


def maxwell_bc_3d(
    positions, velocities,
    Lx, Ly, Lz,
    alpha, Tw_x, Tw_y, Tw_z,
    rng=np.random,
):
    

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]

    def handle_wall(mask, wall_pos, normal_axis, tangent_axes, into_sign):
        """
        mask: boolean array for particles that ended outside across this wall
        wall_pos: 0 or L (the coordinate of the wall along normal_axis)
        normal_axis: 0 (x), 1 (y), or 2 (z)
        tangent_axes: tuple with the other two axes, e.g., (1,2)
        into_sign: +1 if post-wall normal velocity must be >0 (into domain), -1 if <0.
        """
        if not np.any(mask):
            return

        idx = np.flatnonzero(mask)

        pos_n = positions[idx, normal_axis]
        v_n_old = velocities[idx, normal_axis]
        t1, t2 = tangent_axes
        v_t1_old = velocities[idx, t1]
        v_t2_old = velocities[idx, t2]

        # overshoot distance along the normal (always positive)
        if into_sign == +1:   # wall at 0 (left/bottom/front)
            overshoot = -(pos_n - wall_pos)
        else:                 # wall at L (right/top/back)
            overshoot =  (pos_n - wall_pos)

        # remaining time after collision: Δt̃ = overshoot / |v_n_old|
        # (assumes v_n_old != 0 for those that actually crossed; protect against zeros)
        denom = np.abs(v_n_old)
        # avoid division by zero (if any degenerate case sneaks in)
        denom = np.where(denom == 0.0, 1e-300, denom)
        dt_rem = overshoot / denom

        # positions at the instant of collision (rewind along old velocity)
        pos_t1_coll = positions[idx, t1] - v_t1_old * dt_rem
        pos_t2_coll = positions[idx, t2] - v_t2_old * dt_rem

        # Bernoulli split per particle
        maxwell_flag = rng.random(idx.size) < alpha
        idx_M = idx[maxwell_flag]
        idx_S = idx[~maxwell_flag]

        # --- Specular: flip only the normal component, keep both tangential unchanged ---
        if idx_S.size:
            # old -> new velocities
            v_n_new_S  = -velocities[idx_S, normal_axis]
            v_t1_new_S =  velocities[idx_S, t1]
            v_t2_new_S =  velocities[idx_S, t2]

            # advance from the wall with the new velocity for the remaining time
            positions[idx_S, normal_axis] = wall_pos + into_sign * np.abs(v_n_new_S) * dt_rem[~maxwell_flag]
            positions[idx_S, t1]          = pos_t1_coll[~maxwell_flag] + v_t1_new_S * dt_rem[~maxwell_flag]
            positions[idx_S, t2]          = pos_t2_coll[~maxwell_flag] + v_t2_new_S * dt_rem[~maxwell_flag]

            velocities[idx_S, normal_axis] = v_n_new_S
            velocities[idx_S, t1]          = v_t1_new_S
            velocities[idx_S, t2]          = v_t2_new_S

        # --- Maxwell: sample new velocity from wall Maxwellian and enforce half-space sign on normal ---
        if idx_M.size:
            v_draw = hf.sample_velocities_from_maxwellian_3d(Tw_x, Tw_y, Tw_z, idx_M.size)  # (n,3)
            # Map draws to (n,t1,t2) in the wall's local frame:
            v_n_draw  = v_draw[:, normal_axis]
            v_t1_draw = v_draw[:, t1]
            v_t2_draw = v_draw[:, t2]

            v_n_new_M  = into_sign * np.abs(v_n_draw)  # half-Maxwellian into the domain
            v_t1_new_M = v_t1_draw
            v_t2_new_M = v_t2_draw

            positions[idx_M, normal_axis] = wall_pos + v_n_new_M * dt_rem[maxwell_flag]
            positions[idx_M, t1]          = pos_t1_coll[maxwell_flag] + v_t1_new_M * dt_rem[maxwell_flag]
            positions[idx_M, t2]          = pos_t2_coll[maxwell_flag] + v_t2_new_M * dt_rem[maxwell_flag]

            velocities[idx_M, normal_axis] = v_n_new_M
            velocities[idx_M, t1]          = v_t1_new_M
            velocities[idx_M, t2]          = v_t2_new_M

    # Determine who crossed each wall after free flight
    left   = x < 0.0
    right  = x > Lx
    bottom = y < 0.0
    top    = y > Ly
    front  = z < 0.0
    back   = z > Lz

    # Apply the BCs for all six faces
    handle_wall(left,   wall_pos=0.0, normal_axis=0, tangent_axes=(1,2), into_sign=+1)
    handle_wall(right,  wall_pos=Lx,  normal_axis=0, tangent_axes=(1,2), into_sign=-1)

    handle_wall(bottom, wall_pos=0.0, normal_axis=1, tangent_axes=(0,2), into_sign=+1)
    handle_wall(top,    wall_pos=Ly,  normal_axis=1, tangent_axes=(0,2), into_sign=-1)

    handle_wall(front,  wall_pos=0.0, normal_axis=2, tangent_axes=(0,1), into_sign=+1)
    handle_wall(back,   wall_pos=Lz,  normal_axis=2, tangent_axes=(0,1), into_sign=-1)

    return velocities, positions

   
