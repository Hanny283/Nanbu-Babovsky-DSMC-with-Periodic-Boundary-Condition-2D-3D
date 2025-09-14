def periodic_BC_2d (positions, Lx, Ly):
    positions[:,0] = np.mod(positions[:,0], Lx)
    positions[:,1] = np.mod(positions[:,1], Ly)
    return positions

def periodic_BC_3d (positions, Lx, Ly, Lz):
    positions[:,0] = np.mod(positions[:,0], Lx)
    positions[:,1] = np.mod(positions[:,1], Ly)
    positions[:,2] = np.mod(positions[:,2], Lz)
    return positions