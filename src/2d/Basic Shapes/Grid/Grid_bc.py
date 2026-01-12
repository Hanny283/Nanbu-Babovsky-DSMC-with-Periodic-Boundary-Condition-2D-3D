import numpy as np

def periodic_BC_2d (positions, Lx, Ly):
    positions[:,0] = np.mod(positions[:,0], Lx)
    positions[:,1] = np.mod(positions[:,1], Ly)
    return positions
