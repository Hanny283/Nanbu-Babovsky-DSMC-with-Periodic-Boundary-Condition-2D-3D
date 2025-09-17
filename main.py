import numpy as np
import matplotlib.pyplot as plt
from nanbu_babovsky import Nanbu_Babovsky_2D_Periodic

def main():
    # Simulation parameters
    N = 1000          # Number of particles
    dt = 0.01         # Time step
    n_tot = 100       # Total number of time steps
    e = 1.0           # Energy parameter
    mu = 1.0          # Mass parameter
    alpha = 1.0       # Alpha parameter for VHS cross section
    Lx = 10.0         # Domain length in x
    Ly = 10.0         # Domain length in y
    ncx = 10          # Number of cells in x direction
    ncy = 10          # Number of cells in y direction
    S = 1.0           # Source term parameter
    dx = Lx / ncx     # Cell size in x direction
    T_x0 = 1.0        # Initial temperature in x direction
    T_y0 = 1.0        # Initial temperature in y direction
    
    print("Starting Nanbu-Babovsky 2D DSMC simulation...")
    print(f"Parameters: N={N}, dt={dt}, n_tot={n_tot}")
    print(f"Domain: Lx={Lx}, Ly={Ly}, cells: {ncx}x{ncy}")
    
    # Run the simulation
    positions, velocities, temperature_history = Nanbu_Babovsky_2D_Periodic(
        N, dt, n_tot, e, mu, alpha, Lx, Ly, ncx, ncy, S, dx, T_x0, T_y0
    )
    
    print("Simulation completed!")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Nanbu-Babovsky 2D DSMC Simulation Results', fontsize=16)
    
    # Plot 1: Final particle positions
    ax1 = axes[0, 0]
    ax1.scatter(positions[:, 0], positions[:, 1], alpha=0.6, s=1)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Final Particle Positions')
    ax1.grid(True)
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, Ly)
    
    # Plot 2: Final velocity distribution
    ax2 = axes[0, 1]
    ax2.scatter(velocities[:, 0], velocities[:, 1], alpha=0.6, s=1)
    ax2.set_xlabel('Vx')
    ax2.set_ylabel('Vy')
    ax2.set_title('Final Velocity Distribution')
    ax2.grid(True)
    
    # Plot 3: Temperature evolution
    ax3 = axes[1, 0]
    time_steps = np.arange(n_tot)
    ax3.plot(time_steps, temperature_history)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Temperature Evolution')
    ax3.grid(True)
    
    # Plot 4: Velocity magnitude distribution
    ax4 = axes[1, 1]
    velocity_magnitude = np.linalg.norm(velocities, axis=1)
    ax4.hist(velocity_magnitude, bins=50, alpha=0.7, density=True)
    ax4.set_xlabel('Velocity Magnitude')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Velocity Magnitude Distribution')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('nanbu_babovsky_2d_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'nanbu_babovsky_2d_results.png'")
    
    # Show the plot (optional - comment out if running in headless mode)
    # plt.show()
    
    # Print some statistics
    print(f"\nSimulation Statistics:")
    print(f"Final average velocity magnitude: {np.mean(velocity_magnitude):.3f}")
    print(f"Final velocity std: {np.std(velocity_magnitude):.3f}")
    print(f"Final temperature: {temperature_history[-1]:.3f}")
    print(f"Temperature change: {temperature_history[-1] - temperature_history[0]:.3f}")

if __name__ == "__main__":
    main()
