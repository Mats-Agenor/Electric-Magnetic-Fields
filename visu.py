import os
import re
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable



# ======================
# STYLE CONFIGURATIONS
# ======================
# These settings control the appearance of all matplotlib plots
plt.rcParams["axes.labelsize"] = 20       # Font size for axis labels
plt.rcParams["xtick.labelsize"] = 18      # Font size for x-axis ticks
plt.rcParams["ytick.labelsize"] = 18      # Font size for y-axis ticks
plt.rcParams['font.size'] = 20            # Base font size
plt.rc('font', **{'family':'serif', 'serif':['Times']})  # Font family
mpl.rcParams['figure.dpi'] = 100          # Figure resolution
# mpl.rcParams['text.usetex'] = True      # Uncomment to use LaTeX rendering
mpl.rcParams['legend.frameon'] = False    # No frame around legend
mpl.rcParams['font.family'] = 'STIXGeneral'  # Math font family
mpl.rcParams['mathtext.fontset'] = 'stix' # Math font style
mpl.rcParams['xtick.direction'] = 'in'    # X ticks inside the plot
mpl.rcParams['ytick.direction'] = 'in'    # Y ticks inside the plot
mpl.rcParams['xtick.top'] = True          # Show ticks on top axis
mpl.rcParams['ytick.right'] = True        # Show ticks on right axis
mpl.rcParams['xtick.major.size'] = 5      # Length of major x ticks
mpl.rcParams['xtick.minor.size'] = 3      # Length of minor x ticks
mpl.rcParams['ytick.major.size'] = 5      # Length of major y ticks
mpl.rcParams['ytick.minor.size'] = 3      # Length of minor y ticks
mpl.rcParams['xtick.major.width'] = 0.79  # Width of major x ticks
mpl.rcParams['xtick.minor.width'] = 0.79  # Width of minor x ticks
mpl.rcParams['ytick.major.width'] = 0.79  # Width of major y ticks
mpl.rcParams['ytick.minor.width'] = 0.79  # Width of minor y ticks
plt.rcParams['figure.constrained_layout.use'] = True  # Better layout management

# ======================
# FILE MANAGEMENT
# ======================
# Create directory for output plots if it doesn't exist
os.makedirs('outputs_plots', exist_ok=True)

# ======================
# DATA PROCESSING FUNCTIONS
# ======================

def read_simulation_data(filename):
    """
    Read particle simulation data from a file and parse it into a structured format.
    
    Args:
        filename (str): Path to the input data file
        
    Returns:
        list: A list of time steps, each containing particle data dictionaries
    """
    with open(filename, 'r') as f:
        data = f.read()
    
    # Split the data into different time steps (separated by double newlines)
    time_steps = [step for step in data.split('\n\n') if step.strip()]
    
    all_data = []
    # Process each time step with progress bar
    for step in tqdm(time_steps, desc="Processing time steps"):
        particles = []
        # Process each particle in the time step
        for line in step.split('\n'):
            if line.strip():
                parts = list(map(float, line.split()))
                # Create dictionary for each particle's properties
                particles.append({
                    'x': parts[0], 'y': parts[1], 'z': parts[2],  # Position
                    'q': parts[3],                                 # Charge
                    'Ex': parts[4], 'Ey': parts[5], 'Ez': parts[6],  # Electric field
                    'Bx': parts[7], 'By': parts[8], 'Bz': parts[9]    # Magnetic field
                })
        all_data.append(particles)
    
    return all_data

def create_density_map(x, y, values, weights, grid_size=100, sigma=0.02):
    """
    Create a 2D density map using Gaussian kernel smoothing.
    
    Args:
        x (list): X-coordinates of particles
        y (list): Y-coordinates of particles
        values (list): Values to be distributed (e.g., field magnitudes)
        weights (list): Weights for each particle (e.g., charge)
        grid_size (int): Number of grid points in each dimension
        sigma (float): Standard deviation for Gaussian kernel
        
    Returns:
        tuple: (xx, yy, density) meshgrid and resulting density map
    """
    # Create coordinate grid
    grid_x = np.linspace(0, 1, grid_size)
    grid_y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    density = np.zeros_like(xx)
    
    # Apply Gaussian kernel to each particle
    for xi, yi, val, w in zip(x, y, values, weights):
        # 2D Gaussian distribution centered at particle position
        gauss = np.exp(-((xx-xi)**2 + (yy-yi)**2)/(2*sigma**2))
        density += val * w * gauss
    
    return xx, yy, density

# ======================
# VISUALIZATION FUNCTIONS
# ======================

def plot_initial_distribution_3d(data):
    """
    Create a 3D scatter plot of the initial particle distribution.
    
    Args:
        data (list): Simulation data containing particle information
    """
    initial = data[0]  # Get first time step
    x = [p['x'] for p in initial]
    y = [p['y'] for p in initial]
    z = [p['z'] for p in initial]
    q = [p['q'] for p in initial]  # Charge for coloring
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D scatter plot with color mapping based on charge
    sc = ax.scatter(x, y, z, c=q, cmap='coolwarm', alpha=0.8, 
                   edgecolors='w', linewidth=0.3, s=30)
    
    # Add colorbar and labels
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, label='Charge (q)')
    ax.set_title('Initial Particle Distribution (3D)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    plt.savefig('outputs_plots/initial_distribution_3d.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_initial_distribution(data):
    """
    Create a 2D scatter plot of the initial particle distribution.
    
    Args:
        data (list): Simulation data containing particle information
    """
    initial = data[0]  # Get first time step
    x = [p['x'] for p in initial]
    y = [p['y'] for p in initial]
    q = [p['q'] for p in initial]  # Charge for coloring
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=q, cmap='coolwarm', alpha=0.7, 
               edgecolors='w', linewidths=0.2, s=20)
    plt.colorbar(label='Charge (q)')
    plt.title('Initial Particle Distribution (2D)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True, alpha=0.2)
    plt.savefig('outputs_plots/initial_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_evolution_animation(data, frames=100):
    """
    Create an animation showing particle evolution over time.
    
    Args:
        data (list): Complete simulation data
        frames (int): Number of frames for the animation
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select subset of frames to animate
    step = max(1, len(data)//frames)
    anim_data = data[::step]
    
    def update(frame):
        """Update function for animation frames."""
        ax.clear()
        particles = anim_data[frame]
        x = [p['x'] for p in particles]
        y = [p['y'] for p in particles]
        q = [p['q'] for p in particles]
        
        # Create scatter plot for current frame
        sc = ax.scatter(x, y, c=q, cmap='coolwarm', alpha=0.7, 
                       edgecolors='w', linewidths=0.2, s=20)
        ax.set_title(f'Time Evolution - Step {frame*step}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(0, 1)  # Fixed limits for consistent animation
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        return sc,
    
    # Create animation object
    ani = FuncAnimation(fig, update, frames=len(anim_data), 
                        interval=50, blit=True)
    
    # Save animation as MP4
    print("Saving particle evolution animation...")
    ani.save('outputs_plots/particle_evolution.mp4', writer='ffmpeg', 
             fps=5, dpi=150, bitrate=1800)
    plt.close()

def plot_final_fields(data):
    """
    Plot 2D density maps of electric and magnetic fields at final time step.
    
    Args:
        data (list): Complete simulation data
    """
    final = data[-1]  # Get last time step
    x = [p['x'] for p in final]
    y = [p['y'] for p in final]
    q = [abs(p['q']) for p in final]  # Use absolute charge as weight
    
    # Calculate electric field magnitudes (convert to kV/m)
    Ex = [p['Ex']*1e-3 for p in final]
    Ey = [p['Ey']*1e-3 for p in final]
    Emag = [np.sqrt(ex**2 + ey**2) for ex, ey in zip(Ex, Ey)]
    
    # Calculate magnetic field magnitudes (convert to microTesla)
    Bx = [p['Bx']*1e6 for p in final]
    By = [p['By']*1e6 for p in final]
    Bmag = [np.sqrt(bx**2 + by**2) for bx, by in zip(Bx, By)]
    
    # Parameters for density maps
    grid_size = 200
    sigma = 0.02
    
    print("Calculating electric field map...")
    xx, yy, Edens = create_density_map(x, y, Emag, q, grid_size, sigma)
    
    print("Calculating magnetic field map...")
    _, _, Bdens = create_density_map(x, y, Bmag, q, grid_size, sigma)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot electric field
    im1 = ax1.pcolormesh(xx, yy, Edens, cmap='inferno', shading='auto')
    fig.colorbar(im1, ax=ax1, label='|E| (kV/m)')
    ax1.set_title('Final Electric Field Magnitude')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    
    # Plot magnetic field
    im2 = ax2.pcolormesh(xx, yy, Bdens, cmap='viridis', shading='auto')
    fig.colorbar(im2, ax=ax2, label='|B| (μT)')
    ax2.set_title('Final Magnetic Field Magnitude')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    
    plt.savefig('outputs_plots/final_fields.png', dpi=200, bbox_inches='tight')
    plt.close()

def create_field_animation(data, frames=50):
    """
    Create an animation showing the evolution of electric and magnetic fields.
    
    Args:
        data (list): Complete simulation data
        frames (int): Number of frames for the animation
    """
    # Select subset of frames to animate
    step = max(1, len(data)//frames)
    anim_data = data[::step]
    grid_size = 100
    sigma = 0.03
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def update(frame):
        """Update function for animation frames."""
        particles = anim_data[frame]
        x = [p['x'] for p in particles]
        y = [p['y'] for p in particles]
        q = [abs(p['q']) for p in particles]  # Use absolute charge as weight
        
        # Calculate electric field magnitudes (convert to kV/m)
        Ex = [p['Ex']*1e-3 for p in particles]
        Ey = [p['Ey']*1e-3 for p in particles]
        Emag = [np.sqrt(ex**2 + ey**2) for ex, ey in zip(Ex, Ey)]
        
        # Calculate magnetic field magnitudes (convert to microTesla)
        Bx = [p['Bx']*1e6 for p in particles]
        By = [p['By']*1e6 for p in particles]
        Bmag = [np.sqrt(bx**2 + by**2) for bx, by in zip(Bx, By)]
        
        # Create density maps for current frame
        xx, yy, Edens = create_density_map(x, y, Emag, q, grid_size, sigma)
        _, _, Bdens = create_density_map(x, y, Bmag, q, grid_size, sigma)
        
        # Clear previous frame
        ax1.clear()
        ax2.clear()
        
        # Plot electric field
        im1 = ax1.pcolormesh(xx, yy, Edens, cmap='inferno', shading='auto')
        ax1.set_title(f'Electric Field | Step {frame*step}')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        
        # Plot magnetic field
        im2 = ax2.pcolormesh(xx, yy, Bdens, cmap='viridis', shading='auto')
        ax2.set_title(f'Magnetic Field | Step {frame*step}')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        
        # Add colorbars with proper positioning
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im1, cax=cax1, label='|E| (kV/m)')
        
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax2, label='|B| (μT)')
        
        return im1, im2
    
    # Create animation object
    ani = FuncAnimation(fig, update, frames=len(anim_data), 
                        interval=100, blit=False)
    
    # Save animation as MP4
    print("Saving field evolution animation...")
    ani.save('outputs_plots/field_evolution.mp4', writer='ffmpeg', 
             fps=5, dpi=150, bitrate=1800)
    plt.close()

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    data_file = "trajectories_fields.dat"
    
    print("Reading simulation data...")
    simulation_data = read_simulation_data(data_file)
    
    print("\nCreating visualizations:")
    print("1. Plotting initial 2D distribution...")
    plot_initial_distribution(simulation_data)
    
    print("2. Plotting initial 3D distribution...")
    plot_initial_distribution_3d(simulation_data)
    
    print("3. Creating particle evolution animation...")
    create_evolution_animation(simulation_data, frames=100)
    
    print("4. Plotting final fields...")
    plot_final_fields(simulation_data)
    
    print("5. Creating field evolution animation...")
    create_field_animation(simulation_data, frames=50)
    
    print("\nAll plots and animations saved to 'outputs_plots/'")
