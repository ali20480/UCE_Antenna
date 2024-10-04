"""
A. Al-Zawqari, A. Safa, G. Vandersteen "Automating the Design of Multi-band Microstrip Antennas via Uniform Cross-Entropy Optimization," 
2024 

Method of Moments Antenna simulation back-end

Vrije Universiteit Brussels, Belgium
College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

plt.close('all')


def execute_patch_simulation(patch_length_ = 6e-3, patch_width_ = 6e-3, h1_ = 0.4, h2_ = 0.3, w1_ = 0.1):
    # Constants
    patch_length = patch_length_ #6e-3  # 4 mm (length)
    patch_width = patch_width_   # 4 mm (width)
    Z0 = 50              # Characteristic impedance (50 Ohms)
    frequency_start = 10e9  # Start frequency (1 GHz)
    frequency_end = 50e9   # End frequency (40 GHz)
    frequencies = np.linspace(frequency_start, frequency_end, 21)
    
    # Create patch with cutout geometry and grid points
    def create_cut_rectangle_patch(patch_size1, patch_size2, h1=0.4, h2=0.3, w1=0.1, resolution=20):
        """
        Create a patch with a cutout in the corner and generate mesh grid inside the patch boundary.
        """
        w1 = np.minimum(w1, 1) * patch_size1
        h2 = np.minimum(h1 + h2, 1) * patch_size2
        h1 = np.minimum(h1, 1) * patch_size2
    
        # Define the patch boundary (cutout design)
        x_boundary = np.array([
            0, patch_size1, 
            patch_size1, patch_size1 - w1,
            patch_size1 - w1, patch_size1,
            patch_size1, 0, 
            0
        ])
        y_boundary = np.array([
            0, 0, 
            patch_size2 - h2, patch_size2 - h2,
            patch_size2 - h1, patch_size2 - h1,
            patch_size2, patch_size2, 
            0
        ])
    
        # Create polygon boundary path
        patch_boundary = np.column_stack((x_boundary, y_boundary))
        patch_path = Path(patch_boundary)
    
        # Create meshgrid over the bounding box of the patch
        x_min, x_max = 0, patch_size1
        y_min, y_max = 0, patch_size2
        X, Y = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    
        # Filter points inside the patch boundary
        points = np.column_stack((X.ravel(), Y.ravel()))
        mask = patch_path.contains_points(points)
        X_inside = X.ravel()[mask]
        Y_inside = Y.ravel()[mask]
    
        return X_inside, Y_inside, x_boundary, y_boundary
    
    
    
    # Generate patch geometry
    num_segments = 15  # Number of segments
    X_inside, Y_inside, x_boundary, y_boundary = create_cut_rectangle_patch(patch_length, patch_width, h1=h1_, h2=h2_, w1=w1_, resolution=num_segments)
    

    # Compute the Green's function
    def greens_function(r1, r2, freq):
        k = 2 * np.pi * freq / 3e8  # Wavenumber
        r = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2)
        if r == 0:
            return 1 / (4 * np.pi)  # Handle singularity case
        return np.exp(-1j * k * r) / (4 * np.pi * r)
    
    # Function to solve matrix equation
    def solve_for_currents(A, b):
        try:
            currents = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Matrix A is singular or ill-conditioned.")
            currents = np.zeros_like(b)
        return currents
    
    # Set up arrays for results
    S11 = np.zeros(len(frequencies), dtype=complex)
    Zin = np.zeros(len(frequencies), dtype=complex)
    
    # Discretize the patch into segments and setup MoM equations
    for i, freq in enumerate(frequencies):

        A = np.zeros((len(X_inside), len(X_inside)), dtype=complex)
    
        # Define a frequency-dependent 2D sinusoidal current distribution
        sinusoidal_voltage = np.zeros(len(X_inside), dtype=complex)
        wavelength = 3e8 / freq
        for m in range(len(X_inside)):
            x = X_inside[m]
            y = Y_inside[m]
            sinusoidal_voltage[m] = np.sin(2 * np.pi * (x + y) / wavelength) #np.sin(2 * np.pi * ((x -len(X_inside)//2) + (y -len(X_inside)//2)) / wavelength)  # 2D sinusoidal distribution
    
        # Construct matrix A and vector b
        for m in range(len(X_inside)):
            for n in range(len(X_inside)):
                x1, y1 = X_inside[m], Y_inside[m]
                x2, y2 = X_inside[n], Y_inside[n]
                A[m, n] = greens_function((x1, y1), (x2, y2), freq)
    
        # Solve for currents
        currents = solve_for_currents(A, sinusoidal_voltage)
    
        # Compute S11 and Input Impedance at the center segment
        center_index = len(X_inside) // 2
        def compute_s11(currents, freq, center_index):
            Z = np.abs(sinusoidal_voltage[center_index] / currents[center_index])
            S11 = (Z - Z0) / (Z + Z0)
            return S11
    
        S11[i] = compute_s11(currents, freq, center_index)
        Zin[i] = Z0 * (1 + S11[i]) / (1 - S11[i])
        

    S11 = np.maximum(S11, 0.01)
        
    return frequencies, S11, X_inside, Y_inside, x_boundary, y_boundary

