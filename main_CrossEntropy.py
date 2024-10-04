"""
A. Al-Zawqari, A. Safa, G. Vandersteen "Automating the Design of Multi-band Microstrip Antennas via Uniform Cross-Entropy Optimization," 
2024.

MAIN FILE: Using the Cross Entropy sampling method for finding the dimensions of a Microstrip Patch antenna with cutout.

Vrije Universiteit Brussels, Belgium
College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar

"""

import numpy as np
from SimulateAntennaMoM import execute_patch_simulation
import matplotlib.pyplot as plt
import time
plt.close('all')

LL = 0.1 # maximum dimensions of antenna in meters

"""
Define target Spectrum
"""

target_spectrum = np.load("Target_S11_dB_Double.npz", allow_pickle = True)['data'] #Double Peak
peak_idx = np.argsort(target_spectrum)[:2]
def optimization(x):
    frequencies, S11, X_inside, Y_inside, x_boundary, y_boundary = execute_patch_simulation(patch_length_ = x[0], patch_width_ = x[1], h1_ = x[2], h2_ = x[3], w1_ = x[4])
    # Compute the residuals
    residual = target_spectrum - 20 * np.log10(np.abs(S11))
    
    quantile = 0.9
    
    # Dynamically set c as the specified quantile of the absolute residuals
    c = np.quantile(np.abs(residual), quantile)
    
    # Compute the squared loss for small residuals
    squared_loss = np.abs(residual)
    
    # Compute the mixed L1 and squared loss for large residuals
    mixed_loss = (residual ** 2 + c**2) / (2 * c)
    
    # Apply the piecewise function
    loss = np.sum(np.where(np.abs(residual) <= c, squared_loss, mixed_loss))
    
    
    return loss


"""
Cross-Entropy optimization
"""
# Cross-Entropy Method
def cross_entropy_method(
    func,         # The function to optimize
    n_samples,    # Number of samples to generate each iteration
    elite_frac,   # Fraction of samples considered elite
    n_iterations, # Number of iterations
    mu_init,      # Initial mean of the sampling distribution
    sigma_init    # Initial standard deviation of the sampling distribution
):
    
    global LL
    
    mu = mu_init
    sigma = sigma_init
    best_sol = 0
    scores_prev = 10000

    n_elite = int(np.ceil(elite_frac * n_samples))
    
    LOSS = []

    for iteration in range(n_iterations):

        samples = np.zeros((n_samples, 5))
    
        
        As = mu - sigma * np.sqrt(12) / 2 #transform mean and standard deviations computed by the method to 
        Bs = mu + sigma * np.sqrt(12) / 2 # the bounds of the Uniform distribution [A, B]
        
        As = np.minimum(np.maximum(As, 0), 1) #Restrain the bounds to 0, LL for x,y and 0,1 (0,100%) for the cutted area
        As[:2] = np.minimum(As[:2], LL)
        
        Bs = np.minimum(np.maximum(Bs, 0), 1)
        Bs[:2] = np.minimum(Bs[:2], LL)
        
        samples[:,0] = np.random.uniform(As[0], Bs[0], n_samples) #Sample the Uniform 
        samples[:,1] = np.random.uniform(As[1], Bs[1], n_samples)
        samples[:,2] = np.random.uniform(As[2], Bs[2], n_samples)
        samples[:,3] = np.random.uniform(As[3], Bs[3], n_samples)
        samples[:,4] = np.random.uniform(As[4], Bs[4], n_samples)
        
        
        # Evaluate the function for all samples
        scores = np.apply_along_axis(func, 1, samples)
        
        idx = np.argmin(scores)
        if scores[idx] < scores_prev:
            scores_prev = scores[idx]
            best_sol = samples[idx]
        
        # Select the elite samples
        elite_indices = np.argsort(scores)[:n_elite]
        elite_samples = samples[elite_indices]
        
        # Update the mean and standard deviation
        mu = np.mean(elite_samples, axis=0)
        sigma = np.std(elite_samples, axis=0)
        
        LOSS.append(np.min(scores) * 1)
        
        
        # Print the progress
        print(f"Iteration {iteration + 1}, Mean: {mu}, Best score: {min(scores)}")
        
        
    LOSS = np.array(LOSS)
    
    return mu, best_sol, LOSS

# Parameters for the CEM
n_samples = 30 #100
elite_frac = 0.15 #0.05
n_iterations = 15#0 #50

mu_init = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Start near (0, 0)
sigma_init = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) 

start = time.time()
# Run the CEM to minimize the Rosenbrock function
best_solution, best_sol, Loss = cross_entropy_method(
    func=optimization,
    n_samples=n_samples,
    elite_frac=elite_frac,
    n_iterations=n_iterations,
    mu_init=mu_init,
    sigma_init=sigma_init
)

stop = time.time()

print(stop - start)

print("Best solution found:", best_solution)
print("Minimum value of the function:", optimization(best_solution))

x = best_sol * 1
frequencies, S11, X_inside, Y_inside, x_boundary, y_boundary = execute_patch_simulation(patch_length_ = x[0], patch_width_ = x[1], h1_ = x[2], h2_ = x[3], w1_ = x[4])
plt.figure(3)
plt.plot(frequencies, target_spectrum, '.-', label = "Target Spectrum")
plt.plot(frequencies, 20 * np.log10(np.abs(S11)), '.-', label = "Output Spectrum")
plt.grid("on")
plt.xlabel("Frequency [Ghz]")
plt.ylabel("S11 [dB]")
plt.legend()


plt.figure(6)
plt.plot(Loss, ".-")
plt.grid("on")
plt.xlabel("Iteration")
plt.ylabel("Loss")
