#/---------
"""
four modes incompressible Euler test on toroidal geometry
"""
#/---------

import numpy as np
from cmm.core import evolution_functions as evol
from cmm.core import periodic_simulations as sims
from cmm.core import spectral_tools as fspace
from scipy.fft import fftshift, ifftshift, fft2, ifft2

import pickle, pdb, time, scipy.io

# simulation definition
k = 2.
T = 1

name = "four_modes"
def omega_0(X):
    return np.cos(X[0]) + np.cos(X[1]) + 0.6*np.cos(2*X[0]) + 0.2*np.cos(3*X[0])

def tracer(x,y):
    return np.cos(x*10)*np.sin(y*7)


# fine mesh to evaluate the error
phi_finer = np.linspace(0, 2*np.pi, 1024, endpoint = False)
theta_finer = np.linspace(0, 2*np.pi, 1024, endpoint = False)
XX = np.meshgrid(phi_finer, theta_finer)


Ns = np.array([16, 32, 64, 128, 256, 512])

L_inf = []
l_2 = []

omg_k0 = fspace.energy(fft2(omega_0(XX)))


l_inf = []
l_2 = []

for N in Ns:

    start, start_clock = time.perf_counter(), time.process_time()
    inv_map_n = sims.euler_simulation_T2(L = 256, Nt = 2*N, T = T, Nx = N,
                                         vorticity = omega_0)

    finish, finish_clock = time.perf_counter(), time.process_time()

    print("wall time (s):", finish - start)
    print("CPU time (s):", finish_clock - start_clock)

    omg_n_fine = omega_0(inv_map_n(XX))
    omg_k = fspace.energy(fft2(omg_n_fine))

    error = np.absolute(omg_k0 - omg_k) #/np.sum(u_k0)

    l_2.append(error)
    print(l_2)




# l2 = np.array(l_2)
#
# orders = np.log(l2[1::]/l2[0::-1])/np.log(Ns[1::]/Ns[0::-1])
