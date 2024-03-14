#/ -----
"""
Searching for a faster spherical harmonic transform implementation .....
"""
#/ -----
import pdb, time, scipy.io
import numpy as np
import pyssht as pysh
from ..core import utils
from ..core import mesh_functions as meshes
from ..core import spherical_harmonic_tools as sph_tools
import torch
import torch_harmonics as th


L = 256

## -------------------------------------
def source(phi,theta, l = 30):
    # rhs of \Delta u = f
    return l*(l+1)*np.sin(theta)**l*np.cos(l*phi) + (l+1)*(l+2)*np.cos(theta)*np.cos(l*phi)*np.sin(theta)**l


def solution(phi, theta, l = 30):
    # coordinate conventions are different for this function
    # in scipy, theta = azimuthal, phi = polar
    return -(np.sin(theta)**l)*np.cos(l*phi) - np.cos(theta)*(np.sin(theta)**l)*np.cos(l*phi)
# -------------------------------------
[Phi, The] = sph_tools.MW_sampling(L)

F_samples = source(Phi, The)
F_true = solution(Phi,The)

start, start_clock = time.perf_counter(), time.process_time()

f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = True, backend = 'ducc', nthreads = 8)
# f_lm_coeffs = sph_tools.coeff_array(f_lm, L)
# L_flm = sph_tools.lm_array(sph_tools.Poisson_Solve(f_lm_coeffs,L),L)

f_num = pysh.inverse(f_lm, L, Spin = 0, Method = 'MWSS', Reality = True, backend = 'ducc', nthreads = 8)

finish, finish_clock = time.perf_counter(), time.process_time()

print("wall time (s):", finish - start)
print("CPU time (s):", finish_clock - start_clock)


# # # s2fft test:
# from jax.config import config
# config.update("jax_enable_x64", True)
# import s2fft 

# [Phi, The] = sph_tools.MW_sampling(L)

# F_samples = source(Phi, The)
# F_true = solution(Phi,The)

# precomps = s2fft.generate_precomputes_jax(L, forward=True, sampling = "mwss")
# precomps_inv = s2fft.generate_precomputes_jax(L, forward=False, sampling = "mwss")

# start, start_clock = time.perf_counter(), time.process_time()
# flm_pre = s2fft.forward_jax(F_samples, L, reality=True, precomps = precomps, sampling = "mwss")

# f_recov_pre = s2fft.inverse_jax(flm_pre, L, reality=True, precomps = precomps_inv, sampling = "mwss")

# finish, finish_clock = time.perf_counter(), time.process_time()

# print("wall time (s):", finish - start)
# print("CPU time (s):", finish_clock - start_clock)


# nlat = L
# nlon = 2*nlat
# batch_size = 32
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# signal = torch.randn(batch_size, nlat, nlon).to(device)

# # transform data on an equiangular grid

# sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)

# start, start_clock = time.perf_counter(), time.process_time()

# coeffs = sht(signal)

# finish, finish_clock = time.perf_counter(), time.process_time()

# print("wall time (s):", finish - start)
# print("CPU time (s):", finish_clock - start_clock)
