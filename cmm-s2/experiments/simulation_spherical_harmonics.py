# ------------------------------------------------------------------------------
"""
Sum of spherical harmonics numerical experiment data generation script
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle, scipy.io
import pyssht as pysh
from ..core.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import spherical_simulations as sphere_sims
from ..core import spherical_harmonic_tools as sph_tools

#--------------------------------- Setup --------------------------------------
k = 6
L = 256 
T = 4
Nt = 1000
n_maps = 20
save_steps = 5


# # to generate the initial vorticity:
# np.random.seed(42) # why 42?
# coeffs = np.random.uniform(-5,5, 20**2)

# # populate first \ell = 20 modes and make them real
# lms0 = np.zeros([20, 41])
# lm = 0
# for ell in range(1,20):
#     for m in range(ell):
#         if m == 0:
#             lms0[ell,ell] = coeffs[lm]
#             lm +=1
#         else:
#             lms0[ell,-m] = coeffs[lm]
#             lms0[ell,m-ell] = coeffs[lm]
#             lm +=1

# L1 = 1024
# lms = np.zeros([L1, 2*L1 + 1], dtype = 'complex128')
# lms[0:20,0:41] = lms0
# mesh_omg = meshes.structure_spherical_triangulation(L = L1)
# # map these into an array for the spherical harmonic coefficients up to large band limit
# vorticity = sph_tools.project_onto_S12(lms, L1, mesh_omg)

# # save this for future use:
# pickle.dump(vorticity, open('./data/intial_vorticity_sph_harm_simulation.txt', "wb"))

vorticity = pickle.load(open('/mnt/c/Users/setht/Research/GitHub_Repos/CMM-S2/data/intial_vorticity_sph_harm_simulation.txt','rb'))

# run the simulation -----------------------------------------------------------------

# load pre-computed mesh
mesh = pickle.load(open('/mnt/c/Users/setht/Research/GitHub_Repos/CMM-S2/cmm-s2/data/icosahedral_mesh_ref_%s.txt' %k, "rb"))

# run and time the simulation:
start, start_clock = time.perf_counter(), time.process_time()

file_name = './data/spherical_harmonics_simulation'

sphere_sims.euler_simulation_sphere_remapping_video(L, Nt, T, mesh, vorticity, 2*np.pi, n_maps, file_name, save_steps)

finish, finish_clock = time.perf_counter(), time.process_time()

print("wall time (s):", finish - start)
print("CPU time (s):", finish_clock - start_clock)

