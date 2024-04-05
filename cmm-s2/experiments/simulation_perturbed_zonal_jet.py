# ------------------------------------------------------------------------------
"""
Two two-stream instability numerical experiment data generation script
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle, scipy.io
import pyssht as pysh
from ..core.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as vel
from ..core import spherical_simulations as sphere_sims

#--------------------------------- Setup --------------------------------------
vorticity = vel.multi_jet

k = 5
L = 256 
T = 10
Nt = 1000
n_maps = 10
save_steps = 5

# run the simulation -----------------------------------------------------------------
path_to_data = ''

# load pre-computed mesh
mesh = pickle.load(open(path_to_data + '/data/icosahedral_mesh_ref_%s.txt' %k, "rb"))

# run and time the simulation:
start, start_clock = time.perf_counter(), time.process_time()

file_name = './data/multi_jet_simulation'

sphere_sims.euler_simulation_sphere_remapping_video(L, Nt, T, mesh, vorticity, 2*np.pi, n_maps, file_name, save_steps)

finish, finish_clock = time.perf_counter(), time.process_time()

print("wall time (s):", finish - start)
print("CPU time (s):", finish_clock - start_clock)
