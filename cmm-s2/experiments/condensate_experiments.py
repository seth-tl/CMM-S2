# ------------------------------------------------------------------------------
"""
Condensate formation numerical experiment data generation script
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle, scipy.io
from ..core import utils
from ..core import dynamical_fields as vel
from ..core import spherical_simulations as sphere_sims

#--------------------------------- Setup --------------------------------------
vorticity = vel.perturbed_rossby_wave

k = 5
L = 256 
T = 100
Nt = 10000
n_maps = 100
save_steps = 50

rot_rate = 2*np.pi
file_name = './data/condensate_experiment_rotating'

# run the simulation -----------------------------------------------------------------

path_to_data = ''

# load pre-computed mesh
mesh = pickle.load(open(path_to_data + 'data/icosahedral_mesh_ref_%s.txt' %k, "rb"))

# run and time the simulation:
start, start_clock = time.perf_counter(), time.process_time()

sphere_sims.euler_simulation_sphere_remapping_video(L, Nt, T, mesh, vorticity, rot_rate, n_maps, file_name, save_steps)

finish, finish_clock = time.perf_counter(), time.process_time()

print("wall time (s):", finish - start)
print("CPU time (s):", finish_clock - start_clock)
