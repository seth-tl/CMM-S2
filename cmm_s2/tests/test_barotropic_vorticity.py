# ------------------------------------------------------------------------------
"""
Scripts to produce all the convergence tests for barotropic vorticity solver
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle, sys
import pyssht as pysh
from ..core.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as vel
from ..core import spherical_simulations as sphere_sims

#--------------------------------- Setup --------------------------------------

# define grid to evaluate the error as we compute (not plotted results)
N_pts = 500
N_pts = N_pts
phis = np.linspace(0, 2*np.pi, N_pts, endpoint = False)
thetas = np.linspace(0, np.pi, N_pts, endpoint = False)
XX = np.meshgrid(phis, thetas)
s_points = utils.sphere2cart(XX[0],XX[1]) 
eval_pts = np.array([s_points[0].reshape([N_pts*N_pts,]), s_points[1].reshape([N_pts*N_pts,]),
        s_points[2].reshape([N_pts*N_pts,])]).T


name = sys.argv[1]

path_to_data = ''

T = 1

if name == "zonal_jet":
    vorticity = vel.zonal_jet
    omega_true = vorticity(XX[0], XX[1])
    rot_rate = 2*np.pi

if name == "rossby_wave":
    vorticity = vel.rossby_wave
    omega_true = vorticity(XX[0], XX[1], t = 1)
    rot_rate = 2*np.pi

if name == "rossby_wave_static":
    vorticity = vel.rossby_wave
    omega_true = vorticity(XX[0], XX[1], t = 0)
    rot_rate = 0


if name == "gaussian_vortex":
    vorticity = vel.gaussian_vortex
    omega_true = vorticity(XX[0], XX[1])  # only to compare for conservation error
    rot_rate = 2*np.pi

remapping = False
if sys.argv[2] == "remapping":
    remapping = True

for k in range(8):
    L = 128
    t_res = 2**(k+1) + 10 

    #define mesh:
    # load pre-computed mesh
    mesh = pickle.load(open(path_to_data + '/data/icosahedral_mesh_ref_%s.txt' %k, "rb"))

    # run and time the simulation:
    start, start_clock = time.perf_counter(), time.process_time()

    if remapping:
        maps = sphere_sims.euler_simulation_sphere_remapping(L, t_res, T, mesh, vorticity, rot_rate = rot_rate, n_maps = 10)
    else:
        maps = sphere_sims.euler_simulation_sphere(L, t_res, T, mesh, vorticity, rot_rate = rot_rate)

    finish, finish_clock = time.perf_counter(), time.process_time()

    print("wall time (s):", finish - start)
    print("CPU time (s):", finish_clock - start_clock)

    evals = maps(eval_pts).T

    angs = utils.cart2sphere(evals)
    angs = [angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts])]

    omega_num = vorticity(angs[0], angs[1]) + 2*rot_rate*np.cos(angs[1]) - 2*rot_rate*np.cos(XX[1])

    # compute the error as we go.-----------------
    l_inf_k = np.max(np.absolute(omega_true - omega_num))
    print("L-inf error:", l_inf_k)
    # -------------------------------------------
    # #save the data:
    if remapping:
        file = open('./data/convergence_test_%s_%s_remapping.txt' %(name, k), "wb")
        pickle.dump(maps, file)

    else:
        file = open('./data/convergence_test_%s_%s.txt' %(name, k), "wb")
        pickle.dump(maps, file)