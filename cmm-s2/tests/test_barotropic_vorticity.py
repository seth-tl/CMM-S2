# ------------------------------------------------------------------------------
"""
Scripts to produce all the convergence tests for barotropic vorticity solver
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle, sys
import pyssht as pysh
from ..core.interpolants.spherical_spline import sphere_diffeomorphism
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
        s_points[2].reshape([N_pts*N_pts,])])


name = sys.argv[1]
rotating = True

if name == "zonal_jet":
    vorticity = vel.zonal_jet
    omega_true = vorticity(XX[0], XX[1])
    T = 0.5

if name == "rossby_wave":
    vorticity = vel.rossby_wave
    omega_true = vorticity(XX[0], XX[1], t = 1)
    T = 1

if name == "rossby_wave_static":
    vorticity = vel.rossby_wave
    rotating = False
    omega_true = vorticity(XX[0], XX[1], t = 0)
    T = 1

if name == "gaussian_vortex":
    vorticity = vel.gaussian_vortex
    omega_true = vorticity(XX[0], XX[1])  # only to compare for conservation error
    T = 1

remapping = False

if sys.argv[2] == "remapping":
    remapping = True


tri_size = []
l_inf = []


for k in range(7):
    ico_k = k+1
    L = 128 #resolutions[k] #2**(k+1) + 10
    t_res = 2**(k+2) 
    n_maps = 20

    #define mesh:
    ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=ico_k)
    mesh = meshes.spherical_triangulation(ico0.points)
    tri_size.append(np.max(ico0.edge_lengths()))

    # run and time the simulation:
    start, start_clock = time.perf_counter(), time.process_time()

    if remapping and rotating:
        maps = sphere_sims.euler_simulation_rotating_sphere_remapping(L, t_res, T, n_maps, mesh, vorticity)

    if remapping and not rotating:
        maps = sphere_sims.euler_simulation_static_sphere_remapping(L, t_res, T, n_maps, mesh, vorticity)

    if rotating and not remapping:
        maps = sphere_sims.euler_simulation_rotating_sphere(L, t_res, T, mesh, vorticity)

    if not rotating and not remapping:
        maps = sphere_sims.euler_simulation_static_sphere(L, t_res, T, mesh, vorticity)


    finish, finish_clock = time.perf_counter(), time.process_time()

    print("wall time (s):", finish - start)
    print("CPU time (s):", finish_clock - start_clock)

    if remapping:
        evals = evol.compose_maps(maps, eval_pts)
    else:
        evals = maps(eval_pts)

    angs = utils.cart2sphere(evals)
    angs = [angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts])]

    if rotating:
        omega_num = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0],angs[1]) - vel.rotating_frame(XX[0],XX[1])
    else:
        omega_num = vorticity(angs[0], angs[1]) 

    # compute the error as we go.-----------------
    l_inf_k = np.max(np.absolute(omega_true - omega_num))
    l_inf.append(l_inf_k)
    print("L-inf error:", l_inf_k)
    # -------------------------------------------

    #save the data:
    if remapping:
        file = open('./data/convergence_test_%s_%s_remapping.txt' %(name, ico_k), "wb")
        pickle.dump(maps, file)

    else:
        file = open('./data/convergence_test_%s_%s.txt' %(name, ico_k), "wb")
        pickle.dump(maps, file)