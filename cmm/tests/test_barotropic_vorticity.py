# ------------------------------------------------------------------------------
"""
Basic script to test the vorticity equation solver on the sphere
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle
import pyssht as pysh
from ..core.interpolants.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as vel
from ..core.spherical_simulations import euler_simulation_rotating_sphere_remapping, euler_simulation_rotating_sphere

#--------------------------------- Setup --------------------------------------
# define grid to evaluate the error
N_pts = 500
N_pts = N_pts
phis = np.linspace(0, 2*np.pi, N_pts, endpoint = False)
thetas = np.linspace(0, np.pi, N_pts, endpoint = False)
XX = np.meshgrid(phis, thetas)

name = "zonal_jet"
vorticity = vel.zonal_jet

if name == "rossby_wave":
    omega_true = vorticity(XX[0], XX[1], t = 1)

else:
    omega_true = vorticity(XX[0], XX[1])

# these are used to define the velocity field.
s_points = utils.sphere2cart(XX[0],XX[1]) # the sample points of size (3, L+1, 2*L)
eval_pts = np.array([s_points[0].reshape([N_pts*N_pts,]), s_points[1].reshape([N_pts*N_pts,]),
        s_points[2].reshape([N_pts*N_pts,])])

# omg_T_lms = pysh.forward(omega_true, N_pts, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
# enst0 = np.absolute(np.sum(omg_T_lms*omg_T_lms.conjugate()))

tri_size = []
l_inf = []
l_inf_map = []
error_over_time = []
Enst = []

resolutions = [16, 32, 64, 128, 256, 512, 1024]

remapping = False

for k in range(7):
    ico_k = k+1
    L = 256 #resolutions[k] #2**(k+1) + 10
    T = 1
    t_res = L//2
    n_maps = 20
    #define mesh:
    ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=ico_k)
    mesh = meshes.spherical_triangulation(ico0.points)
    tri_size.append(np.max(ico0.edge_lengths()))

    # run and time the simulation:
    start, start_clock = time.perf_counter(), time.process_time()

    if remapping:
        maps = euler_simulation_rotating_sphere_remapping(L, t_res, T, n_maps, mesh, vorticity)

    else:
        maps = euler_simulation_rotating_sphere(L, t_res, T, mesh, vorticity)

    finish, finish_clock = time.perf_counter(), time.process_time()

    print("wall time (s):", finish - start)
    print("CPU time (s):", finish_clock - start_clock)

    if remapping:
        evals = evol.compose_maps(maps, eval_pts)
    else:
        evals = maps(eval_pts)

    angs = utils.cart2sphere(evals)
    angs = [angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts])]

    omega_num = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0],angs[1]) - vel.rotating_frame(XX[0],XX[1])

    # omg_n_lms = pysh.forward(omega_num, N_pts, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
    # enst_error = np.absolute(np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate())) - enst0)
    # Enst.append(enst_error/enst0)
    # print("Enstrophy Error:", Enst)

    l_inf_k = np.max(np.absolute(omega_true - omega_num))
    l_inf.append(l_inf_k)
    print("L-inf error:", l_inf_k)

    #save the data:
    if remapping:
        file = open('./data/convergence_test_%s_%s_remapping.txt' %(name, L), "wb")
        pickle.dump(maps, file)

    else:
        file = open('./data/convergence_test_%s_%s.txt' %(name, L), "wb")
        pickle.dump(maps, file)