# ------------------------------------------------------------------------------
"""
Basic script to test the vorticity equation solver on the sphere
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle, scipy.io
import pyssht as pysh
from ..core.interpolants.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as vel
from ..core import spherical_simulations as sphere_sims

#--------------------------------- Setup --------------------------------------

name = "/mnt/c/Users/setht/Research/Code/cmm/data/BVE_paper_data/perturbed_zonal_jet_remapped"
vorticity = vel.perturbed_zonal_jet


k = 4
L = 128 #resolutions[k] #2**(k+1) + 10
T = 4
t_res = 1000
n_maps = 20
save_steps = 5

# run the simulation -----------------------------------------------------------------

# params = {"L":L, "T":T, "k": k, "tres": t_res, "save_steps": save_steps, "n_maps": n_maps}

# #define mesh:
# ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels= k)
# mesh = meshes.spherical_triangulation(ico0.points)

# # run and time the simulation:
# start, start_clock = time.perf_counter(), time.process_time()


# sphere_sims.euler_simulation_video_rotating_sphere_remapping(L, t_res, T, n_maps, mesh, vorticity, save_steps, name, params)

# finish, finish_clock = time.perf_counter(), time.process_time()

# print("wall time (s):", finish - start)
# print("CPU time (s):", finish_clock - start_clock)



# post-processing the data ==========================================================================
    
# define grid to evaluate the error
N_pts = 500
N_pts = N_pts
phis = np.linspace(0, 2*np.pi, N_pts, endpoint = False)
thetas = np.linspace(0, np.pi, N_pts, endpoint = False)
XX = np.meshgrid(phis, thetas)

# these are used to define the velocity field.
s_points = utils.sphere2cart(XX[0],XX[1]) # the sample points of size (3, L+1, 2*L)
eval_pts = np.array([s_points[0].reshape([N_pts*N_pts,]), 
                     s_points[1].reshape([N_pts*N_pts,]),
                     s_points[2].reshape([N_pts*N_pts,])])

remapping = True

if not remapping:
    # load the simulation data:
    params, maps = pickle.load(open(name + "_maps" + ".txt", "rb"))

    omegas = np.zeros([len(maps), N_pts, N_pts])

    i = 0
    for map in maps:
        evals = map(eval_pts)

        angs = utils.cart2sphere(evals)
        angs = [angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts])]

        omega_num = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0],angs[1]) - vel.rotating_frame(XX[0],XX[1])
        omegas[i] = omega_num
        i +=1 


    # save the vorticity data for plotting:
    scipy.io.savemat(name + "_vorticities.mat", {"omegas": omegas})

else:
    # load the simulation data:
    Nt = t_res//save_steps
    omegas = np.zeros([Nt, N_pts, N_pts])

    for i in range(Nt):
        params, maps = pickle.load(open(name + "_maps_step%s" % (i+1) + ".txt", "rb"))

        evals = evol.compose_maps(maps,eval_pts)

        angs = utils.cart2sphere(evals)
        angs = [angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts])]

        omega_num = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0],angs[1]) - vel.rotating_frame(XX[0],XX[1])
        omegas[i] = omega_num

    # save the vorticity data for plotting:
    scipy.io.savemat(name + "_vorticities.mat", {"omegas": omegas})

# omg_n_lms = pysh.forward(omega_num, N_pts, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
# enst_error = np.absolute(np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate())) - enst0)
# Enst.append(enst_error/enst0)
# print("Enstrophy Error:", Enst)