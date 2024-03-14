# ------------------------------------------------------------------------------
"""
Utility script to process data for video generation
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle
from ..core import mesh_functions as meshes
from ..core import evolution_functions as evol
from ..core import dynamical_fields as vel
from ..core import utils


pi = np.pi

def rotating_frame(phi, theta):
    # this is f
    return 4*pi*np.cos(theta)

#-------------- Simulation Data Parameters--------------------------------------
u_res = 256
map_res = 5
structured = False
t_res = 10000
n_maps = 100 #remapping every 100 time steps

T = 100

name = "rotating_long_term_perturbed_RHwave"


N_pts = 2000

#over whole globe
phi_finer = np.linspace(0, 2*pi, N_pts, endpoint = False)
theta_finer = np.linspace(0, pi, N_pts, endpoint = False)
XX2 = np.meshgrid(phi_finer, theta_finer)
XYZ_0 = utils.sphere2cart(XX2[0], XX2[1])
eval_pts = np.array([XYZ_0[0].reshape([N_pts**2,]), XYZ_0[1].reshape([N_pts**2,]),
                     XYZ_0[2].reshape([N_pts**2,])])

vorticity = vel.perturbed_rossby_wave


direct = "../vortex_dynamics/euler_scripts/data/simulations/random_vorticity/"

file = open('%s/%s_ures%s_tscl_%s_k%s_T%s.txt' %(direct, name, u_res, t_res, map_res, T), "rb")

maps = pickle.load(file)
j = 0
#start threads for parallel evaluation
import multiprocessing
n_threads = 32
pool = multiprocessing.Pool(processes=n_threads)
remaps = []
for map in maps:
    print(j)
    remaps.append(map)
    evals = evol.compose_maps_parallel(remaps, eval_pts, pool, n_threads)
    #advected_quantities: -----------------------------------------
    angs = utils.cart2sphere(evals)
    angels = [angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts])]

    omega_num = vorticity(angels[0], angels[1]) + rotating_frame(angels[0], angels[1]) - rotating_frame(XX2[0], XX2[1])
    scipy.io.savemat("./data/simulations/%s/figures/omega_%s_ures%s_T%s_%s.mat" %(direct, name,u_res,T,j), {"glob": omega_num})
    j +=1


#
# for j in j_list[1::]:
#     print(j)
#     file = open('./data/simulations/%s/%s_ures%s_tscl_%s_k%s_T%s_number_%s.txt' %(direct, name, u_res, t_res, map_res, T, j), "rb")
#     interpolant = pickle.load(file)
#
#     #have to build up list of interpolants every second one is reinitialized
#     if counter % 2 == 0:
#         remaps.append(interpolant)
#         evals = compose_maps(remaps, eval_pts)
#
#     else:
#         evals = compose_maps(remaps + [interpolant], eval_pts)
#
#     counter +=1
#     #advected_quantities: -----------------------------------------
#     #over whole globe
#     # evals = interpolant.eval(eval_pts)
#     angs = cart2sphere(evals)
#     angels = [angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts])]
#     #advected tracers
#     # tracer_global = tracer(angels[0], angels[1])
#     #advected vorticity
#     omega_num = vorticity(angels[0], angels[1]) + rotating_frame(angels[0], angels[1]) #- rotating_frame(XX2[0], XX2[1])
#
#
#     scipy.io.savemat("./data/simulations/%s/figures/omega_%s_ures%s_T%s_%s.mat" %(direct, name,u_res,T,j), {"glob": omega_num})
#     # scipy.io.savemat("./data/simulations/%s/advected_quantities/passive_tracer_%s_ures%s_T%s_%s.mat" %(direct, name,u_res,T,j) , {"glob": tracer_global})
    # ---------------------------------------------
    #
    # #window position:
    # # centred at position of a vortex
    # xyz_0 = sphere2cart(phi = pi/24, theta = pi-pi/8)
    #
    # window_centre = compose_maps(interpolant, np.array([xyz_0, xyz_0, xyz_0]))
    # focal_points.append(window_centre[0,:])
    #
    # [phi_n,theta_n] = cart2sphere(window_centre)
    # phi_n = phi_n[0]
    # theta_n = theta_n[0]
    # # zoom window one
    # phi_w1 = np.linspace(phi_n - 1/2, phi_n + 1/2, N_pts, endpoint = False)
    # theta_w1 = np.linspace(theta_n - 1/2, theta_n + 1/2, N_pts, endpoint = False)
    # XX_w1 = np.meshgrid(phi_w1, theta_w1)
    # XYZ_w1 = sphere2cart(XX_w1[0], XX_w1[1])
    # eval_pts_w1 = np.array([XYZ_w1[0].reshape([N_pts**2,]), XYZ_w1[1].reshape([N_pts**2,]),
    #                      XYZ_w1[2].reshape([N_pts**2,])])
    #
    #
    # phi_w2 = np.linspace(phi_n - 2**-3, phi_n + 2**-3, N_pts, endpoint = False)
    # theta_w2 = np.linspace(theta_n - 2**-3, theta_n + 2**-3, N_pts, endpoint = False)
    # XX_w2 = np.meshgrid(phi_w2, theta_w2)
    # XYZ_w2 = sphere2cart(XX_w2[0], XX_w2[1])
    # eval_pts_w2 = np.array([XYZ_w2[0].reshape([N_pts**2,]), XYZ_w2[1].reshape([N_pts**2,]),
    #                      XYZ_w2[2].reshape([N_pts**2,])])
    #
    # #window one:
    # # evals_w1 = interpolant.eval(eval_pts_w1)
    # evals_w1 = compose_maps(interpolant, eval_pts_w1)
    # angs_w1 = cart2sphere(evals_w1)
    #
    # #advected tracers
    # tracer_w1 = tracer2(angs_w1[0].reshape([N_pts, N_pts]), angs_w1[1].reshape([N_pts, N_pts]))
    # #advected vorticity
    # omega_num_w1 = vorticity(angs_w1[0].reshape([N_pts, N_pts]), angs_w1[1].reshape([N_pts, N_pts]))
    #
    # #window two:
    # # evals_w2 = interpolant.eval(eval_pts_w2)
    # evals_w2 = compose_maps(interpolant, eval_pts_w2)
    # angs_w2 = cart2sphere(evals_w2)
    #
    # #advected tracers
    # tracer_w2 = tracer2(angs_w2[0].reshape([N_pts, N_pts]), angs_w2[1].reshape([N_pts, N_pts]))
    # #advected vorticity
    # omega_num_w2 = vorticity(angs_w2[0].reshape([N_pts, N_pts]), angs_w2[1].reshape([N_pts, N_pts]))
    #
    #
    # scipy.io.savemat("./data/simulations/random_vorticity/advected_quantities/window_omega_%s_ures%s_T%s_%s.mat" %(name,u_res,T,j), {"window1": omega_num_w1 , "window2": omega_num_w2})
    # scipy.io.savemat("./data/simulations/random_vorticity/advected_quantities/window_passive_tracer_%s_ures%s_T%s_%s.mat" %(name,u_res,T,j) , {"window1": tracer_w1 , "window2": tracer_w2})
