import numpy as np
from numpy import sin, cos, pi
from InterpT2 import Hermite_T2, hermite_density
from evolution_functions import Advect_Project
import pickle
import pdb

import scipy.io
from datetime import datetime


def identity(phi,theta):
    return np.ones(np.shape(phi))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# define the velocity field

# define intial density to match
#
# def rho_0(x,y):
#     return 1 + 0.01*sin(10*x)*cos(2*y)
#
# #define angle
# angle = np.arccos(39.4783/(4*pi**2))
def mu_0(x,y):
    return 1 + 0.01*cos(x)

def mu_1(x,y):
    return 1 + 0.01*cos(4*y)

#--------------------------------- Setup ---------------------------------------

# finer grid for evaluation
L = 1024
phi_finer = np.linspace(-pi, pi, L, endpoint = False)
theta_finer = np.linspace(-pi, pi, L, endpoint = False)

XX = np.meshgrid(phi_finer, theta_finer)
[Phi, Theta] = XX


Ns = np.array([16, 32, 64])#, 128, 256])#, 512])

# =============== Generic Test =================================================
name = "inexact_matching"

errors = []
areas = []
rho_1 = mu_1(XX[0], XX[1])
rho_0 = mu_0(XX[0], XX[1])

for N in Ns:
    file = open('./data/density_correction_inexact_matching_N%s.txt' %N, "rb")
    interpolant_s = pickle.load(file)
    H_phi = interpolant_s[0]; H_the = interpolant_s[1]
    rho = hermite_density(H_phi, H_the)
    s_pts = [H_phi.eval(XX[0], XX[1]) + XX[0], H_the.eval(XX[0], XX[1]) + XX[1]]

    rho_f = rho(XX[0], XX[1])
    dens_f = rho_f*mu_0(s_pts[0], s_pts[1])

    errors.append(np.max(np.absolute(dens_f-rho_1)))
    # compute Jacobian determinants
    print(errors)
    scipy.io.savemat("./data/density_correction_%s_N%s.mat" %(name,N), {"dens_i": rho_0, "dens_c": rho_1, "dens_f": dens_f, "rho_f": rho_f})

scipy.io.savemat('./data/density_correction_errors_%s.mat' %name, {"ns": Ns, "errors": errors})




#------ Compressible Advection Test---------------------------------------------
#
# file = open('./data/initial_diff.txt', "rb")
# interpolant = pickle.load(file)
#
#
# # compare for the smoothed density
# L = 1000
# [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
# N,M = len(phis), len(thetas)
# #phis range from 0 to 2pi so shift these positions
# [Phi, The] = np.meshgrid(phis, thetas)
# # these are used to define the velocity field.
# s_points = sphere2cart(Phi,The) # the sample points of size (3, L+1, 2*L)
# spts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#         s_points[2].reshape([(L+1)*2*L,])])
#
# s_map_pts = interpolant.eval(spts)
# rho_samples = interpolant.det_jac(spts, np.array(s_map_pts))
#
# #get the initial density
# rho_lms = sph_tools.coeff_array(pysh.forward(rho_samples.reshape([M,N]),
#         L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 5),L)
#
# # angle = np.arccos((np.sqrt(4*pi)*rho_lms[0,0].real)/(4*pi))
#
# N, M = len(phis), len(thetas)
# XX = np.meshgrid(phis, thetas[1:-1])
# s_mesh = spherical_mesh(XX[0], XX[1], N, M-2)
#
# simplices, msimplices = full_assembly(len(phis), len(thetas))
# grid_dict0 = {"phis": phis, "thetas": thetas, "simplices": simplices,
#              "msimplices": msimplices, "mesh": s_mesh, "sample_points": s_points}
#
# rho_smoothed = sph_tools.project_onto_S12_PS(rho_lms, L, grid = grid_dict0)
#
#
# map_pts = interpolant.eval(eval_pts)
# J_i = interpolant.det_jac(eval_pts, np.array(map_pts))
#
# errors = []
# areas = []
# L = 256
# for k in range(7):
#     tres = 2**k + 10
#     s_span = np.linspace(0, 1, tres, endpoint = False)
#     ds = s_span[1]-s_span[0]
#     #
#     ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels= k)
#     edge_length = np.max(ico0.edge_lengths())
#     areas.append(edge_length)
#     file = open('./data/correction_diff_tres_%s_k%s_L%s.txt'%(tres,k,L), "rb")
#     interpolant_s = pickle.load(file)
#
#     s_pts = interpolant_s.eval(eval_pts)
#     map_pts_n = interpolant.eval(np.array(s_pts))
#
#     J_s = interpolant_s.det_jac(eval_pts, np.array(s_pts))
#     J_f = rho_smoothed.eval(np.array(s_pts))
#     # J_f = interpolant.det_jac(np.array(s_pts), np.array(map_pts_n))
#
#     rho = J_s*J_f
#     errors.append(np.max(np.absolute(1-rho)))
#     # compute Jacobian determinants
#     print(errors)
#     scipy.io.savemat("./data/density_correction_k%s.mat" %k, {"dens_i": J_i, "dens_c": J_s, "dens_f": J_f})
#
# scipy.io.savemat('./data/density_correction_errors.mat', {"hs": areas, "errors": errors})


# =============== Euler Test ==================================================

#
# a = 1. #6.37122e6
# u_0 = 2*pi
#
# h_0 = 1.
# # phi_c1, the_c1 = -pi/2, pi/2
# # phi_c2, the_c2 = 0, pi/2
# phi_c1, the_c1 = -pi/6 + pi, -pi/2
# phi_c2, the_c2 = pi/6 + pi, -pi/2
#
# R = 1/2
#
# # ##cosine bell initial condition
# def tracer(xyz):
#     [phi,theta] = cart2sphere(xyz)
#     #return cos(10*phi)*sin(10*theta) - cos(100*phi)*sin(100*theta) + 0.1*cos(1000*phi)*sin(1000*theta)
#     return (sph_harm(10,20, phi, theta) + 0.5*sph_harm(-3,5,phi,theta) + 3*sph_harm(8,9,phi,theta)).real
#
# #advection test
# # Initialization
# name = "RHWave"
#
# file = open('../EulerScripts/data/rhwave/%s_maps_k4_ures_256_T_1.txt' % name, "rb")
# interpolant = pickle.load(file)
#
#
# def vorticity(phi, theta):
#     # # # # Rossby Haurwitz wave
#     return 30*cos(theta)*cos(4*phi)*sin(theta)**4 #+ 4*pi*cos(theta)
#
#
# N_pts = 500
# phi_finer = np.linspace(0, 2*pi, N_pts, endpoint = False)
# theta_finer = np.linspace(0, pi, N_pts, endpoint = True)
#
# XX2 = np.meshgrid(phi_finer, theta_finer)
# XYZ_0 = sphere2cart(XX2[0], XX2[1])
# eval_pts = np.array([XYZ_0[0].reshape([N_pts**2,]), XYZ_0[1].reshape([N_pts**2,]),
#                      XYZ_0[2].reshape([N_pts**2,])])
#
# map_pts = interpolant.eval(eval_pts)
#
# J_i = interpolant.det_jac(eval_pts, np.array(map_pts))
#
# omega_true = vorticity(XX2[0], XX2[1])
#
# angsN = cart2sphere(map_pts)
# angsN = [angsN[0].reshape([N_pts, N_pts]),angsN[1].reshape([N_pts, N_pts])]
# # djac = interpolant.det_jac(q_pts = eval_pts, map_pts = np.array(evals))
# omega_num = vorticity(angsN[0],angsN[1])
#
# print("omega error", np.max(np.absolute(omega_num - omega_true)))
#
# errors = []
# errors_omg = []
#
# areas = []
# L = 256
# for k in range(7):
#     tres = 2**k + 10
#     s_span = np.linspace(0,1, tres, endpoint = False)
#     ds = s_span[1]-s_span[0]
#     #
#     ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels = k)
#     edge_length = np.max(ico0.edge_lengths())
#     areas.append(edge_length)
#     file = open('./data/euler_%s_correction_diff_tres_%s_k%s_L%s.txt'%(name,tres,k,L), "rb")
#     interpolant_s = pickle.load(file)
#
#     s_pts = interpolant_s.eval(eval_pts)
#     map_pts_n = interpolant.eval(np.array(s_pts))
#     J_s = interpolant_s.det_jac(eval_pts, np.array(s_pts))
#     J_f = interpolant.det_jac(np.array(s_pts), np.array(map_pts_n))
#
#     rho = J_s*J_f
#     errors.append(np.max(np.absolute(1-rho)))
#     # compute Jacobian determinants
#     print(errors)
#     scipy.io.savemat("./data/euler_density_correction_k%s.mat" %k, {"dens_i": J_i, "dens_c": J_s, "dens_f": J_f})
#
#     # vorticity errors
#     # with correction
#
#     angsN = cart2sphere(map_pts_n)
#     angsN = [angsN[0].reshape([N_pts, N_pts]),angsN[1].reshape([N_pts, N_pts])]
#     # djac = interpolant.det_jac(q_pts = eval_pts, map_pts = np.array(evals))
#     omega_num_c = vorticity(angsN[0],angsN[1])
#     scipy.io.savemat("./data/omega_correction_k%s.mat" %k, {"omega_i": omega_true, "omega_num": omega_num, "omega_c": omega_num_c})
#
#     errors_omg.append(np.max(np.absolute(omega_true-omega_num_c)))
#     print(errors_omg)
#
#
# scipy.io.savemat('./data/euler_density_correction_errors.mat', {"hs": areas, "errors": errors, "omega_errors": errors_omg})
#
#



# q_1 = tracer(s_pts)
#
# q_i = tracer(interpolant.eval(eval_pts))
#
# q_f = tracer(map_pts_n)
#
#
# #save the two densities
# J_s = interpolant_s.det_jac(eval_pts, np.array(s_pts))
# J_f = interpolant.det_jac(np.array(s_pts), np.array(map_pts_n))
#


# file = open('./data/initial_diff.txt', "rb")
# interpolant = pickle.load(file)


# file2 = open('./data/correction_diff.txt', "rb")
# interpolant_s = pickle.load(file2)

# L = 1000
# [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
# N,M = len(phis), len(thetas)
# #phis range from 0 to 2pi so shift these positions
# [Phi, The] = np.meshgrid(phis, thetas)
# # these are used to define the velocity field.
# s_points = sphere2cart(Phi,The) # the sample points of size (3, L+1, 2*L)
# spts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#         s_points[2].reshape([(L+1)*2*L,])])
#
# s_pts_fine = interpolant_s.eval(spts)
# map_pts_fine = interpolant.eval(spts)
# map_pts_n_fine = interpolant.eval(np.array(s_pts_fine))
#
# #save the two densities
# J_i = interpolant.det_jac(spts, np.array(map_pts_fine))
# J_s = interpolant_s.det_jac(spts, np.array(s_pts_fine))
# J_f = interpolant.det_jac(np.array(s_pts_fine), np.array(map_pts_n_fine))
#
# J_f = J_f*J_s
#
#
# J_i_lm = sph_tools.coeff_array(pysh.forward(J_i.reshape([L+1,2*L]), L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 5),L)
# J_f_lm = sph_tools.coeff_array(pysh.forward(J_f.reshape([L+1,2*L]), L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 5),L)
#
# mean_i = np.sqrt(4*pi)*J_i_lm[0,0].real - 4*pi
#
# mean_f = np.sqrt(4*pi)*J_f_lm[0,0].real - 4*pi
#
# print("initial mass:", mean_i)
# print("final mass:", mean_f)
