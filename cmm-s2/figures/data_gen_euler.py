import numpy as np
#from Remapping import Jacobian, det_Jac, compose
from evolution_functions import compose_maps
from scipy.fftpack import fft2, ifft2, ifftshift, fftshift

#from Initial_Vorticity import vorticity_0
from spherical_spline import full_assembly, spline_interp_structured, spline_interp_vec
import stripy

import spherical_harmonic_tools as sph_tools
import pyssht as pysh


import pickle
import pdb
import threading
from scipy.special import sph_harm
import scipy.io
from datetime import datetime

#-----------------------------------------------------------------------------

"""
What does this script do?

Euler tests using the CM method on a triangulation

"""
#--------------------------------Utilities--------------------------------------

def cos(x):
    return np.cos(x)

def arccos(x):
    return np.arccos(x)

def sin(x):
    return np.sin(x)

def cosh(x):
    return np.cosh(x)

def sinh(x):
    return np.sinh(x)

def tanh(x):
    return np.tanh(x)

def sech(x):
    return 1/np.cosh(x)

def sphere2cart(phi, theta):

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return [x,y,z]

def sphere2cartT2(phi0, theta0):
    phi, theta = phi0.copy(), theta0.copy()
    #N = len(phi[:,0])

    p_u, p_l = phi[np.where(phi >= pi)], phi[np.where(phi < -pi)]
    t_u, t_l = theta[np.where(theta >= pi)], theta[np.where(theta < -pi)]

    p_u = (p_u + pi) % (2*pi) - pi
    t_u = (t_u + pi) % (2*pi) - pi
    p_l = (p_l + pi) % (2*pi) - pi
    t_l = (t_l + pi) % (2*pi) - pi
    #pdb.set_trace()

    phi[np.where(phi >= pi)], phi[np.where(phi < -pi)] = p_u, p_l
    theta[np.where(theta >= pi)], theta[np.where(theta < -pi)] = t_u, t_l

    x = sin(theta)*cos(phi)
    y = sin(theta)*sin(phi)
    z = cos(theta)
    return [x,y,z]

def cart2sphere(XYZ):
    """
    This is modified to lambda \in [0,2pi)
    """
    x, y, z = XYZ[0], XYZ[1], XYZ[2]
    phi = (np.arctan2(y,x) + 2*pi) % (2*pi)
    theta = np.arctan2(np.sqrt(y**2 + x**2), z)
    return [phi, theta]


def cross(a,b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def Mmul(A,B):
    return np.matmul(A,B)

def spherical_mesh(phi, theta, N, M):
    """
    Inputs: phi, theta - Numpy meshgrids.
    outputs aa stripy.mesh class based on the meshgrid defined by phi, theta
    """
    phis = list(phi.reshape([N*M,]))
    thetas = list(-theta.reshape([N*M,]) + pi/2)

    #convert to coordinates that stripy enjoys

    #add point at the poles
    phis.insert(0,0)
    phis.append(0)
    thetas.insert(0,pi/2)
    thetas.append(-pi/2)

    return stripy.sTriangulation(lons = np.array(phis),
                                 lats = np.array(thetas),
                                 permute = True)
def rotate(L, XYZ):
    """
    Apply matrix L to the points XYZ
    """
    outx = L[0,0]*XYZ[0] + L[0,1]*XYZ[1] + L[0,2]*XYZ[2]
    outy = L[1,0]*XYZ[0] + L[1,1]*XYZ[1] + L[1,2]*XYZ[2]
    outz = L[2,0]*XYZ[0] + L[2,1]*XYZ[1] + L[2,2]*XYZ[2]

    return [outx, outy, outz]

def rot_z(alpha):
    return np.array([[cos(alpha), -sin(alpha),0], [sin(alpha), cos(alpha), 0],[0,0,1]])


def tracer(phi,theta):

    #return cos(10*phi)*sin(10*theta) - cos(100*phi)*sin(100*theta) + 0.1*cos(1000*phi)*sin(1000*theta)
    return (sph_harm(10,20, phi, theta) + 0.5*sph_harm(-3,5,phi,theta) + 3*sph_harm(8,9,phi,theta)).real
    #return cos(10*phi)*sin(10*theta)

def tracer2(phi,theta):
    return tracer(phi,theta) + cos(10*phi)*sin(10*theta) # - cos(100*phi)*sin(100*theta) + 0.1*cos(1000*phi)*sin(1000*theta)
    # return cos(50*phi)*sin(50*theta)

np.random.seed(42)
coeffs = np.random.uniform(-1,1, 33**2)
def random_tracer(phi,theta):
    # [phi,theta] = cart2sphere(xyz)/
    ells = np.arange(0,33)
    out = 0
    lm = 0
    for l in ells:
        for m in range(-l,l):
            out += coeffs[lm]*sph_harm(m, l, phi, theta).real
            lm += 1

    return out


pi = np.pi

# #Enstrophy computation
# def enstrophy(grid, vals):
#     """
#     function to compute the enstrophy on a spherical mesh defined by 'grid'
#     vals are the values at the grid points.
#     """
#
#
#     return

# Norms ######---------------------------------------------------------------
#---------------------------------------------------------------------------------

def vorticity_RH(phi, theta):
    # Rossby Haurwitz wave
    return 30*cos(theta)*cos(4*phi)*sin(theta)**4

def vorticity_RH_true(phi, theta, t):
    # # # # Rossby Haurwitz wave
    ell = 5
    C = 1 #(ell*(ell + 1))/((ell*(ell + 1)) -2)
    alpha = 0.5*(2/(ell*(ell+1)))
    Omega = 2*pi
    return 30*cos(theta)*cos(4*(phi + 2*Omega*alpha*t))*sin(theta)**4


alpha = pi/3
R = np.array([[cos(alpha), 0, sin(alpha)],[0,1,0],
              [-sin(alpha), 0, cos(alpha)]])


def vorticity_RH_rot(phi, theta):

    #rotate the points
    [X_r, Y_r, Z_r] = rotate(R, sphere2cart(phi,theta))
    return 30*Z_r*(Y_r**4 + X_r**4 - 6*(X_r**2)*Y_r**2)


def vorticity_rev(phi, theta):
    [x,y,z] = sphere2cart(phi,theta)
    sigma, C = 1, 10
    return C*np.exp(-((x-1)**2 + y**2 + z**2)/sigma)

def vorticity_ZJ(phi, theta):
    #unperturbed Zonal Jet
    beta2 = 12**2
    theta_c = pi/4
    u_lt = (pi/2)*np.exp(-2*beta2*(1-cos(-theta+pi/2-theta_c)))
    return (cos(-theta+pi/2)*(2*beta2*(cos(theta_c)*sin(-theta+pi/2) - sin(theta_c)*cos(-theta+pi/2))) + sin(pi/2 - theta))*u_lt

# def vorticity_perturbed_ZJ(phi,theta):
#     #Perturbed Zonal Jet
#     beta2 = 12**2
#     theta_c = pi/4 + 0.01*cos(12*phi)
#     u_lt = (pi/2)*np.exp(-2*beta2*(1-cos(-theta+pi/2-theta_c)))
#     return cos(-theta+pi/2)*(2*beta2*(cos(theta_c))*sin(-theta+pi/2) - sin(theta_c)*cos(-theta+pi/2))*u_lt

def rotating_frame(phi, theta):
    return 4*pi*cos(theta)

def vorticity_GV(phi, theta):
    [x,y,z] = sphere2cart(phi,theta)
    sigma, C = 1/16, 4*pi
    return C*np.exp(-((x-1)**2 + y**2 + z**2)/sigma)


from Initial_Vorticity import random_vorticity, vorticity_0


#-------------------------------------------------------------------------------

#--------------------------------- Setup ---------------------------------------
def identity(xyz):
    return [xyz[0], xyz[1], xyz[2]]

def identity_x(xyz):
    return xyz[0] #[xyz[0], xyz[1], xyz[2]]

def grad_x(xyz):
    return np.array([0*xyz[0] + 1, 0*xyz[1], 0*xyz[2]])

def grad_y(xyz):
    return np.array([0*xyz[0], 0*xyz[1] + 1, 0*xyz[2]])

def grad_z(xyz):
    return np.array([0*xyz[0], 0*xyz[1], 0*xyz[2] + 1])

# # finer grid for evaluation
N_pts = 1000
phi_finer = np.linspace(0, 2*pi, N_pts, endpoint = False)
theta_finer = np.linspace(0, pi, N_pts, endpoint = True)
dphi = abs(phi_finer[1]-phi_finer[0])
dthe = abs(theta_finer[1]-theta_finer[0])
# np.random.seed(303)

XX2 = np.meshgrid(phi_finer, theta_finer)
XYZ_0 = sphere2cart(XX2[0], XX2[1])
eval_pts = np.array([XYZ_0[0].reshape([N_pts*N_pts,]), XYZ_0[1].reshape([N_pts*N_pts,]),
                     XYZ_0[2].reshape([N_pts*N_pts,])])


#------------------------------------------------------------------------------
#--------------------- Rossby-Haurwitz wave data generation --------------------
# #
# # # # Everything Visualization
# k, Num, T = 2, 64, 0.1
#
# tspan = np.linspace(0, T, 32, endpoint = False)
# dt = tspan[1]-tspan[0]
#
# file = open("data/rhwave/RHWave_rotating_maps_icosahedral_sph_k%s_ures_%s_T_%s.txt" %(int(k),Num,T), "rb")
# interpolant = pickle.load(file)
# evals = interpolant.eval(eval_pts, order = 2)
# angs = cart2sphere(evals)
#
# #advected tracers
# tracer0 = tracer2(XX2[0], XX2[1])
# tracer_T = tracer2(angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts]))
#
# #advected vorticity
# omega_0 = vorticity_RH_true(XX2[0], XX2[1],T)
# omega_num = vorticity_RH(angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts])) + rotating_frame(angs[0].reshape([N_pts, N_pts]),angs[1].reshape([N_pts, N_pts]))  - rotating_frame(XX2[0],XX2[1])
#
# scipy.io.savemat("./data/rhwave/test_RHWave_rotating_passive_tracer_t%s.mat" %T, {"tracer0": tracer0, "tracer_T": tracer_T})
#
# scipy.io.savemat("./data/rhwave/omega_RHWave_rotating_u%s_T_%s.mat" %(Num, T), {"omega_num": omega_num, "omega_0": omega_0})
#
#
# resolutions = [16, 32, 64, 128, 256, 512, 1024]
#
# L_inf = []
# Enst = []
# Energy = []
#
# # form the velocity field grid
# L = 1000
# [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
# XX = np.meshgrid(phis, thetas)
# Method = "MWSS"
#
#
# #True data:
# direct = "rhwave"
# name = "RHWave_rotated_remapped"
# vorticity = vorticity_RH_rot
#
# omega_true = vorticity(XX[0], XX[1])
# omg_T_lms = pysh.forward(omega_true, L, Spin = 0, Method = Method, Reality = False, backend = 'ducc', nthreads = 5)
# enst_true = np.absolute(np.sum(omg_T_lms*omg_T_lms.conjugate()))
# # pdb.set_trace()
# # psi_lms0 = sph_tools.lm_array(sph_tools.Poisson_Solve_Sampled(omega_true, L),L)
# # n_U0 = np.absolute(0.5*np.sum(np.real(omg_T_lms*psi_lms0.conjugate())))
# n_U0 = np.absolute(0.5*np.sum(sph_tools.energy_spectrum(sph_tools.coeff_array(omg_T_lms,L),L)))
#
#
# # these are used to define the velocity field.
# s_points = sphere2cart(XX[0],XX[1]) # the sample points of size (3, L+1, 2*L)
# xyzs = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#         s_points[2].reshape([(L+1)*2*L,])])
# edges = []

# for k in range(6):
#     ico_k = k+1
#     u_res = resolutions[k] #2**(k+1) + 10
#     Num = u_res
#     T = 1
#     tscl = int(resolutions[k]/2)
#     file = open("data/%s/%s_maps_icosahedral_sph_k%s_ures_%s_T_%s.txt" %(direct,name, int(k),Num,T), "rb")
#
#     # This is only for the edge_length
#     ico = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=ico_k)
#     edges.append(np.max(ico.edge_lengths()))
#
#     # without remapping
#     # interpolant = pickle.load(file)
#     # xyz = interpolant.eval(xyzs)
#
#     #if remapped
#     remaps = pickle.load(file)
#     xyz = compose_maps(remaps, xyzs)
#
#
#     angles = cart2sphere(xyz)
#
#     omega_num = vorticity(angles[0].reshape([L+1, 2*L]), angles[1].reshape([L+1, 2*L]))
#     error = np.absolute(omega_num - omega_true)
#
#     #calculate energy error
#     Method = "MWSS"
#     omg_n_lms = pysh.forward(omega_num, L, Spin = 0, Method = Method, Reality = False, backend = 'ducc', nthreads = 5)
#
#     L_inf.append(np.max(error)/np.max(np.absolute(omega_true)))
#     enst_num = np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate()))
#     Enst.append(np.absolute(enst_num-enst_true)/enst_true)
#
#     n_Us = 0.5*np.absolute(np.sum(sph_tools.energy_spectrum(sph_tools.coeff_array(omg_n_lms,L),L)))
#
#     Energy.append(np.absolute(n_U0-n_Us)/n_U0)
#     print("Energy:", Energy)
#     print("Error:", L_inf)
#     print("Enstrophy:", Enst)
#
# edges = np.array(edges)
# l_inf = np.array(L_inf)
# energies = np.array(Energy)
# enstrophies = np.array(Enst)
# orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(edges[1::]/edges[0:-1])
# orders1 = np.log(energies[1::]/energies[0:-1])/np.log(edges[1::]/edges[0:-1])
# orders2 = np.log(enstrophies[1::]/enstrophies[0:-1])/np.log(edges[1::]/edges[0:-1])
#
# print("orders:", orders, orders1, orders2)
#
# print(Energy, Enst, L_inf)
# scipy.io.savemat("./data/errors/%s_tests_tscl_%s_u%s_T%s.mat" %(name, tscl, Num, T), {"linf": L_inf, "energy": Energy, "enstrophy": Enst, "edges": edges})
#


# # rotating simulations =======================================================
#
#
# resolutions = [16, 32, 64, 128, 256, 512, 1024]
#
# L_inf = []
# Enst = []
# Energy = []
#
# # form the velocity field grid
# L = 1000
# [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
# XX = np.meshgrid(phis, thetas)
# Method = "MWSS"
#
#
# #True data:
# direct = "rhwave"
# name = "RHWave_rotating_remapped"
# vorticity = vorticity_RH_true
# T = 1
#
# zeta_true = vorticity(XX[0], XX[1],T)
#
# # total vorticity is conserved
# omg_T_lms = pysh.forward(zeta_true + rotating_frame(XX[0], XX[1]), L, Spin = 0, Method = Method, Reality = False, backend = 'ducc', nthreads = 5)
# enst_true = np.absolute(np.sum(omg_T_lms*omg_T_lms.conjugate()))
# # pdb.set_trace()
# # psi_lms0 = sph_tools.lm_array(sph_tools.Poisson_Solve_Sampled(omega_true, L),L)
# # n_U0 = np.absolute(0.5*np.sum(np.real(omg_T_lms*psi_lms0.conjugate())))
# n_U0 = np.absolute(0.5*np.sum(sph_tools.energy_spectrum(sph_tools.coeff_array(omg_T_lms,L),L)))
#
#
# # these are used to define the velocity field.
# s_points = sphere2cart(XX[0],XX[1]) # the sample points of size (3, L+1, 2*L)
# xyzs = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#         s_points[2].reshape([(L+1)*2*L,])])
# edges = []
# for k in range(6):
#     print(k)
#     ico_k = k+1
#     u_res = resolutions[k] #2**(k+1) + 10
#     Num = u_res
#     tscl = int(resolutions[k]/2)
#
#     #zonal jet naming convention
#     # file = open("data/%s/%s_maps_icosahedral_k%s_ures_%s_tscl_%s_T_%s.txt" %(direct, name, int(k),u_res,tscl,T), "rb")
#
#     # This is only for the edge_length
#     ico = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=ico_k)
#     edges.append(np.max(ico.edge_lengths()))
#
#
#     # # #rhwave naming convention
#     file = open("data/%s/%s_maps_icosahedral_sph_k%s_ures_%s_T_%s.txt" %(direct, name, int(k),u_res,T), "rb")
#
#     #
#     # interpolant = pickle.load(file)
#     # xyz = interpolant.eval(xyzs)
#
#     #if remapped
#     remaps = pickle.load(file)
#     xyz = compose_maps(remaps, xyzs)
#
#     # sample the vorticity at a fine grid.
#     angles = cart2sphere(xyz)
#     angs = [angles[0].reshape([L+1,2*L]), angles[1].reshape([L+1,2*L])]
#
#     zeta_num = vorticity(angs[0], angs[1],0) + rotating_frame(angs[0],angs[1]) - rotating_frame(XX[0],XX[1])
#     omega_num = vorticity(angs[0], angs[1],0) + rotating_frame(angs[0],angs[1])
#     error = np.absolute(zeta_num - zeta_true)
#
#     #calculate energy error
#     Method = "MWSS"
#     omg_n_lms = pysh.forward(omega_num, L, Spin = 0, Method = Method, Reality = False, backend = 'ducc', nthreads = 5)
#
#     L_inf.append(np.max(error)/np.max(np.absolute(zeta_true)))
#     enst_num = np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate()))
#     Enst.append(np.absolute(enst_num-enst_true)/enst_true)
#
#     n_Us = 0.5*np.absolute(np.sum(sph_tools.energy_spectrum(sph_tools.coeff_array(omg_n_lms,L),L)))
#
#     Energy.append(np.absolute(n_U0-n_Us)/n_U0)
#     print("Energy:", Energy)
#     print("Error:", L_inf)
#     print("Enstrophy:", Enst)
#
# edges = np.array(edges)
# l_inf = np.array(L_inf)
# energies = np.array(Energy)
# enstrophies = np.array(Enst)
# orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(edges[1::]/edges[0:-1])
# orders1 = np.log(energies[1::]/energies[0:-1])/np.log(edges[1::]/edges[0:-1])
# orders2 = np.log(enstrophies[1::]/enstrophies[0:-1])/np.log(edges[1::]/edges[0:-1])
#
# print("orders:", orders, orders1, orders2)
#
# print(Energy, Enst, L_inf)
# scipy.io.savemat("./data/errors/%s_tests_tscl_%s_u%s_T%s.mat" %(name, tscl, Num, T), {"linf": L_inf, "energy": Energy, "enstrophy": Enst, "edges": edges})
#
#
#
# #
#

#
# #-----------------------------------------------------------------------------
# #--------------------- Reversing Test Case Data Generation ---------------------
# # # # Everything Visualization
# k, Num, T = 5, 512, 1
# file = open("data/reversing_test/reversing_maps_halfway_k%s_ures_%s_T_%s.txt" %(int(k),Num,T), "rb")
# interpolant = pickle.load(file)
# evals = interpolant.eval(eval_pts, order = 2)
# angs = cart2sphere(evals)
#
# #advected tracers
# tracer_b = tracer(XX2[0], XX2[1])
# tracerd = tracer(angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts]))
#
# #advected vorticity
# omega_0 = vorticity_rev(XX2[0], XX2[1])
# omega_num = vorticity_rev(angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts]))
#
# scipy.io.savemat("./data/reversing_test/test_passive_halfway_tracer_t%s.mat" %T , {"u_a": tracerd, "u_b": tracer_b})
#
# scipy.io.savemat("./data/reversing_test/omega_halfway_u%s_T_%s.mat" %(Num, T) , {"omega_num": omega_num, "omega_0": omega_0})
#
#
# resolutions = [16, 32, 64, 128, 256, 512, 1024]
#
# L_inf = []
# Enst = []
# Energy = []
# l_inf_map = []
# # form the velocity field grid
# L = 1000
# [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
# XX = np.meshgrid(phis, thetas)
# Method = "MWSS"
#
#
# #True data:
# direct = "reversing_test"
# name = "gaussian_vortex_remapped"
# vorticity = vorticity_GV
#
# zeta_true = vorticity(XX[0], XX[1])
#
# # total vorticity is conserved
# omg_T_lms = pysh.forward(zeta_true + rotating_frame(XX[0], XX[1]), L, Spin = 0, Method = Method, Reality = False, backend = 'ducc', nthreads = 5)
# enst_true = np.absolute(np.sum(omg_T_lms*omg_T_lms.conjugate()))
# # pdb.set_trace()
# # psi_lms0 = sph_tools.lm_array(sph_tools.Poisson_Solve_Sampled(omega_true, L),L)
# # n_U0 = np.absolute(0.5*np.sum(np.real(omg_T_lms*psi_lms0.conjugate())))
# n_U0 = np.absolute(0.5*np.sum(sph_tools.energy_spectrum(sph_tools.coeff_array(omg_T_lms,L),L)))
#
#
# # these are used to define the velocity field.
# s_points = sphere2cart(XX[0],XX[1]) # the sample points of size (3, L+1, 2*L)
# xyzs = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#         s_points[2].reshape([(L+1)*2*L,])])
# edges = []
#
# for k in range(6):
#     u_res = resolutions[k] #2**(k+1) + 10
#     Num = u_res
#     T = 1
#     tscl = int(resolutions[k]/2)
#     file = open("data/reversing_test/%s_maps_k%s_ures_%s_T_%s.txt" %(name, int(k),Num,T), "rb")
#     interpolant = pickle.load(file)
#
#     # This is only for the edge_length
#     ico = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=k+1)
#     edges.append(np.max(ico.edge_lengths()))
#
#     # sample the vorticity at a fine grid.
#     # xyz = interpolant.eval(xyzs)
#     xyz = compose_maps(interpolant, xyzs)
#     angles = cart2sphere(xyz.copy())
#
#     angs = [angles[0].reshape([L+1,2*L]), angles[1].reshape([L+1,2*L])]
#
#     zeta_num = vorticity(angs[0], angs[1]) + rotating_frame(angs[0],angs[1]) - rotating_frame(XX[0],XX[1])
#     omega_num = vorticity(angs[0], angs[1]) + rotating_frame(angs[0],angs[1])
#     error = np.absolute(zeta_num - zeta_true)
#
#     #calculate energy error
#     Method = "MWSS"
#     omg_n_lms = pysh.forward(omega_num, L, Spin = 0, Method = Method, Reality = False, backend = 'ducc', nthreads = 5)
#
#     L_inf.append(np.max(error)/np.max(np.absolute(zeta_true)))
#     enst_num = np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate()))
#     Enst.append(np.absolute(enst_num-enst_true)/enst_true)
#
#     n_Us = 0.5*np.absolute(np.sum(sph_tools.energy_spectrum(sph_tools.coeff_array(omg_n_lms,L),L)))
#
#     Energy.append(np.absolute(n_U0-n_Us)/n_U0)
#     # print("map error:", l_inf_map)
#     print("Energy:", Energy)
#     print("Error:", L_inf)
#     print("Enstrophy:", Enst)
#
#
# edges = np.array(edges)
# l_inf = np.array(L_inf)
# energies = np.array(Energy)
# enstrophies = np.array(Enst)
# orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(edges[1::]/edges[0:-1])
# orders1 = np.log(energies[1::]/energies[0:-1])/np.log(edges[1::]/edges[0:-1])
# orders2 = np.log(enstrophies[1::]/enstrophies[0:-1])/np.log(edges[1::]/edges[0:-1])
#
# print("orders:", orders, orders1, orders2)
#
#
# scipy.io.savemat("./data/errors/%s_tests_tscl_%s_u%s_T%s.mat" %(name, tscl, Num, T), {"linf": L_inf, "energy": Energy, "enstrophy": Enst, "edges": edges})
# #
# # #--------------------------------------------------------------------------------------------------------

# # # # # ----------------------- Zonal Jet Test Case ------------------------------------

#advected tracers
# tracer0 = tracer2(XX2[0], XX2[1])
# #advected vorticity
# omega_zj = vorticity_ZJ(XX2[0], XX2[1])
# omega_pzj = vorticity_perturbed_ZJ(XX2[0], XX2[1])
#
# scipy.io.savemat("./data/zonaljet/initial_conditions.mat" , {"tracer":tracer0, "omega_zj":omega_zj, "omega_pzj": omega_pzj})


# #evaluate in a band near the jet
# N_pts = 2000
# phi_finer = np.linspace(0, 2*pi, N_pts, endpoint = False)
# theta_finer = np.linspace(pi/8, pi/4 + pi/8, N_pts, endpoint = True)
# dphi = abs(phi_finer[1]-phi_finer[0])
# dthe = abs(theta_finer[1]-theta_finer[0])
# # np.random.seed(303)
#
# bXX2 = np.meshgrid(phi_finer, theta_finer)
# bXYZ_0 = sphere2cart(bXX2[0], bXX2[1])
# band_pts = np.array([bXYZ_0[0].reshape([N_pts*N_pts,]), bXYZ_0[1].reshape([N_pts*N_pts,]),
#                      bXYZ_0[2].reshape([N_pts*N_pts,])])
#
# #-----------------------------
# k, u_res, t_res, T = 6, 200, 640, 0.5
# # file = open("./data/zonaljet/ZJ_maps_sph_k%s_ures_%s_tscl_%s_T_%s.txt" %(int(k),u_res,t_res,T), "rb")
#
# file = open("./data/simulations/unperturbed_zonal_jet_ures%s_tscl_%s_k%s_T%s_number_635.txt" %(u_res,t_res,int(k),T), "rb")
#
#
# interpolant = pickle.load(file)
# evals = interpolant.eval(band_pts, order = 2)
# # remaps = pickle.load(file)
# # evals = compose_maps(remaps, band_pts)
#
# angs = cart2sphere(evals)
#
# #advected tracers
# # tracer0 = tracer2(XX2[0], XX2[1])
# tracer_T = tracer2(angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts]))
#
# #advected vorticity
# # omega_0 = vorticity_perturbed_ZJ(bXX2[0], bXX2[1])
# omega_num = vorticity_ZJ(angs[0].reshape([N_pts, N_pts]), angs[1].reshape([N_pts, N_pts]))
#
# # scipy.io.savemat("./data/errors/sim_omega_pZJ_band_IC.mat", {"u": omega_0})
# scipy.io.savemat("./data/errors/sim_omega_ZJ_band_k%s_u_%s_T_%s.mat" %(k, u_res, T), {"u": omega_num})
# scipy.io.savemat("./data/errors/sim_test_ZJ_band_passive_tracer_T%s.mat" %T , {"tracer_T": tracer_T})

#
L_inf = []
Enst = []
Energy = []
edges = []
resolutions = [16, 32, 64, 128, 256, 512, 1024]

L = 1000
[thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
XX = np.meshgrid(phis, thetas)
# these are used to define the velocity field.
s_points = sphere2cart(XX[0],XX[1]) # the sample points of size (3, L+1, 2*L)
xyzs = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
        s_points[2].reshape([(L+1)*2*L,])])

omega_true = vorticity_ZJ(XX[0], XX[1]) + rotating_frame(XX[0], XX[1])
omg_T_lms = pysh.forward(omega_true, L, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
enst_true = np.absolute(np.sum(omg_T_lms*omg_T_lms.conjugate()))
psi_lms0 = sph_tools.lm_array(sph_tools.Poisson_Solve_Sampled(omega_true, L),L)
n_U0 = np.absolute(0.5*np.sum(np.real(omg_T_lms*psi_lms0.conjugate())))

#
T = 0.5
for k in range(6):
    u_res = resolutions[k]
    Num = u_res
    tscl = int(resolutions[k]/2)

    file = open("data/zonaljet/ZJ_rotating_maps_icosahedral_k%s_ures_%s_tscl_%s_T_%s.txt" %(int(k),u_res,tscl,T), "rb")
    interpolant = pickle.load(file)
    xyz = interpolant.eval(xyzs)
    #
    # file = open("data/zonaljet/ZJ_rotating_remapped_maps_icosahedral_sph_k%s_ures_%s_T_%s.txt" %(int(k),u_res,T), "rb")
    # remaps = pickle.load(file)
    # xyz = compose_maps(remaps, xyzs)

    # the corresponding grid
    ico = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=k+1)
    edges.append(np.max(ico.edge_lengths()))

    # sample the vorticity at a fine grid.
    angles = cart2sphere(xyz)
    omega_num = vorticity_ZJ(angles[0].reshape([L+1, 2*L]), angles[1].reshape([L+1, 2*L])) \
             + rotating_frame(angles[0].reshape([L+1, 2*L]), angles[1].reshape([L+1, 2*L]))
    error = np.absolute(omega_num - omega_true)

    #calculate energy error
    Method = "MWSS"
    omg_n_lms = pysh.forward(omega_num, L, Spin = 0, Method = Method, Reality = False, backend = 'ducc', nthreads = 5)

    L_inf.append(np.max(error)/np.max(np.absolute(omega_true)))
    enst_num = np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate()))
    Enst.append(np.absolute(enst_num-enst_true)/enst_true)

    psi_lms_n = sph_tools.lm_array(sph_tools.Poisson_Solve_Sampled(omega_num, L),L)

    n_Us = 0.5*np.absolute(np.sum(omg_n_lms*psi_lms_n.conjugate()))

    Energy.append(np.absolute(n_U0-n_Us)/n_U0)
    print("Energy:", Energy)
    print("Error:", L_inf)
    print("Enstrophy:", Enst)

edges = np.array(edges)
l_inf = np.array(L_inf)
energies = np.array(Energy)
enstrophies = np.array(Enst)
orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(edges[1::]/edges[0:-1])
orders1 = np.log(energies[1::]/energies[0:-1])/np.log(edges[1::]/edges[0:-1])
orders2 = np.log(enstrophies[1::]/enstrophies[0:-1])/np.log(edges[1::]/edges[0:-1])

print("orders:", orders, orders1, orders2)

print(Energy, Enst, L_inf)
scipy.io.savemat("./data/zonaljet/errors_ZJ_tscl_%s_u%s_T%s.mat" %(tscl, Num, T), {"linf": L_inf, "energy": Energy, "enstrophy": Enst, "edges": edges})
