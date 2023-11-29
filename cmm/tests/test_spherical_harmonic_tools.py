#/---
"""
This script provides a variety of tests related to the spherical harmonic tools
"""
#/---
import pdb, time, scipy.io
import numpy as np
import pyssht as pysh
from ..core import utils
from ..core import mesh_functions as meshes
from ..core import spherical_harmonic_tools as sph_tools

# Testing Functions --------------------------------------------------------------------------
def source(phi,theta, l = 30):
    # rhs of \Delta u = f
    return l*(l+1)*np.sin(theta)**l*np.cos(l*phi) + (l+1)*(l+2)*np.cos(theta)*np.cos(l*phi)*np.sin(theta)**l


def solution(phi, theta, l = 30):
    # coordinate conventions are different for this function
    # in scipy, theta = azimuthal, phi = polar
    return -(np.sin(theta)**l)*np.cos(l*phi) - np.cos(theta)*(np.sin(theta)**l)*np.cos(l*phi)
#-----------------------------------------------------------------------------------------------
# Poisson solver test using the SSHT package
Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]
L_inf = []
for L in Ns:
    
    [Phi, The] = sph_tools.MW_sampling(L)
    F_samples = source(Phi, The)
    F_true = solution(Phi,The)

    start = time.perf_counter()

    f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = True, backend = 'ducc', nthreads = 8)
    f_lm_coeffs = sph_tools.coeff_array(f_lm, L)
    L_flm = sph_tools.lm_array(sph_tools.Poisson_Solve(f_lm_coeffs,L),L)


    f_num = pysh.inverse(L_flm, L, Spin = 0, Method = 'MWSS', Reality = True, backend = 'ducc', nthreads = 8)

    error = np.max(np.absolute(f_num - F_true))
    L_inf.append(error)
    print("time:", time.perf_counter() - start)
    print(error)

#Project onto splines interpolant test --------------------------------------
# import stripy
# def spherical_mesh(phi, theta, N, M):
#     """
#     Inputs: phi, theta - Numpy meshgrids.
#     outputs aa stripy.mesh class based on the meshgrid defined by phi, theta
#     """
#     phis = list(phi.reshape([N*M,]))
#     thetas = list(-theta.reshape([N*M,]) + pi/2)
#
#     #convert to coordinates that stripy enjoys
#
#     #add point at the poles
#     phis.insert(0,0)
#     phis.append(0)
#     thetas.insert(0,pi/2)
#     thetas.append(-pi/2)
#
#     return stripy.sTriangulation(lons = np.array(phis),
#                                  lats = np.array(thetas),
#                                  permute = True)
#
# #
#
# # ## create a fine grid to perform evaluation:
# N_pts = 400
# phi_finer = np.linspace(0, 2*pi, N_pts, endpoint = False)
# theta_finer = np.linspace(0, pi, N_pts, endpoint = True)
# # np.random.seed(303)
# # phi_rand = np.random.uniform(-pi, pi, N_pts)
# # theta_rand = np.random.uniform(0, pi, N_pts)
# # XX_rand = np.meshgrid(phi_rand, the_rand)
# XX2 = np.meshgrid(phi_finer, theta_finer)
# XYZ_0 = sphere2cart(XX2[0], XX2[1])
# eval_pts = np.array([XYZ_0[0].reshape([N_pts**2,]), XYZ_0[1].reshape([N_pts**2,]),
#                      XYZ_0[2].reshape([N_pts**2,])])
#
# u_true = sph_harm(2,10, XX2[0], XX2[1]) #grad_perp_stream(eval_pts[0], eval_pts[1], eval_pts[2])
#
# #
# Ns = [16, 32, 64, 128, 256, 512]
# #
#
# # Vector field projection =====================================================
# l_infx = []
# l_infy = []
# l_infz = []
# areas = []
# for L in [Ns[-1]]:
#     tic = datetime.now()
#     [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
#     [Phi, The] = np.meshgrid(phis, thetas)
#     [X,Y,Z] = sphere2cart(Phi,The)
#     F_samples = stream(Phi, The)
#     f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = False, backend = 'ducc', nthreads = 5)
#     f_lm_coeffs = coeff_array(f_lm, L)
#
#     u_lms = [A for A in angular_momentum(f_lm_coeffs,L)]
#
#     #create a dictionary for the grid
#     N, M = len(phis), len(thetas)
#     XX = np.meshgrid(phis, thetas[1:-1])
#     ico = spherical_mesh(XX[0], XX[1], N, M-2)
#     a = np.max(ico.edge_lengths())
#     areas.append(a)
#
#     simplices, msimplices = full_assembly(len(phis), len(thetas))
#     grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
#                  "msimplices": msimplices, "mesh": ico}
#
#     interp = project_onto_S12_PS(u_lms, L, grid_dict, vector = True, Method = "MWSS")
#     u_num = interp.eval(eval_pts, order = 2, map = False)
#
#     error = np.absolute(np.array(u_num)- np.array(u_true))
#     errors = [np.max(error[0,:]), np.max(error[1,:]), np.max(error[2,:])]
#     l_infx.append(errors[0])
#     l_infy.append(errors[1])
#     l_infz.append(errors[2])
# #
#     print(L, errors)
#     print("time:", datetime.now()-tic)

# import scipy.io
# scipy.io.savemat("../MatlabScripts/tests/spherical_harmonics/angular_momentum_operation.mat", {"u_true": np.array(u_true).real, "u_num": np.array(u_num).real})
#
# l_inf = np.array(l_inf)
# areas = np.array(areas)
# orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(areas[1::]/areas[0:-1])
# print("orders",orders)


# # # scalar function projection ---------------------------------
# areas = []
# l_inf = []
# for L in Ns:
#
#     [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
#     [Phi, The] = np.meshgrid(phis, thetas)
#     s_points = sphere2cart(Phi,The) # the sample points of size (3, L+1, 2*L)
#     F_samples =  sph_harm(2,10, Phi, The) #test_func_pt(Phi, The)
#     f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = False, backend = 'ducc', nthreads = 5)
#     f_lm_coeffs = coeff_array(f_lm, L)
#
#     #create a dictionary for the grid
#     N, M = len(phis), len(thetas)
#     XX = np.meshgrid(phis, thetas[1:-1])
#     ico = spherical_mesh(XX[0], XX[1], N, M-2)
#     a = np.max(ico.edge_lengths())
#     areas.append(a)
#
#     simplices, msimplices = full_assembly(len(phis), len(thetas))
#     grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
#                  "msimplices": msimplices, "mesh": ico, "sample_points": s_points}
#     interp = project_onto_S12_PS(f_lm_coeffs, L, grid_dict, vector = False, Method = "MWSS")
#     u_num = interp.eval(eval_pts, order = 2).reshape([N_pts, N_pts])
#
#     error = np.max(np.absolute(u_num- u_true))
#     l_inf.append(error)
#
#     print(L, error)
#
# l_inf = np.array(l_inf)
# areas = np.array(areas)
# orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(areas[1::]/areas[0:-1])
# print("orders",orders)
#
#

#-------------------------------------------------------------------------------
# # Gradient Test:----------------------------------------------------------------
# Ns = [16, 32, 64, 128, 256]
# L_inf = []
# for L in Ns:
#
#     [The, Phi] = pysh.sample_positions(L, Method = "MWSS", Grid = True)
#     [X,Y,Z] = sphere2cart(Phi,The)
#     F_samples = stream(Phi, The)
#     #u_true = grad_perp_stream(X,Y,Z)
#     u_true = grad_stream(X,Y,Z)
#
#     tic = datetime.now()
#
#     f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = False, backend = 'ducc', nthreads = 5)
#     f_lm_coeffs = coeff_array(f_lm, L)
#     u_lms = [lm_array(A,L) for A in angular_momentum(f_lm_coeffs,L)]
#
#     u_num = []
#     for i in [0,1,2]:
#         u_num.append(pysh.inverse(u_lms[i], L, Spin = 0, Method = 'MWSS', Reality = False, backend = 'ducc', nthreads = 5))
#     # gradient operator
#     u_num = tan_proj(cross(u_num,[X,Y,Z]), [X,Y,Z])
#
#     error = [np.max(np.absolute(u_num[0] - u_true[0])),
#              np.max(np.absolute(u_num[1] - u_true[1])),
#              np.max(np.absolute(u_num[2] - u_true[2]))]
#
#     L_inf.append(error)
#     print("time:", datetime.now() - tic)
#     print(error)
#
# scipy.io.savemat("../MatlabScripts/tests/spherical_harmonics/angular_momentum_operation.mat", {"u_true": np.array(u_true).real, "u_num": np.array(u_num).real})

#--------- Test functions -----------------------------------------------------
# alpha = pi/3
# def stream(phi, theta):
#     return  2*pi*(cos(theta)*cos(alpha) - cos(phi-pi/2)*sin(theta)*sin(alpha))
#     #return sin(phi)**2*(sin(theta)**2)
#     #return sin(theta)*cos(phi)
#
# def test_func_pt(phi, theta):
#     return stream(phi,theta)**10
#
# def test_func(xyz):
#     #
#     x = xyz[:,0]
#     y = xyz[:,1]
#     z = xyz[:,2]
#     r = np.sqrt(x**2 + y**2 + z**2)
#
#     # return (1/r**3)*(1/8)*(np.sqrt(35/pi))*((x-1j*y)**3).real + x**2 #(x**4 - 6*(x**2)*(y**2) + y**4)/(r_2**2)
#     phi, theta = cart2sphere([x,y,z])
#     #pdb.set_trace()
#
#     return stream(phi,theta)**10
#
#
#
# def grad_perp_stream(X,Y,Z):
# #     #SBR
#     phi, theta = cart2sphere([X,Y,Z])
#     # #
#     dtpsi = 2*pi*(-sin(theta)*cos(alpha) - cos(phi-pi/2)*cos(theta)*sin(alpha))
#     dlpsi = 2*pi*(sin(phi-pi/2)*sin(alpha))
#     grad_psi = [-sin(phi)*dlpsi + cos(phi)*cos(theta)*dtpsi,
#                 cos(phi)*dlpsi + sin(phi)*cos(theta)*dtpsi,
#                 -sin(theta)*dtpsi]
#     # #deformation flow
#     # dtpsi = 2*sin(theta)*cos(theta)*sin(phi)**2
#     # dlpsi = 2*sin(phi)*cos(phi)*sin(theta)
#     #
#     # grad_psi = [-sin(phi)*dlpsi + cos(phi)*cos(theta)*dtpsi,
#     #             cos(phi)*dlpsi + sin(phi)*cos(theta)*dtpsi,
#     #             -sin(theta)*dtpsi]
# # # # #     #other tests
# #     dx_psi = sin(phi)**2 + (cos(theta)**2)*cos(phi)**2
# #     dy_psi = -cos(phi)*sin(phi) + sin(phi)*cos(phi)*cos(theta)**2
# #     dz_psi = -sin(theta)*cos(phi)*cos(theta)
# #     grad_psi = [dx_psi, dy_psi, dz_psi]
#
#     return cross([X,Y,Z],grad_psi)
#
# def grad_stream(X,Y,Z):
# #     #SBR
#     phi, theta = cart2sphere([X,Y,Z])
#     # #
#     dtpsi = 2*pi*(-sin(theta)*cos(alpha) - cos(phi-pi/2)*cos(theta)*sin(alpha))
#     dlpsi = 2*pi*(sin(phi-pi/2)*sin(alpha))
#     grad_psi = [-sin(phi)*dlpsi + cos(phi)*cos(theta)*dtpsi,
#                 cos(phi)*dlpsi + sin(phi)*cos(theta)*dtpsi,
#                 -sin(theta)*dtpsi]
#     # #deformation flow
#     # dtpsi = 2*sin(theta)*cos(theta)*sin(phi)**2
#     # dlpsi = 2*sin(phi)*cos(phi)*sin(theta)
#     #
#     # grad_psi = [-sin(phi)*dlpsi + cos(phi)*cos(theta)*dtpsi,
#     #             cos(phi)*dlpsi + sin(phi)*cos(theta)*dtpsi,
#     #             -sin(theta)*dtpsi]
# # # # #     #other tests
# #     dx_psi = sin(phi)**2 + (cos(theta)**2)*cos(phi)**2
# #     dy_psi = -cos(phi)*sin(phi) + sin(phi)*cos(phi)*cos(theta)**2
# #     dz_psi = -sin(theta)*cos(phi)*cos(theta)
# #     grad_psi = [dx_psi, dy_psi, dz_psi]
#
#     return grad_psi


#-------------------------------------------------------------------------------------------
