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
from ..core import dynamical_fields as vel
from ..core.interpolants.spherical_spline import spline_interp_vec, spline_interp_structured
from ..core import spherical_harmonic_tools as sph_tools
from ..core import spherical_simulations as sphere_sims

# Testing Functions --------------------------------------------------------------------------
def source(phi, theta, l = 8):
    # rhs of \Delta u = f
    return (l*(l+1) + (l+1)*(l+2)*np.cos(theta))*np.cos(l*phi)*np.sin(theta)**l

def solution(phi, theta, l = 8):
    # coordinate conventions are different for this function
    # in scipy, theta = azimuthal, phi = polar
    return -(np.sin(theta)**l)*np.cos(l*phi) - np.cos(theta)*(np.sin(theta)**l)*np.cos(l*phi)

# source_dphi = -l*(l*(l+1) + (l+1)*(l+2)*np.cos(theta))*np.sin(theta)**l*np.sin(l*phi) 
# source_dthe = -(l+1)*(l+2)*sin(theta)**(l+1)*cos(l*phi) + \
#             (l*(l+1) + (l+1)*(l+2)*np.cos(theta))*l*np.sin(theta)**(l-1)*np.cos(theta)**np.cos(l*phi)

def grad_source(phi, theta, l= 8):
    # gradient on sphere in spherical coordinates:
    grad_x = (l*(l+1) + (l+1)*(l+2)*np.cos(theta))*l*np.sin(theta)**(l-1)*np.sin(l*phi)*np.sin(phi) \
            + np.cos(phi)*np.cos(theta)*(-(l+1)*(l+2)*np.sin(theta)**(l+1)*np.cos(l*phi) \
            + (l*(l+1) + (l+1)*(l+2)*np.cos(theta))*l*np.sin(theta)**(l-1)*np.cos(theta)*np.cos(l*phi))
    
    grad_y = -(l*(l+1) + (l+1)*(l+2)*np.cos(theta))*l*np.sin(theta)**(l-1)*np.sin(l*phi)*np.cos(phi) \
            + np.sin(phi)*np.cos(theta)*(-(l+1)*(l+2)*np.sin(theta)**(l+1)*np.cos(l*phi) + \
             (l*(l+1) + (l+1)*(l+2)*np.cos(theta))*l*np.sin(theta)**(l-1)*np.cos(theta)*np.cos(l*phi))
    
    grad_z = -np.sin(theta)*(-(l+1)*(l+2)*np.sin(theta)**(l+1)*np.cos(l*phi) + \
             (l*(l+1) + (l+1)*(l+2)*np.cos(theta))*l*np.sin(theta)**(l-1)*np.cos(theta)*np.cos(l*phi))

    return np.array([grad_x, grad_y, grad_z])
#-----------------------------------------------------------------------------------------------
# Poisson solver test using the SSHT package
Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]

def source(phi,theta):
    return vel.rossby_wave(phi,theta, t= 0)

def solution(phi,theta):
    x,y,z = utils.sphere2cart(phi,theta)

    return -z*((1-z**2)**2 -8*(1-z**2)*x**2 + 8*x**4)
    #return -np.cos(theta)*np.cos(4*phi)*np.sin(theta)**4

def velocity(phi,theta, L):
    x,y,z = np.array(utils.sphere2cart(phi,theta)).reshape(3,2*L*(L+1))
    psi_x = -32*z*x**3 + 16*z*(1-z**2)*x
    psi_z = -(1-z**2)**2 + 4*z**2*(1-z**2) + 8*x**2*(1-3*z**2) - 8*x**4
    return np.array([y*psi_z, z*psi_x - x*psi_z, -y*psi_x])

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


    #project velocity field
    [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
    [Phi, The] = np.meshgrid(phis, thetas)
    s_points = utils.sphere2cart(Phi,The)
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])])
    #obtain stream function
    psi0 = source(Phi, The)
    psi_lms = sph_tools.inv_Laplacian(psi0, L)

    #create a dictionary for the grid
    N, M = len(phis), len(thetas)
    XX = np.meshgrid(phis, thetas[1:-1])
    ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

    simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
    grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
                 "msimplices": msimplices, "mesh": ico, "sample_points": s_points}

    u0 = sph_tools.project_stream(psi_lms, L, grid = grid_dict, Method = "MWSS")
    error_u = np.max(np.absolute(u0.eval(sample_pts) - velocity(Phi,The,L)))

    error = np.max(np.absolute(f_num - F_true))
    L_inf.append(error)
    print("time:", time.perf_counter() - start)
    print(error, error_u)

#Project onto splines interpolant test -------------------------------------------------------
# # # ## create a fine grid to perform evaluation:
# N_pts = 400
# phi_finer = np.linspace(0, 2*np.pi, N_pts, endpoint = False)
# theta_finer = np.linspace(0, np.pi, N_pts, endpoint = True)
# XX2 = np.meshgrid(phi_finer, theta_finer)
# XYZ_0 = utils.sphere2cart(XX2[0], XX2[1])
# eval_pts = np.array([XYZ_0[0].reshape([N_pts**2,]), XYZ_0[1].reshape([N_pts**2,]),
#                      XYZ_0[2].reshape([N_pts**2,])])

# # u_true = source(XX2[0], XX2[1]).reshape([N_pts**2,]) 
# u_true = grad_source(XX2[0], XX2[1]).reshape([3, N_pts**2,])

# Ns = [16, 32, 64, 128, 256, 512, 1024]
# # Vector field projection ==
# l_infx = []
# l_infy = []
# l_infz = []
# areas = []

# for L in Ns:

#     [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
#     [Phi, The] = np.meshgrid(phis, thetas)
#     s_points = np.array(utils.sphere2cart(Phi,The))
#     sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#             s_points[2].reshape([(L+1)*2*L,])])

#     # create a dictionary for the grid
#     N, M = len(phis), len(thetas)
#     XX = np.meshgrid(phis, thetas[1:-1])
#     ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

#     simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
#     simplices = np.array(simplices); msimplices = np.array(msimplices)

#     grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
#                  "msimplices": msimplices, "mesh": ico, "sample_points": s_points, 'points': ico.points}
    
#     # pre-allocate arrays for the velocity field interpolants:
#     # pre-allocate arrays used during interpolant evaluation

#     verts0 = ico.points.T # vertices of the mesh
#     v_pts = ico.points[simplices]
#     Hs, Gs, Es = sphere_sims.ps_split(v_pts)

#     NN = N_pts**2
#     EE = np.zeros([7,NN,3])  
#     Bcc = np.zeros([NN,3])
#     nCs = np.zeros([NN,6], dtype = 'int64')
#     Cfs = np.zeros([6,NN])

#     #placeholder arrays
#     vals = np.array([ico.x, ico.y, ico.z])
#     grad_vals = 0*np.array([utils.grad_x(ico.points.T),
#                           utils.grad_y(ico.points.T),
#                           utils.grad_z(ico.points.T)])
    
#     # initialize coefficient arrays:
#     Nv = v_pts.shape[0]
#     cfs_map = np.zeros([3,Nv,3])
#     grad_fx = np.zeros([3,3,Nv]); grad_fy = np.zeros([3,3,Nv]); grad_fz = np.zeros([3,3,Nv])

#     # arrays used for the querying of the velocity field:
#     trangs_qs = np.zeros([N_pts**2,], dtype = 'int64')
#     verts_qs = np.zeros([N_pts**2, 3, 3])

#     phi_l = np.zeros([N_pts**2,], dtype = int)
#     theta_l = np.zeros([N_pts**2,], dtype = int)
#     vel_vals = np.zeros([3,N_pts**2])

#     coeffs0 = np.zeros([3,19,v_pts.shape[0]])
#     coeffs = sphere_sims.assemble_coefficients(coeffs0, simplices, vals, grad_vals, cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)

#     # u0[i,j,:] gives values of D_j u^i on grid points
#     u_coeffs = np.zeros([3, 4, L+1, 2*L])
#     u0 = np.zeros([3, 4, (L-1)*2*L + 2])
#     u_lms = np.zeros([3, L, 2*L+1], dtype = 'complex128')

#     outs = [u_lms[0,l,m] for l in range(0,L) for m in range(0,2*l+1)]

#     L1 = len(outs) # \sum_{\ell = 0}^L {\sum_{|m| \leq \ell} 
#     u_lms1d = np.zeros([3, 4, L1], dtype = 'complex128')
    
#     #initialize angular momentum operator:
#     L_plus = np.array(np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l-1)), (L,2*L+1), dtype = 'float64'))
#     L_minus = np.array(np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l+1)), (L,2*L+1), dtype = 'float64'))
#     L_minus[np.where(np.isnan(L_minus))] = 0.
#     L_z = np.array(np.fromfunction(lambda l,m: m-l, (L, 2*L+1), dtype = 'float64'))

#     psi0 = source(Phi, The)
#     # psi_lms = sphere_sim.inv_Laplacian(psi0, L)

#     # compute spherical harmonic coefficients of velocity from stream function
#     psi_lms = sph_tools.coeff_array(pysh.forward(psi0, L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 8),L)
#     u_lms[:] = sphere_sims.angular_momentum(psi_lms, u_lms, L_plus, L_minus, L_z)
    
#     start, start_clock = time.perf_counter(), time.process_time()
#     u0[:] = sphere_sims.project_onto_S12_PS_vector(u_lms, u_lms1d, u_coeffs, u0, L, L_plus, L_minus, L_z, grid_dict["sample_points"], Method = "MWSS")


#     coeffs[:] = sphere_sims.assemble_coefficients(coeffs0, simplices, u0[:,0,:], u0[:,1::,:], cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)

#     # compute preliminary indices:
#     bcc, trangs, v_pts = sphere_sims.query_vector_spline(eval_pts, phis, thetas, phi_l, theta_l, Bcc, verts0.T, msimplices, trangs_qs, verts_qs)

#     u_num = sphere_sims.vector_spline_eval(vel_vals, coeffs, eval_pts, EE, nCs, Bcc, Cfs, bcc, trangs, v_pts)  

#     finish, finish_clock = time.perf_counter(), time.process_time()


#     #project to get gradient
#     u_num = sphere_sims.tan_proj(u_num, eval_pts, sphere_sims.cross(u_num, u_num.copy(), eval_pts))

#     error = np.absolute(np.array(u_num)- np.array(u_true))
#     print(np.max(error))

#     # # errors = [np.max(error[0,:]), np.max(error[1,:]), np.max(error[2,:])]
#     # l_infx.append(errors[0])
#     # l_infy.append(errors[1])
#     # l_infz.append(errors[2])
#     # print(L, error)

#     print("time:", finish-start)

# l_infx = np.array(l_infx)
# areas = np.array(Ns)
# orders = np.log(l_infx[1::]/l_infx[0:-1])/np.log(areas[1::]/areas[0:-1])
# print("orders",orders)


# # # scalar function projection ---------------------------------
# u_true = source(XX2[0], XX2[1])

# Ns = [16, 32, 64, 128, 256, 512, 1024]
# areas = []
# l_inf = []
# for L in Ns:
#     [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
#     [Phi, The] = np.meshgrid(phis, thetas)
#     s_points = utils.sphere2cart(Phi,The) # the sample points of size (3, L+1, 2*L)
#     F_samples =  source(Phi, The)
#     grad_F = grad_source(Phi, The)

#     f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = False, backend = 'ducc', nthreads = 5)
#     f_lm_coeffs = sph_tools.coeff_array(f_lm, L)

#     #create a dictionary for the grid
#     N, M = len(phis), len(thetas)
#     XX = np.meshgrid(phis, thetas[1:-1])
#     ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)
#     a = np.max(ico.edge_lengths())
#     areas.append(a)

#     simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
#     grid = {"phis": phis, "thetas": thetas, "simplices": simplices,
#                  "msimplices": msimplices, "mesh": ico, "sample_points": s_points}
    
#     interp = sph_tools.project_onto_S12_PS(f_lm_coeffs, L, grid, Method = "MWSS")

#     # interp = spline_interp_structured(grid = grid["mesh"], simplices = grid["simplices"],
#     #              msimplices = grid["msimplices"], phi = grid["phis"], theta = grid["thetas"],
#     #              vals = F_samples, grad_vals = list(grad_F))
    
#     u_num = interp.eval(eval_pts).reshape([N_pts, N_pts])

#     error = np.max(np.absolute(u_num- u_true))
#     l_inf.append(error)

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
# Ns = [16, 32, 64, 128, 256, 512]
# L_inf = []

# for L in Ns:

#     [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
#     [Phi, The] = np.meshgrid(phis, thetas)
#     s_points = utils.sphere2cart(Phi,The)
#     sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#             s_points[2].reshape([(L+1)*2*L,])])

#     # create a dictionary for the grid
#     N, M = len(phis), len(thetas)
#     XX = np.meshgrid(phis, thetas[1:-1])
#     ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

#     simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
#     grid = {"phis": phis, "thetas": thetas, "simplices": simplices,
#                  "msimplices": msimplices, "mesh": ico, "sample_points": s_points}
    
#     F_samples = source(Phi, The)
#     u_true = grad_source(Phi, The)

#     f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = False, backend = 'ducc', nthreads = 5)
#     f_lm_coeffs = sph_tools.coeff_array(f_lm, L)
#     u_lms = [sph_tools.lm_array(A,L) for A in sph_tools.angular_momentum(f_lm_coeffs,L)]

#     u_num = []
#     for i in [0,1,2]:
#         u_num.append(pysh.inverse(u_lms[i], L, Spin = 0, Method = 'MWSS', Reality = False, backend = 'ducc', nthreads = 5))
    
#     # gradient operator
#     u_num = utils.tan_proj(s_points, utils.cross(u_num, s_points))
    
#     error = np.max(np.absolute(u_num - u_true))

#     L_inf.append(error)
#     print(error)

# scipy.io.savemat("../MatlabScripts/tests/spherical_harmonics/angular_momentum_operation.mat", {"u_true": np.array(u_true).real, "u_num": np.array(u_num).real})

#--------- Test functions -----------------------------------------------------
# alpha = pi/3
# def stream(phi, theta):
#     return  2*pi*(cos(theta)*cos(alpha) - cos(phi-pi/2)*sin(theta)*sin(alpha))
#     #return sin(phi)**2*(sin(theta)**2)
#     #return sin(theta)*cos(phi)

# def test_func_pt(phi, theta):
#     return stream(phi,theta)**10

# def test_func(xyz):
#     #
#     x = xyz[:,0]
#     y = xyz[:,1]
#     z = xyz[:,2]
#     r = np.sqrt(x**2 + y**2 + z**2)

#     # return (1/r**3)*(1/8)*(np.sqrt(35/pi))*((x-1j*y)**3).real + x**2 #(x**4 - 6*(x**2)*(y**2) + y**4)/(r_2**2)
#     phi, theta = cart2sphere([x,y,z])
#     #pdb.set_trace()

#     return stream(phi,theta)**10


# #
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

#     return cross([X,Y,Z],grad_psi)

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

#     return grad_psi


#-------------------------------------------------------------------------------------------
