#/---
"""
This script provides a variety of unit tests related to the spherical harmonic tools
"""
#/---
import pdb, time, scipy.io
import numpy as np
import pyssht as pysh
from ..core import utils
from ..core import mesh_functions as meshes
from ..core import dynamical_fields as vel
from ..core.spherical_spline import spline_interp_vec, spline_interp_structured
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
# # Poisson solver test using the SSHT package
# Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]

# def source(phi,theta):
#     return vel.rossby_wave(phi,theta, t= 0)

# def solution(phi,theta):
#     x,y,z = utils.sphere2cart(phi,theta)

#     return -z*((1-z**2)**2 -8*(1-z**2)*x**2 + 8*x**4)
#     #return -np.cos(theta)*np.cos(4*phi)*np.sin(theta)**4

# def velocity(phi,theta, L):
#     x,y,z = np.array(utils.sphere2cart(phi,theta)).reshape(3,2*L*(L+1))
#     psi_x = -32*z*x**3 + 16*z*(1-z**2)*x
#     psi_z = -(1-z**2)**2 + 4*z**2*(1-z**2) + 8*x**2*(1-3*z**2) - 8*x**4
#     return np.array([y*psi_z, z*psi_x - x*psi_z, -y*psi_x])

# L_inf = []
# for L in Ns:
    
#     [Phi, The] = sph_tools.MW_sampling(L)
#     F_samples = source(Phi, The)
#     F_true = solution(Phi,The)

#     start = time.perf_counter()

#     f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = True, backend = 'ducc', nthreads = 8)
#     f_lm_coeffs = sph_tools.coeff_array(f_lm, L)

#     L_flm = sph_tools.lm_array(sph_tools.Poisson_Solve(f_lm_coeffs,L),L)

#     f_num = pysh.inverse(L_flm, L, Spin = 0, Method = 'MWSS', Reality = True, backend = 'ducc', nthreads = 8)


#     #project velocity field
#     [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
#     [Phi, The] = np.meshgrid(phis, thetas)
#     s_points = utils.sphere2cart(Phi,The)
#     sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#             s_points[2].reshape([(L+1)*2*L,])])
#     #obtain stream function
#     psi0 = source(Phi, The)
#     psi_lms = sph_tools.inv_Laplacian(psi0, L)

#     #create a dictionary for the grid
#     N, M = len(phis), len(thetas)
#     XX = np.meshgrid(phis, thetas[1:-1])
#     ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

#     simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
#     grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
#                  "msimplices": msimplices, "mesh": ico, "sample_points": s_points}

#     u0 = sph_tools.project_stream(psi_lms, L, grid = grid_dict, Method = "MWSS")
#     error_u = np.max(np.absolute(u0.eval(sample_pts) - velocity(Phi,The,L)))

#     error = np.max(np.absolute(f_num - F_true))
#     L_inf.append(error)
#     print("time:", time.perf_counter() - start)
#     print(error, error_u)

#Project onto splines interpolant test -------------------------------------------------------
# # scalar function projection ---------------------------------
N_pts = 1000
phis = np.linspace(0,2*np.pi, N_pts, endpoint = False)
thetas = np.linspace(0,np.pi, N_pts)
XX = np.meshgrid(phis,thetas)
eval_pts = np.array(utils.sphere2cart(XX[0], XX[1])).reshape([3, N_pts**2])

u_true = source(XX[0], XX[1])



Ns = [16, 32, 64, 128, 256, 512, 1024]
areas = []
l_inf = []
for L in Ns:
    [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
    [Phi, The] = np.meshgrid(phis, thetas)
    s_points = utils.sphere2cart(Phi,The) # the sample points of size (3, L+1, 2*L)
    F_samples =  source(Phi, The)
    grad_F = grad_source(Phi, The)

    f_lm = pysh.forward(F_samples, L, Spin = 0, Method = 'MWSS', Reality = False, backend = 'ducc', nthreads = 5)
    f_lm_coeffs = sph_tools.coeff_array(f_lm, L)

    #create a dictionary for the grid
    N, M = len(phis), len(thetas)
    XX = np.meshgrid(phis, thetas[1:-1])
    ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)
    a = np.max(ico.edge_lengths())
    areas.append(a)

    simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
    grid = {"phis": phis, "thetas": thetas, "simplices": simplices,
                 "msimplices": msimplices, "mesh": ico, "sample_points": s_points}
    
    interp = sph_tools.project_onto_S12_PS(f_lm_coeffs, L, grid, Method = "MWSS")

    # interp = spline_interp_structured(grid = grid["mesh"], simplices = grid["simplices"],
    #              msimplices = grid["msimplices"], phi = grid["phis"], theta = grid["thetas"],
    #              vals = F_samples, grad_vals = list(grad_F))
    
    u_num = interp.eval(eval_pts).reshape([N_pts, N_pts])

    error = np.max(np.absolute(u_num- u_true))
    l_inf.append(error)

    print(L, error)

l_inf = np.array(l_inf)
areas = np.array(areas)
orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(areas[1::]/areas[0:-1])
print("orders",orders)



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
#-------------------------------------------------------------------------------------------
