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
from ..core.spherical_spline import spline_interp_velocity
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
Ls = [16, 32, 64, 128, 256, 512, 1024, 2048]

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
Lf = 1024
[thetas, phis] = pysh.sample_positions(L= Lf, Method = "MWSS", Grid = False)
[Phif, Thef] = np.meshgrid(phis, thetas)
s_points = utils.sphere2cart(Phif,Thef) # the sample points of size (3, L+1, 2*L)
sample_pts_f = np.array([s_points[0].reshape([(Lf+1)*2*Lf,]), s_points[1].reshape([(Lf+1)*2*Lf,]),
        s_points[2].reshape([(Lf+1)*2*Lf,])]).T

for L in Ls:

    mesh_u = meshes.structure_spherical_triangulation(L = L)
    # (p = 3, x, 19, Nv) points needed to define interpolant pth-order time and space
    coeffs_u = np.zeros([3, 3, 19, np.shape(mesh_u.points[np.array(mesh_u.simplices)])[0]])
    vals_u = np.zeros([3, 3, 4, len(mesh_u.points)])

    # this defines where the map will be sampled
    [Phi, The] = np.meshgrid(mesh_u.phi, mesh_u.theta)
    s_points = utils.sphere2cart(Phi,The)
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])]).T
   
    # intialize the interpolant:
    U = spline_interp_velocity(mesh_u, vals_u, coeffs_u, ts = [0,0,0])
    #----------------------------------------------------------
    # define the operators for the spherical harmonic transform
    # the inverse Laplacian:
    els = np.arange(0,L)
    els[0] = 1
    D_inv = -1/(els*(els + 1))
    D_inv[0] = 0. # zero out first mode

    # precompute components for the angular momentum operator
    L_plus = np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l-1)), (L,2*L+1), dtype = 'float64')
    L_minus = np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l+1)), (L,2*L+1), dtype = 'float64')
    L_plus[np.where(np.isnan(L_plus))] = 0. 
    L_minus[np.where(np.isnan(L_minus))] = 0. 
    L_z = np.fromfunction(lambda l,m: m-l, (L, 2*L+1), dtype = 'float64')

    #obtain initial velocity field
    psi0 = source(Phi, The)
    psi_lms = sph_tools.inv_Laplacian(psi0, L, D_inv)
    
    temp = 0*psi_lms # temporary array for the projection step

    start = time.perf_counter()
    sph_tools.project_stream(psi_lms, L, L_plus, L_minus, L_z, U, 0, temp) 
    print("time:", time.perf_counter() - start)

    #project velocity field
    error_u = np.max(np.absolute(U(0,0,sample_pts_f.T) - velocity(Phif,Thef,Lf)))

    print("error:", error_u)



#Project onto splines interpolant test -------------------------------------------------------
# # scalar function projection ---------------------------------
# N_pts = 1000
# phis = np.linspace(0,2*np.pi, N_pts, endpoint = False)
# thetas = np.linspace(0,np.pi, N_pts)
# XX = np.meshgrid(phis,thetas)
# eval_pts = np.array(utils.sphere2cart(XX[0], XX[1])).reshape([3, N_pts**2])

# u_true = source(XX[0], XX[1])



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

# l_inf = np.array(l_inf)
# areas = np.array(areas)
# orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(areas[1::]/areas[0:-1])
# print("orders",orders)

