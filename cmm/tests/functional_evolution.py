#-----------------------------------------------------------------------------
"""
Basic script to test the linear advection solver on the sphere
"""
# -----------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, igl
import numba as nb
# from ..core.interpolants.spherical_spline import PS_split_coeffs
from ..core.interpolants.spherical_spline import sphere_diffeomorphism
from ..core import spherical_simulations as sphere_sim
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as velocity

#--------------------------------- Setup --------------------------------------
# # Test identifications
names = ['sbr0', 'sbrpi2', 'sbrpi4', 'off_axis', 'deformational', 'rotating_deformational',
         'rotating_deformational_offaxis', 'rotating_deformational_offaxis_pi4']

# parameters to identify the file
# name = names[0] # name of test case
T = 1 # final integration time
remapping = False # flag for remapping
n_maps = 10 # default parameter for remapping steps
##-----------------------------------------------------------------------------

# U = velocity.u_deform_rot

@nb.jit(nopython = True)
def U(t,dt,X):
    # deformational flow test
    k = 2
    out = X.copy()
    c1 = 2*k*np.cos(np.pi*t/T)

    #rewrite over array:
    out[0] = -c1*X[2]*X[1]
    out[1] = X[2]*0 
    out[2] = c1*X[0]*X[1] 

    return  out

# define a grid to evaluate error as sanity check:
N_pts = 500
phi_finer = np.linspace(0, 2*np.pi, N_pts, endpoint = False)
theta_finer = np.linspace(0, np.pi, N_pts, endpoint = True)

XX2 = np.meshgrid(phi_finer, theta_finer)
XYZ_0 = utils.sphere2cart(XX2[0], XX2[1])
eval_pts = np.array([XYZ_0[0].reshape([N_pts**2,]), XYZ_0[1].reshape([N_pts**2,]),
                     XYZ_0[2].reshape([N_pts**2,])])

# to compute the error over the convergence data
tri_size, l_inf, linf_map, linf_grad_map = [], [], [], []

# convergence test for linear advection solver
# test function to check the tracer error
def test_func(xyz):
    x = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    return np.sin(x**2)*(10*np.cos(y-2)) + np.cos(10*z**5) + 2*np.sin(x*3*(z-0.1) + y**7)

u_true = test_func(eval_pts.T)

# =================================================================

for j in range(9):

    # pre-compute all quantities derived from the mesh-----------------
    ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=j)
    mesh = meshes.spherical_triangulation(ico0.points)
    ss = mesh.s_pts.shape
    vertices = mesh.vertices
    ps_split = mesh.ps_split()
    Hs = np.array(ps_split[0]); Gs = np.array(ps_split[1]); Es = np.array(ps_split[2])

    gammas = np.array(mesh.tan_vects)
    inds = np.array(mesh.simplices)
    v_pts = vertices[inds]

    vals = np.array([mesh.x, mesh.y, mesh.z])
    grad_vals = np.array([utils.grad_x(vertices.T),
                          utils.grad_y(vertices.T),
                          utils.grad_z(vertices.T)])
    
    vals_n = vals.copy(); grad_vals_n = grad_vals.copy()
    #for the integration:
    verts0 = mesh.vertices.T # vertices of the mesh
    spts = mesh.s_pts.reshape([3, ss[1]*ss[2]]) #stencil points about the vertices

    # pre-allocate arrays used during interpolant evaluation
    NN = len(verts0.T)
    edges_list = np.zeros([7,NN,3])  
    Bcc = np.zeros([NN,3])
    nCs = np.zeros([NN,6], dtype = 'int64')
    Cfs = np.zeros([6,NN])
    s_evals = np.zeros([3,4,NN])

    # initialize coefficient arrays:
    Nv = v_pts.shape[0]
    cfs_map = np.zeros([3,Nv,3])
    grad_fx = np.zeros([3,3,Nv]); grad_fy = np.zeros([3,3,Nv]); grad_fz = np.zeros([3,3,Nv])

    coeffs0 = np.zeros([3,19,v_pts.shape[0]])

    # largest radius for circumscribed triangle.
    tri_size.append(np.max(ico0.edge_lengths())) # note this is only suitable for the icosahedral discretization
    #--------------------------------------------------------------------
    # initialize the coefficients defining the interpolant:
    coeffs0 = sphere_sim.assemble_coefficients(coeffs0, inds, vals, grad_vals, cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)
    coeffs = coeffs0.copy()

    Nt = 2**j + 10 # number of time steps
    tspan = np.linspace(0, T, Nt, endpoint = False); dt = tspan[1]-tspan[0]

    # initialize empty list for remaps.
    remaps = []

    # timing in both CPU and wallclock time
    start, start_clock = time.perf_counter(), time.process_time()
    identity = True

    count = 1
    for t in tspan:

        # compute the one step map:
        yn = sphere_sim.RK4_proj(t, dt, verts0, U)
        spts_n = sphere_sim.RK4_proj(t,dt, spts, U).reshape(ss) 

        # query the mesh at the vertex points:
        bcc, trangs, vpts = sphere_sim.query(yn, vertices, inds, Bcc)

        if not identity:
        #perform evaluation of stencil points at previous map
            s_evals = sphere_sim.stencil_eval(coeffs, spts_n, bcc, trangs, vpts, nCs, Cfs, edges_list, Bcc, s_evals)
        else:
            s_evals =  spts_n

        # then redefine the interpolant:    
        vals_n, grad_vals_n[0,:,:], grad_vals_n[1,:,:], grad_vals_n[2,:,:] = sphere_sim.spline_proj_sphere(s_evals, gammas)

        #redefine the coefficients from the values:
        coeffs = sphere_sim.assemble_coefficients(coeffs, inds, vals_n, grad_vals_n, cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)
        identity = False

        
        # # if remapping and (count % n_maps == 0):
        # #     remaps.append(mapping)
        # #     mapping = mapping0
        # #     identity = True
        # #     jr = True

        count +=1

    finish, finish_clock = time.perf_counter(), time.process_time()
    mapping = sphere_diffeomorphism(mesh, vals_n, grad_vals_n)


    # if jr == False:
    #     remaps.append(mapping)

    # Compute the error 
    if remapping:
        evals = evol.compose_maps(remaps, eval_pts)
    else:
        evals = mapping(eval_pts)
        # grad_evals = mapping.eval_grad(eval_pts, evals)

    u_num = test_func(np.array(evals).T)


    print("wall time (s):", finish - start)
    print("CPU time (s):", finish_clock - start_clock)

    error = np.max(np.absolute(u_true-u_num))
    map_error_x = np.max(np.absolute(np.array(evals[0]) - eval_pts[0]))
    map_error_y = np.max(np.absolute(np.array(evals[1]) - eval_pts[1]))
    map_error_z = np.max(np.absolute(np.array(evals[2]) - eval_pts[2]))

    # map_grad_error_x = np.max(np.absolute(np.array(grad_evals[0]) - grad_true[0]))
    # map_grad_error_y = np.max(np.absolute(np.array(grad_evals[1]) - grad_true[1]))
    # map_grad_error_z = np.max(np.absolute(np.array(grad_evals[2]) - grad_true[2]))

    l_inf.append(error)
    linf_map.append([map_error_x, map_error_y, map_error_z])
    # l_inf_grad_map.append([map_grad_error_x, map_grad_error_y, map_grad_error_z])

    print("solution error:", error)
    print("map error:", map_error_x, map_error_y, map_error_z)
    # print("grad map error:", map_grad_error_x, map_grad_error_y, map_grad_error_z)



