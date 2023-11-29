#/----
"""
This script includes convergence tests for all interpolants in the
interpolants module. The tests are labelled and simply commented out;
meant to test for any modifications made to a single interplant class.
"""
#/----
import pdb, time, scipy.io, sys
import numpy as np
# sys.path.append('/mnt/c/Users/Admin/Uni/McGill/PhD_Research/Code')
from ..core import utils
from ..core import mesh_functions as meshes
from ..core.interpolants import torus_interpolants as interpT2
#===============================================================================
# # Scalar spherical spline interpolation test
# from ..core.interpolants.spherical_spline import spherical_spline
#
# N = 512
# phis = np.linspace(0, 2*np.pi, N)
# theta = np.linspace(0, np.pi, N)
# pth = np.meshgrid(phis, theta)
#
# qs = utils.sphere2cart(pth[0],pth[1])
# eval_pts = np.array([qs[0].reshape([N**2,]), qs[1].reshape([N**2,]), qs[2].reshape([N**2,])])
#
# l_inf = []
# l_inf_grad = []
# tri_size = []
#
# def d_Proj(X, A):
#     outx = (1-X[:,0]**2)*A[0] + (-X[:,0]*X[:,1])*A[1] + (-X[:,0]*X[:,2])*A[2]
#     outy = (-X[:,0]*X[:,1])*A[0] + (1-X[:,1]**2)*A[1] + (-X[:,1]*X[:,2])*A[2]
#     outz = (-X[:,0]*X[:,2])*A[0] + (-X[:,1]*X[:,2])*A[1] + (1-X[:,2]**2)*A[2]
#
#     return [outx, outy, outz]
#
# def test_func(X):
#     return X[:,0]**5 + X[:,0]*X[:,1]*X[:,2] + np.sin(X[:,1])
#
# def grad_test_func(X):
#     out_x = 5*X[:,0]**4 + X[:,2]*X[:,1]
#     out_y = X[:,2]*X[:,0] + np.cos(X[:,1])
#     out_z = X[:,1]*X[:,0]
#     return d_Proj(X, [out_x, out_y, out_z])
#
# u_true = test_func(eval_pts.T)
# u_grad_true = grad_test_func(eval_pts.T)
#
# import stripy
# for j in range(8):
#
#     ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=j)
#     mesh = meshes.spherical_triangulation(ico0.points)
#     vals = test_func(mesh.vertices)
#     grad_vals = grad_test_func(mesh.vertices)
#
#     # note this is only suitable for the icosahedral discretization
#     # largest radius for circumscribed triangle.
#     tri_size.append(np.max(ico0.edge_lengths()))
#
#
#     # precompute mesh quantities for ps_split
#     ps_split_mesh = mesh.ps_split()
#
#
#     interpolant = spherical_spline(mesh = mesh, vals = vals,
#                                    grad_vals = grad_vals,
#                                    ps_split = ps_split_mesh)
#
#     # timing the evaluation both in CPU and wallclock time
#     start, start_clock = time.perf_counter(), time.process_time()
#
#     u_num = interpolant(eval_pts)
#
#     finish, finish_clock = time.perf_counter(), time.process_time()
#
#     print("wall time (s):", finish - start)
#     print("CPU time (s):", finish_clock - start_clock)
#
#     u_grad_num = interpolant.gradient(eval_pts)
#
#     # compute l-inf error at the query points
#     error = np.max(np.absolute(u_num - u_true))
#     grad_error = np.max(np.absolute(np.array(u_grad_num) - np.array(u_grad_true)))
#
#     l_inf.append(error)
#     l_inf_grad.append(grad_error)
#
#     print("error:", error)
#     print("grad error:", grad_error)
#
# scipy.io.savemat("./cmm/tests/data/spline_interpolant.mat",{"u_true":u_true, "u_num":u_num,
#                                                             "u_grad_true": u_grad_true,
#                                                             "u_grad_num": u_grad_num} )
#
# # convergence against triangle maximum edge length -----
# l_inf = np.array(l_inf)
# areas = np.array(tri_size)
# orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(areas[1::]/areas[0:-1])
# print("order of convergence:",orders)
# print("l-inf errors:", l_inf)

## ==============================================================================
# # Spherical diffeomorphism interpolation test
# from ..core.interpolants.spherical_spline import sphere_diffeomorphism

# N = 500
# # phis = np.random.uniform(0,2*pi,N)
# # theta = np.random.uniform(0,pi,N)
# phis = np.random.uniform(0, 2*np.pi, N)
# theta = np.random.uniform(0, np.pi, N)
# pth = np.meshgrid(phis, theta)

# qs = utils.sphere2cart(pth[0],pth[1])
# eval_pts = np.array([qs[0].reshape([N**2,]), qs[1].reshape([N**2,]), qs[2].reshape([N**2,])])

# l_inf_map = []
# l_inf_grad_map = []
# tri_size = []

# grad_true = [utils.grad_x(eval_pts), utils.grad_y(eval_pts), utils.grad_z(eval_pts)]

# import stripy
# for j in range(6):

#     ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=j)
#     mesh = meshes.spherical_triangulation(ico0.points)
#     vals = [mesh.x, mesh.y, mesh.z]
#     grad_vals = [utils.grad_x(mesh.vertices.T),
#                  utils.grad_y(mesh.vertices.T),
#                  utils.grad_z(mesh.vertices.T)]
#     # note this is only suitable for the icosahedral discretization
#     # largest radius for circumscribed triangle.
#     tri_size.append(np.max(ico0.edge_lengths()))

#     # timing the generation and evaluation of the map both in CPU and wallclock time
#     start, start_clock = time.perf_counter(), time.process_time()

#     # define initial discretization of the map
#     mapping = sphere_diffeomorphism(mesh = mesh, vals = vals,
#                                         grad_vals = grad_vals)

#     evals = mapping(eval_pts)
#     finish, finish_clock = time.perf_counter(), time.process_time()

#     print("wall time (s):", finish - start)
#     print("CPU time (s):", finish_clock - start_clock)

#     grad_evals = mapping.eval_grad(q_pts = eval_pts, eval_pts = evals)

#     # compute l-inf error at the query points
#     map_error_x = np.max(np.absolute(np.array(evals[0]) - eval_pts[0]))
#     map_error_y = np.max(np.absolute(np.array(evals[1]) - eval_pts[1]))
#     map_error_z = np.max(np.absolute(np.array(evals[2]) - eval_pts[2]))

#     map_grad_error_x = np.max(np.absolute(np.array(grad_evals[0]) - grad_true[0]))
#     map_grad_error_y = np.max(np.absolute(np.array(grad_evals[1]) - grad_true[1]))
#     map_grad_error_z = np.max(np.absolute(np.array(grad_evals[2]) - grad_true[2]))

#     l_inf_map.append([map_error_x, map_error_y, map_error_z])
#     l_inf_grad_map.append([map_grad_error_x, map_grad_error_y, map_grad_error_z])

#     print("map error:", map_error_x, map_error_y, map_error_z)
#     print("grad map error:", map_grad_error_x, map_grad_error_y, map_grad_error_z)

# # convergence against triangle maximum edge length -----
# l_inf_map = np.array(l_inf_map)
# areas = np.array(tri_size)
# orders = np.log(l_inf_map[1::, 0]/l_inf_map[0:-1, 0])/np.log(areas[1::]/areas[0:-1])
# print("order of convergence:",orders)
# print("l-inf errors:", l_inf_map)

# ==============================================================================
# Bilinear T2 test

N = 2000
phis = np.linspace(0, 2*np.pi, N)
theta = np.linspace(0, 2*np.pi, N)
eval_pts = np.meshgrid(phis, theta)


l_inf = []
mesh_size = []

def test_func(X):

    return np.sin(X[0]) + np.cos(X[1])

u_true = test_func(eval_pts)
Ns = [16, 32, 64, 128, 256, 512, 1024]
for N in Ns:

    xs = np.linspace(0, 2*np.pi, N)
    ys = np.linspace(0, 2*np.pi, N)
    X = np.meshgrid(xs, ys)
    vals = test_func(X)

    # define interpolant:
    interpolant = interpT2.Bilinear_T2(xs, ys, vals)
    # timing the evaluation both in CPU and wallclock time
    start, start_clock = time.perf_counter(), time.process_time()

    u_num = interpolant(eval_pts)

    finish, finish_clock = time.perf_counter(), time.process_time()

    print("wall time (s):", finish - start)
    print("CPU time (s):", finish_clock - start_clock)


    # compute l-inf error at the query points
    error = np.max(np.absolute(u_num - u_true))
    mesh_size.append(2*np.pi/N)
    l_inf.append(error)

    print("error:", error)

# convergence against triangle maximum edge length -----
l_inf = np.array(l_inf)
areas = np.array(mesh_size)
orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(areas[1::]/areas[0:-1])
print("order of convergence:",orders)
print("l-inf errors:", l_inf)