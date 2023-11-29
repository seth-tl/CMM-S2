#-----------------------------------------------------------------------------
"""
Basic script to test the linear advection solver on the sphere
"""
# -----------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time
from ..core.interpolants.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as velocity

#--------------------------------- Setup --------------------------------------
# # Test identifications
names = ['sbr0', 'sbrpi2', 'sbrpi4', 'off_axis', 'deformational', 'rotating_deformational',
         'rotating_deformational_offaxis', 'rotating_deformational_offaxis_pi4']

# parameters to identify the file
name = names[0] # name of test case
U = velocity.u_deform_rot # advecting velocity field
T = 5 # final integration time
remapping = True # flag for remapping
n_maps = 10 # default parameter for remapping steps
##-----------------------------------------------------------------------------

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
    x = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2];
    return np.sin(x**2)*(10*np.cos(y-2)) + np.cos(10*z**5) + 2*np.sin(x*3*(z-0.1) + y**7)

u_true = test_func(eval_pts.T)

for j in range(7):

    ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=j)
    mesh = meshes.spherical_triangulation(ico0.points)
    vals = [mesh.x, mesh.y, mesh.z]
    grad_vals = [utils.grad_x(mesh.vertices.T),
                 utils.grad_y(mesh.vertices.T),
                 utils.grad_z(mesh.vertices.T)]

    # note this is only suitable for the icosahedral discretization
    # largest radius for circumscribed triangle.
    tri_size.append(np.max(ico0.edge_lengths()))

    # define initial discretization of the map
    mapping0 = sphere_diffeomorphism(mesh = mesh, vals = vals,
                                    grad_vals = grad_vals)

    Nt = 2**j + 10 # number of time steps
    tspan = np.linspace(0, T, Nt, endpoint = False)
    dt = tspan[1]-tspan[0]
    count = 0

    # initialize empty list for remaps.
    # memory could be pre-allocated if a known number of maps is used
    remaps = []

    # for initialization
    mapping = mapping0

    # timing in both CPU and wallclock time
    start, start_clock = time.perf_counter(), time.process_time()
    identity = True

    for t in tspan:
        jr = False #"just remapped"
        mapping = evol.advect_project_sphere(mapping, evol.RK4_proj, t, dt, U, identity)
        identity = False

        if remapping and (count % n_maps == 0):
            remaps.append(mapping)
            mapping = mapping0
            jr = True

        count +=1

    finish, finish_clock = time.perf_counter(), time.process_time()

    if jr == False:
        remaps.append(mapping)

    ##save maps for convergence plots
#     file = open("data/advection_test/paper_data/test_%s_maps_icosahedral_k%s_Nt_%s_T_%s.txt" %(name, int(j), Nt, T), "wb")
#     pickle.dump(interpolant, file)
#     # pickle.dump(remaps, file)


    # Compute the error 
    if remapping:
        evals = evol.compose_maps(remaps, eval_pts)
    else:
        evals = mapping(eval_pts)
        grad_evals = mapping.eval_grad(eval_pts, evals)

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
