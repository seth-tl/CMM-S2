#-----------------------------------------------------------------------------
"""
Basic script to test the linear advection solver on the sphere
"""
# -----------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle
from ..core.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as velocity

#--------------------------------- Setup --------------------------------------
# parameters to identify the file
U = velocity.u_deform_rot # advecting velocity field
T = 1 # final integration time
remapping = False # flag for remapping
n_maps = 10 # default parameter for remapping steps
##-----------------------------------------------------------------------------

# define a grid to evaluate error as sanity check:
N_pts = 200
phi_finer = np.linspace(0, 2*np.pi, N_pts, endpoint = False)
theta_finer = np.linspace(0, np.pi, N_pts, endpoint = True)

XX2 = np.meshgrid(phi_finer, theta_finer)
XYZ_0 = utils.sphere2cart(XX2[0], XX2[1])
eval_pts = np.array([XYZ_0[0].reshape([N_pts**2,]), XYZ_0[1].reshape([N_pts**2,]),
                     XYZ_0[2].reshape([N_pts**2,])]).T

# to compute the error over the convergence data
tri_size, l_inf, linf_map, linf_grad_map = [], [], [], []

# test function to check the tracer error as we compute
def test_func(xyz):
    x = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    return np.sin(x**2)*(10*np.cos(y-2)) + np.cos(10*z**5) + 2*np.sin(x*3*(z-0.1) + y**7)

# u_true = velocity.static_vort_IC(T, eval_pts)
u_true = test_func(eval_pts)

path_to_repo = os.getcwd() + '/cmm_s2'
# time scaling -----
for j in range(7):

    ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=j)
    tri_size.append(np.max(ico0.edge_lengths()))
    # mesh = meshes.spherical_triangulation(ico0.points)

    # load pre-computed mesh
    mesh = pickle.load(open(path_to_data + '/data/icosahedral_mesh_ref_%s.txt' %j, "rb"))

    vals = np.array([mesh.x, mesh.y, mesh.z])
    grad_vals = np.array([utils.grad_x(mesh.vertices.T),
                    utils.grad_y(mesh.vertices.T),
                    utils.grad_z(mesh.vertices.T)])
    
    jet_vals = np.zeros([3,4, len(vals[0,:])])
    jet_vals[:,0,:] = vals; jet_vals[:,1:,:] = grad_vals

    # largest radius for circumscribed triangle.

    # pre-allocate coefficient array
    coeffs = np.zeros([3,19, np.shape(mesh.vertices[np.array(mesh.simplices)])[0]])

    # define initial discretization of the map
    mapping0 = sphere_diffeomorphism(mesh, jet_vals, coeffs)

    # initialize empty list for remaps.
    remaps = []
    # for initialization
    mapping = mapping0

    # timing in both CPU and wallclock time
    Nt = 2**j + 10
    tspan = np.linspace(0, T, Nt, endpoint = False)
    dt = tspan[1]-tspan[0]

    out_array = eval_pts.copy()

    # # map evaluation timing
    # start, start_clock = time.perf_counter(), time.process_time()
    # evals0 = mapping(eval_pts)

    # finish, finish_clock = time.perf_counter(), time.process_time()

    # print("wall time (s):", finish - start)
    # print("CPU time (s):", finish_clock - start_clock)

    # print(np.max(np.absolute(u_true - test_func(evals0.T))))
    # -----------------------

    start, start_clock = time.perf_counter(), time.process_time()

    identity = True

    count = 1
    for t in tspan:
        mapping = evol.advect_project_sphere(mapping, evol.RK4_proj, t, dt, U, identity)
        identity = False

        if remapping and (count % n_maps == 0):
            remaps.append(mapping)
            mapping = mapping0
            identity = True

        count +=1

    finish, finish_clock = time.perf_counter(), time.process_time()

    if identity == False:
        remaps.append(mapping)

    # Compute the error 
    if remapping:
        evals = evol.compose_maps(remaps, eval_pts)
    else:
        evals = mapping(eval_pts)

    u_num = test_func(evals.T)
    error = np.max(np.absolute(u_true-u_num))

    print("wall time (s):", finish - start)
    print("CPU time (s):", finish_clock - start_clock)
    print("error:", error)
