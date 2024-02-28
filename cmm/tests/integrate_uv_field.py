#-----------------------------------------------------------------------------
"""
Basic script to test the linear advection solver on the sphere
"""
# -----------------------------------------------------------------------------
import numpy as np
import xarray as xr
import glob, cftime
import pdb, stripy, time, pickle
from ..core.interpolants.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as velocity
import warnings
warnings.filterwarnings("ignore")

#--------------------------------- Setup --------------------------------------
# # # Test identifications
# U = velocity.u_deform_rot # advecting velocity field
# T = 1 # final integration time
# remapping = False # flag for remapping
# n_maps = 10 # default parameter for remapping steps
##-----------------------------------------------------------------------------

# prepare all the data -------------------
path = glob.glob("/mnt/c/Users/setht/Research/Code/cmm/data/AtmosphericConnectomeData/*3.nc")

data = xr.open_mfdataset(path, parallel = True)

# get grid points and 
uv_lat = (np.pi/180)*data["lat"].values
uv_lon = (np.pi/180)*data["lon"].values
angs = np.meshgrid(uv_lon, uv_lat)

# select only the upper level for the fields:
U_10 = (data["ua"].values)[:,-3,:,:]; V_10 = (data["va"].values)[:,-3,:,:]

# then transform into Cartesian coordinates on the meshgrid:
# TODO: confirm this transformation

Ux = -(np.sin(angs[0]))[None,:,:,]*U_10 - (np.sin(angs[1])*np.cos(angs[1]))[None,:,:]*V_10
Uy = (np.cos(angs[0]))[None,:,:,]*U_10 - (np.sin(angs[1])*np.sin(angs[0]))[None,:,:]*V_10
Uz = (np.cos(angs[1]))[None,:,:,]*V_10


angs = np.array(angs).reshape([2, len(uv_lat)*len(uv_lon)])

# have to change convention for the grid points
uv_points = utils.sphere2cart(angs[0], -angs[1] + np.pi/2)
mesh_uv = meshes.spherical_triangulation(np.array(uv_points).T)


years = 1
u_dt = years*365*24*60*60 # in seconds
year_0 = data["time"][0]

u_time = np.array([(data["time"][i].values - year_0.values).total_seconds() for i in range(len(data["time"]))])

R_earth = 6371*1000 # in metres

# rescale all the wind velocities to the unit sphere and reshape data
# TODO: is this the correct transformation? should be
ss = np.shape(Ux)

# scale to unit sphere

Ux = (Ux/R_earth).reshape(ss[0], ss[1]*ss[2])
Uy = (Uy/R_earth).reshape(ss[0], ss[1]*ss[2])
Uz = (Uz/R_earth).reshape(ss[0], ss[1]*ss[2])

U = np.zeros([ss[0],3,ss[1]*ss[2]])
U[:,0,:] = Ux; U[:,1,:] = Uy; U[:,2,:] = Uz


class UV_interpolant_linear(object):

    """
    Basic class for spherical linear spline interpolation for a
    mapping of the sphere

    inputs:
        mesh: spherical_triangulation object from mesh_functions.py
        vals: (N,3) np.array of values of each component at mesh.points()
    """

    def __init__(self, mesh, vals, times):

        self.mesh = mesh
        self.vals = vals
        #precompute the coefficients:
        self.inds = np.array(self.mesh.simplices)
        self.times = times
        self.dt = times[1]-times[0]
        # self.coeffs = np.array(self.vals)[:, inds.T]
        return

    def __call__(self, t, dt, q_pts):

        bcc, trangs, v_pts = self.mesh.query(q_pts)

        # TODO: figure out a convenient way to select the time window
        # i.e. get rid of these if statements

        t_ind = np.floor(t /self.dt).astype('int')

        if t_ind == len(self.times):
            t_ind += -1
        
        if t_ind != 0 and t_ind != len(self.times)-1:

            vs = self.vals[t_ind-1:t_ind + 2,:,:]
            coeffs = vs[:,:,self.inds.T]

            cfs = coeffs[:,:,:,trangs]

            out_x = bcc[:,0]*cfs[:,0,0,:] + bcc[:,1]*cfs[:,0,1,:] + bcc[:,2]*cfs[:,0,2,:]
            out_y = bcc[:,0]*cfs[:,1,0,:] + bcc[:,1]*cfs[:,1,1,:] + bcc[:,2]*cfs[:,1,2,:]
            out_z = bcc[:,0]*cfs[:,2,0,:] + bcc[:,1]*cfs[:,2,1,:] + bcc[:,2]*cfs[:,2,2,:]

            # then interpolate in time:
            tau0, tau1, tau2 = self.times[t_ind-1:t_ind+2]

            l0 = (t-tau1)*(t-tau2)/(2*self.dt**2)
            l1 = (t-tau0)*(t-tau2)/(-self.dt**2)
            l2 = (t-tau0)*(t-tau1)/(2*self.dt**2)

            outx = l0*out_x[0] + l1*out_x[1] + l2*out_x[2]
            outy = l0*out_y[0] + l1*out_y[1] + l2*out_y[2]
            outz = l0*out_z[0] + l1*out_z[1] + l2*out_z[2]


            return np.array([outx, outy, outz])
        else:
            vs = self.vals[0:2,:,:]

            coeffs = vs[:,:,self.inds.T]

            cfs = coeffs[:,:,:,trangs]
            out_x = bcc[:,0]*cfs[:,0,0,:] + bcc[:,1]*cfs[:,0,1,:] + bcc[:,2]*cfs[:,0,2,:]
            out_y = bcc[:,0]*cfs[:,1,0,:] + bcc[:,1]*cfs[:,1,1,:] + bcc[:,2]*cfs[:,1,2,:]
            out_z = bcc[:,0]*cfs[:,2,0,:] + bcc[:,1]*cfs[:,2,1,:] + bcc[:,2]*cfs[:,2,2,:]

            tau = (t - self.times[t_ind])/self.dt

            outx = (1-tau)*out_x[0] + tau*out_x[1]
            outy = (1-tau)*out_y[0] + tau*out_y[1]
            outz = (1-tau)*out_z[0] + tau*out_z[1]


            return np.array([outx, outy, outz]) 
  

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


# perform a back and forth flow with a single velocity field

ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=4)
mesh = meshes.spherical_triangulation(ico0.points)
vals = np.array([mesh.x, mesh.y, mesh.z])
grad_vals = np.array([utils.grad_x(mesh.vertices.T),
                utils.grad_y(mesh.vertices.T),
                utils.grad_z(mesh.vertices.T)])

# note this is only suitable for the icosahedral discretization
# largest radius for circumscribed triangle.
# tri_size.append(np.max(ico0.edge_lengths()))

# define initial discretization of the map
mapping0 = sphere_diffeomorphism(mesh = mesh, vals = vals,
                                grad_vals = grad_vals)

Nt = len(u_time)*10 # number of time steps
tspan = np.linspace(0, u_time[-1], Nt, endpoint = False)
dt = tspan[1]-tspan[0]


# initialize empty list for remaps.
# memory could be pre-allocated if a known number of maps is used
remaps = []

# for initialization
mapping = mapping0

# timing in both CPU and wallclock time
start, start_clock = time.perf_counter(), time.process_time()
identity = True

# initialize the velocity
UV = UV_interpolant_linear(mesh = mesh_uv, vals = U, times = u_time)


n_maps = 10

count = 1
j = 0
n_save = 10
saves = np.zeros([len(tspan)//n_maps + 1, 3, 4, len(ico0.points)])

for t in tspan:
    print(t)
    mapping = evol.advect_project_sphere(mapping, evol.RK4_proj, t, dt, UV, identity)
    identity = False

    if (count % n_maps) == 0:
        saves[j,:,0,:] = mapping.vals
        saves[j,:,1::,:] = mapping.grad_vals
        remaps.append(mapping)
        identity = True
        mapping = mapping0
        j+=1

    count +=1

file = open("/mnt/c/Users/setht/Research/Code/cmm/data/AtmosphericConnectomeData/map_data/advection_test1.txt", "wb")
pickle.dump({"values": saves, "mesh": mesh}, file)

finish, finish_clock = time.perf_counter(), time.process_time()

# if jr == False:
#     remaps.append(mapping)




# # Compute the error 
# if remapping:
#     evals = evol.compose_maps(remaps, eval_pts)
# else:
#     evals = mapping(eval_pts)
#     grad_evals = mapping.eval_grad(eval_pts, evals)

# u_num = test_func(np.array(evals).T)


# print("wall time (s):", finish - start)
# print("CPU time (s):", finish_clock - start_clock)

# error = np.max(np.absolute(u_true-u_num))
# map_error_x = np.max(np.absolute(np.array(evals[0]) - eval_pts[0]))
# map_error_y = np.max(np.absolute(np.array(evals[1]) - eval_pts[1]))
# map_error_z = np.max(np.absolute(np.array(evals[2]) - eval_pts[2]))

# # map_grad_error_x = np.max(np.absolute(np.array(grad_evals[0]) - grad_true[0]))
# # map_grad_error_y = np.max(np.absolute(np.array(grad_evals[1]) - grad_true[1]))
# # map_grad_error_z = np.max(np.absolute(np.array(grad_evals[2]) - grad_true[2]))

# l_inf.append(error)
# linf_map.append([map_error_x, map_error_y, map_error_z])
# # l_inf_grad_map.append([map_grad_error_x, map_grad_error_y, map_grad_error_z])

# print("solution error:", error)
# print("map error:", map_error_x, map_error_y, map_error_z)
# # print("grad map error:", map_grad_error_x, map_grad_error_y, map_grad_error_z)






# ===================================================================================================
# convergence tests

# def uv_sampler(U, t, dt, mesh):

#     v1 = UV_interpolant_linear(mesh, U(t, dt, mesh.vertices.T))
#     v2 = UV_interpolant_linear(mesh, U(t + dt, dt, mesh.vertices.T))
#     v3 = UV_interpolant_linear(mesh, U(t + 2*dt, dt, mesh.vertices.T))

#     return velocity.velocity_interp([v1, v2, v3], t0 = t)



# # define velocity grid: 
# ico_uv = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=4)
# mesh_uv = meshes.spherical_triangulation(ico_uv.points)

# for j in range(7):

#     ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=j)
#     mesh = meshes.spherical_triangulation(ico0.points)
#     vals = np.array([mesh.x, mesh.y, mesh.z])
#     grad_vals = np.array([utils.grad_x(mesh.vertices.T),
#                  utils.grad_y(mesh.vertices.T),
#                  utils.grad_z(mesh.vertices.T)])

#     # note this is only suitable for the icosahedral discretization
#     # largest radius for circumscribed triangle.
#     tri_size.append(np.max(ico0.edge_lengths()))

#     # define initial discretization of the map
#     mapping0 = sphere_diffeomorphism(mesh = mesh, vals = vals,
#                                     grad_vals = grad_vals)

#     Nt = 2**j + 10 # number of time steps
#     tspan = np.linspace(0, T, Nt, endpoint = False)
#     dt = tspan[1]-tspan[0]

#     # initialize empty list for remaps.
#     # memory could be pre-allocated if a known number of maps is used
#     remaps = []

#     # for initialization
#     mapping = mapping0

#     # timing in both CPU and wallclock time
#     start, start_clock = time.perf_counter(), time.process_time()
#     identity = True

#     # initialize the velocity
#     UV = uv_sampler(U, t = 0, dt = dt, mesh = mesh_uv)


#     count = 1
#     for t in tspan:
#         jr = False #"just remapped"
#         mapping = evol.advect_project_sphere(mapping, evol.RK4_proj, t, dt, UV, identity)
#         identity = False

#         if remapping and (count % n_maps == 0):
#             remaps.append(mapping)
#             mapping = mapping0
#             jr = True

#         count +=1

#         UV = uv_sampler(U, t = t + dt, dt = dt, mesh = mesh_uv)

#     finish, finish_clock = time.perf_counter(), time.process_time()

#     if jr == False:
#         remaps.append(mapping)

#     ##save maps for convergence plots
# #     file = open("data/advection_test/paper_data/test_%s_maps_icosahedral_k%s_Nt_%s_T_%s.txt" %(name, int(j), Nt, T), "wb")
# #     pickle.dump(interpolant, file)
# #     # pickle.dump(remaps, file)


#     # Compute the error 
#     if remapping:
#         evals = evol.compose_maps(remaps, eval_pts)
#     else:
#         evals = mapping(eval_pts)
#         grad_evals = mapping.eval_grad(eval_pts, evals)

#     u_num = test_func(np.array(evals).T)


#     print("wall time (s):", finish - start)
#     print("CPU time (s):", finish_clock - start_clock)

#     error = np.max(np.absolute(u_true-u_num))
#     map_error_x = np.max(np.absolute(np.array(evals[0]) - eval_pts[0]))
#     map_error_y = np.max(np.absolute(np.array(evals[1]) - eval_pts[1]))
#     map_error_z = np.max(np.absolute(np.array(evals[2]) - eval_pts[2]))

#     # map_grad_error_x = np.max(np.absolute(np.array(grad_evals[0]) - grad_true[0]))
#     # map_grad_error_y = np.max(np.absolute(np.array(grad_evals[1]) - grad_true[1]))
#     # map_grad_error_z = np.max(np.absolute(np.array(grad_evals[2]) - grad_true[2]))

#     l_inf.append(error)
#     linf_map.append([map_error_x, map_error_y, map_error_z])
#     # l_inf_grad_map.append([map_grad_error_x, map_grad_error_y, map_grad_error_z])

#     print("solution error:", error)
#     print("map error:", map_error_x, map_error_y, map_error_z)
#     # print("grad map error:", map_grad_error_x, map_grad_error_y, map_grad_error_z)
