#/----
"""
This script contains the spherical simulations 
"""
#/----
#imports
import pdb, time, scipy.io
import numpy as np
import pyssht as pysh
from . import utils
from . import evolution_functions as evol
from . import mesh_functions as meshes
from . import dynamical_fields as vel
from .interpolants.spherical_spline import sphere_diffeomorphism
from . import spherical_harmonic_tools as sph_tools

#-------------- Spherical simulations --------------------------------------
def euler_simulation_rotating_sphere(L, Nt, T, mesh, vorticity):
    """
    Parameters:
        - L (int): Band-limit defining the sampling of the vorticity
        - Nt (int): number of time steps
        - T (float): Final integration time
        - mesh (spherical_triangulation): mesh that the map is discretized on
        - vorticity (callable): function defining the initial vorticity
    """
    #TODO: include functionality for adaptive time-stepping
    #--------- Velocity Field set-up -------------------------
    [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
    [Phi, The] = np.meshgrid(phis, thetas)
    s_points = utils.sphere2cart(Phi,The)
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])])
    #obtain stream function
    psi0 = vorticity(Phi, The)
    psi_lms = sph_tools.inv_Laplacian(psi0, L)

    #create a dictionary for the grid
    N, M = len(phis), len(thetas)
    XX = np.meshgrid(phis, thetas[1:-1])
    ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

    simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
    grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
                 "msimplices": msimplices, "mesh": ico, "sample_points": s_points}

    u0 = sph_tools.project_stream(psi_lms, L, grid = grid_dict, Method = "MWSS")

    U = vel.velocity_interp(Vs = [u0], t0 = 0)
    #----------- Flow map interpolation set-up ---------------
    vals = [mesh.x, mesh.y, mesh.z]
    grad_vals = [utils.grad_x(mesh.vertices.T),
                 utils.grad_y(mesh.vertices.T),
                 utils.grad_z(mesh.vertices.T)]

    map0 = sphere_diffeomorphism(mesh = mesh, vals = vals,
                                    grad_vals = grad_vals)
    curr_map = map0
    #----------------------------------------------------------
    tspan = np.linspace(0, T, Nt, endpoint = False)
    dt = tspan[1]-tspan[0]
    remaps = []
    # # Bootstrap first two steps:
    # initializes the current map and velocity field list
    t = 0
    #first take a lil' Euler Step
    int0 = evol.advect_project_sphere(map0, evol.Euler_step_proj, t, dt, U, identity = True)
    angs0 = utils.cart2sphere(int0(sample_pts))
    #sample the vorticity and solve for stream function
    angs0 = [angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])]
    zeta0 = vorticity(angs0[0], angs0[1]) + vel.rotating_frame(angs0[0],angs0[1]) - vel.rotating_frame(Phi,The)
    psi_lms1 = sph_tools.inv_Laplacian(zeta0, L)
    #new velocity field
    u1 = sph_tools.project_stream(psi_lms1, L, grid = grid_dict, Method = "MWSS")

    # append into the interpolant
    U.Vs.append(u1)
    # now repeat
    int1 = evol.advect_project_sphere(map0, evol.improved_Euler_proj, t, dt, U, identity = True)
    angs1 = utils.cart2sphere(int1(sample_pts))
    angs1 = [angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])]
    zeta1 = vorticity(angs1[0], angs1[1]) + vel.rotating_frame(angs1[0],angs1[1]) - vel.rotating_frame(Phi,The)
    psi_lms2 = sph_tools.inv_Laplacian(zeta1, L)
    #new velocity field
    u2 = sph_tools.project_stream(psi_lms2, L, grid = grid_dict, Method = "MWSS")
    U.Vs[1] = u2

    # and again
    int2 = evol.advect_project_sphere(int1, evol.RK4_proj, t + dt, dt, U)
    angs2 = utils.cart2sphere(int2(sample_pts))
    angs2 = [angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])]
    zeta2 = vorticity(angs2[0], angs2[1]) + vel.rotating_frame(angs2[0],angs2[1]) - vel.rotating_frame(Phi,The)
    psi_lms3 = sph_tools.inv_Laplacian(zeta2, L)
    #new velocity field
    u3 = sph_tools.project_stream(psi_lms3, L, grid = grid_dict, Method = "MWSS")
    U.Vs.append(u3)
    curr_map = int2
    #flag for whether or not we are starting from a remapping step
    just_remapped = False

    for t in tspan[2::]:
        # print(t)
        curr_map = evol.advect_project_sphere(curr_map, evol.RK4_proj,
                         t, dt, U, just_remapped)

        # XXn = utils.cart2sphere(evol.compose_maps(remaps, sample_pts, current = [curr_map]))
        XXn = curr_map(sample_pts)
        ansn = utils.cart2sphere(XXn)
        angs = [ansn[0].reshape([L+1,2*L]), ansn[1].reshape([L+1,2*L])]
        omg_n = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0],angs[1]) - vel.rotating_frame(Phi,The)
        psi_lms = sph_tools.inv_Laplacian(omg_n, L)

        # new velocity field
        u_n = sph_tools.project_stream(psi_lms, L, grid = grid_dict, Method = "MWSS")

        #update the velocity field list
        U.Vs =  [U.Vs[1], U.Vs[2], u_n]
        U.t0 = U.t0 + dt

    return curr_map

def euler_simulation_static_sphere(L, Nt, T, mesh, vorticity):
    """
    Parameters
        - L (int): Band-limit defining the sampling of the vorticity
        - Nt (int): number of time steps
        - T (float): Final integration time
        - mesh (spherical_triangulation): mesh that the map is discretized on
        - vorticity (callable): function defining the initial vorticity
        
    #TODO: include functionality for adaptive time-stepping
    """
    #--------- Velocity Field set-up -------------------------
    [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
    [Phi, The] = np.meshgrid(phis, thetas)
    s_points = utils.sphere2cart(Phi,The)
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])])
    #obtain stream function
    psi0 = vorticity(Phi, The)
    psi_lms = sph_tools.inv_Laplacian(psi0, L)

    #create a dictionary for the grid
    N, M = len(phis), len(thetas)
    XX = np.meshgrid(phis, thetas[1:-1])
    ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

    simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
    grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
                 "msimplices": msimplices, "mesh": ico, "sample_points": s_points}

    u0 = sph_tools.project_stream(psi_lms, L, grid = grid_dict, Method = "MWSS")

    U = vel.velocity_interp(Vs = [u0], t0 = 0)
    #----------- Flow map interpolation set-up ---------------
    vals = [mesh.x, mesh.y, mesh.z]
    grad_vals = [utils.grad_x(mesh.vertices.T),
                 utils.grad_y(mesh.vertices.T),
                 utils.grad_z(mesh.vertices.T)]

    map0 = sphere_diffeomorphism(mesh = mesh, vals = vals,
                                    grad_vals = grad_vals)
    curr_map = map0
    #----------------------------------------------------------
    tspan = np.linspace(0, T, Nt, endpoint = False)
    dt = tspan[1]-tspan[0]
    remaps = []
    # # Bootstrap first two steps:
    # initializes the current map and velocity field list
    t = 0
    #first take a lil' Euler Step
    int0 = evol.advect_project_sphere(map0, evol.Euler_step_proj, t, dt, U, identity = True)
    angs0 = utils.cart2sphere(int0(sample_pts))
    #sample the vorticity and solve for stream function
    angs0 = [angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])]
    zeta0 = vorticity(angs0[0], angs0[1])
    psi_lms1 = sph_tools.inv_Laplacian(zeta0, L)
    #new velocity field
    u1 = sph_tools.project_stream(psi_lms1, L, grid = grid_dict, Method = "MWSS")

    # append into the interpolant
    U.Vs.append(u1)
    # now repeat
    int1 = evol.advect_project_sphere(map0, evol.improved_Euler_proj, t, dt, U, identity = True)
    angs1 = utils.cart2sphere(int1(sample_pts))
    angs1 = [angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])]
    zeta1 = vorticity(angs1[0], angs1[1])
    psi_lms2 = sph_tools.inv_Laplacian(zeta1, L)
    #new velocity field
    u2 = sph_tools.project_stream(psi_lms2, L, grid = grid_dict, Method = "MWSS")
    U.Vs[1] = u2

    # and again
    int2 = evol.advect_project_sphere(int1, evol.RK4_proj, t + dt, dt, U)
    angs2 = utils.cart2sphere(int2(sample_pts))
    angs2 = [angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])]
    zeta2 = vorticity(angs2[0], angs2[1])
    psi_lms3 = sph_tools.inv_Laplacian(zeta2, L)
    #new velocity field
    u3 = sph_tools.project_stream(psi_lms3, L, grid = grid_dict, Method = "MWSS")
    U.Vs.append(u3)
    curr_map = int2
    #flag for whether or not we are starting from a remapping step
    just_remapped = False

    for t in tspan[2::]:
        print(t)
        curr_map = evol.advect_project_sphere(curr_map, evol.RK4_proj,
                         t, dt, U, just_remapped)

        XXn = utils.cart2sphere(evol.compose_maps(remaps, sample_pts, current = [curr_map]))
        angs = [XXn[0].reshape([L+1,2*L]), XXn[1].reshape([L+1,2*L])]
        omg_n = vorticity(angs[0], angs[1])
        psi_lms = sph_tools.inv_Laplacian(zeta, L)

        # new velocity field
        u_n = sph_tools.project_stream(psi_lms, L, grid = grid_dict, Method = "MWSS")

        #update the velocity field list
        U.Vs =  [U.Vs[1], U.Vs[2], u_n]
        U.t0 = U.t0 + dt

    return curr_map

def euler_simulation_rotating_sphere_remapping(L, Nt, T, n_maps, mesh, vorticity):
    """
    Parameters:
        - L (int): Band-limit defining the sampling of the vorticity
        - Nt (int): number of time steps
        - T (float): Final integration time
        - n_maps (int): remap every n_map steps
        - mesh (spherical_triangulation): mesh that the map is discretized on
        - vorticity (callable): function defining the initial vorticity
    """
    #TODO: include functionality for adaptive time-stepping
    #--------- Velocity Field set-up -------------------------
    [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
    [Phi, The] = np.meshgrid(phis, thetas)
    s_points = utils.sphere2cart(Phi,The)
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])])
    #obtain stream function
    psi0 = vorticity(Phi, The)
    psi_lms = sph_tools.inv_Laplacian(psi0, L)

    #create a dictionary for the grid
    N, M = len(phis), len(thetas)
    XX = np.meshgrid(phis, thetas[1:-1])
    ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

    simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
    grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
                 "msimplices": msimplices, "mesh": ico, "sample_points": s_points}

    u0 = sph_tools.project_stream(psi_lms, L, grid = grid_dict, Method = "MWSS")

    U = vel.velocity_interp(Vs = [u0], t0 = 0)
    #----------- Flow map interpolation set-up ---------------
    vals = [mesh.x, mesh.y, mesh.z]
    grad_vals = [utils.grad_x(mesh.vertices.T),
                 utils.grad_y(mesh.vertices.T),
                 utils.grad_z(mesh.vertices.T)]

    map0 = sphere_diffeomorphism(mesh = mesh, vals = vals,
                                    grad_vals = grad_vals)
    curr_map = map0
    #----------------------------------------------------------
    tspan = np.linspace(0, T, Nt, endpoint = False)
    dt = tspan[1]-tspan[0]
    remaps = []
    # # Bootstrap first two steps:
    # initializes the current map and velocity field list
    t = 0
    #first take a lil' Euler Step
    int0 = evol.advect_project_sphere(map0, evol.Euler_step_proj, t, dt, U, identity = True)
    angs0 = utils.cart2sphere(int0(sample_pts))
    #sample the vorticity and solve for stream function
    angs0 = [angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])]
    zeta0 = vorticity(angs0[0], angs0[1]) + vel.rotating_frame(angs0[0],angs0[1]) - vel.rotating_frame(Phi,The)
    psi_lms1 = sph_tools.inv_Laplacian(zeta0, L)
    #new velocity field
    u1 = sph_tools.project_stream(psi_lms1, L, grid = grid_dict, Method = "MWSS")

    # append into the interpolant
    U.Vs.append(u1)
    # now repeat
    int1 = evol.advect_project_sphere(map0, evol.improved_Euler_proj, t, dt, U, identity = True)
    angs1 = utils.cart2sphere(int1(sample_pts))
    angs1 = [angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])]
    zeta1 = vorticity(angs1[0], angs1[1]) + vel.rotating_frame(angs1[0],angs1[1]) - vel.rotating_frame(Phi,The)
    psi_lms2 = sph_tools.inv_Laplacian(zeta1, L)
    #new velocity field
    u2 = sph_tools.project_stream(psi_lms2, L, grid = grid_dict, Method = "MWSS")
    U.Vs[1] = u2

    # and again
    int2 = evol.advect_project_sphere(int1, evol.RK4_proj, t + dt, dt, U)
    angs2 = utils.cart2sphere(int2(sample_pts))
    angs2 = [angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])]
    zeta2 = vorticity(angs2[0], angs2[1]) + vel.rotating_frame(angs2[0],angs2[1]) - vel.rotating_frame(Phi,The)
    psi_lms3 = sph_tools.inv_Laplacian(zeta2, L)
    #new velocity field
    u3 = sph_tools.project_stream(psi_lms3, L, grid = grid_dict, Method = "MWSS")
    U.Vs.append(u3)
    curr_map = int2
    #flag for whether or not we are starting from a remapping step
    remaps = []
    count = 2
    just_remapped = False

    for t in tspan[2::]:
        # print(t)
        curr_map = evol.advect_project_sphere(curr_map, evol.RK4_proj,
                         t, dt, U, just_remapped)

        # XXn = utils.cart2sphere(evol.compose_maps(remaps, sample_pts, current = [curr_map]))
        XXn = evol.compose_maps(remaps, sample_pts, current = [curr_map])
        ansn = utils.cart2sphere(XXn)
        angs = [ansn[0].reshape([L+1,2*L]), ansn[1].reshape([L+1,2*L])]
        omg_n = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0],angs[1]) - vel.rotating_frame(Phi,The)
        psi_lms = sph_tools.inv_Laplacian(omg_n, L)

        # new velocity field
        u_n = sph_tools.project_stream(psi_lms, L, grid = grid_dict, Method = "MWSS")

        #update the velocity field list
        U.Vs =  [U.Vs[1], U.Vs[2], u_n]
        U.t0 = U.t0 + dt
        count +=1
        just_remapped = False

        if count % n_maps == 0:
            remaps.append(curr_map)
            curr_map = map0
            just_remapped = True

    if just_remapped == False:
        remaps.append(curr_map)


    return remaps

class euler_simulation_sphere_parallel(object):

    def __init__(self, L, T, mesh, vorticity, rotating = False):
        """
        sets up an iterable to perform the Euler simulation with parallelization
        of the map evaluation.
        Parameters
            - L (int): Band-limit defining the sampling of the vorticity
            - T (float): Final integration time
            - mesh (spherical_triangulation): mesh that the map is discretized on
            - vorticity (callable): function defining the initial vorticity
        """
        #TODO: include functionality for adaptive time-stepping
        #--------- Velocity Field set-up -------------------------
        self.L = L
        [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
        [Phi, The] = np.meshgrid(phis, thetas)
        s_points = utils.sphere2cart(Phi,The)
        spts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
                s_points[2].reshape([(L+1)*2*L,])])
        self.spts = spts
        #obtain stream function
        psi0 = vorticity(Phi, The)
        psi_lms = sph_tools.inv_Laplacian(psi0, L)

        #create a dictionary for the grid
        N, M = len(phis), len(thetas)
        XX = np.meshgrid(phis, thetas[1:-1])
        ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

        simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
        self.grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
                     "msimplices": msimplices, "mesh": ico, "sample_points": s_points}

        u0 = sph_tools.project_stream(psi_lms, L, grid = self.grid_dict, Method = "MWSS")
        self.rotating = rotating

        if rotating:
            self.U = vel.velocity_interp_rotating(Vs = [u0], t0 = 0, T = T)
        else:
            self.U = vel.velocity_interp(Vs = [u0], t0 = 0)
        #----------- Flow map interpolation set-up ---------------
        vals = [mesh.x, mesh.y, mesh.z]
        grad_vals = [utils.grad_x(mesh.vertices.T),
                     utils.grad_y(mesh.vertices.T),
                     utils.grad_z(mesh.vertices.T)]

        # define initial discretization of the map
        self.map0 = sphere_diffeomorphism(mesh = mesh, vals = vals,
                                        grad_vals = grad_vals)
        self.curr_map = self.map0
        #----------------------------------------------------------
        self.tspan = np.linspace(0, T, t_res, endpoint = False)
        self.dt = tspan[1]-tspan[0]

        self.remaps = []
        self.t_ind = 2
        # # Bootstrap first two steps:
        # initializes the current map
        self.initialize(0, self.dt, self.map0, vorticity, self.U, self.grid_dict, spts, Phi, The, L, T)

        return

    def step(self, Nt = 1):
        # void function which steps through the current state
        # remapping criterion is meant to be checked outside of this function
        for t in self.tspan[self.t_ind:Nt]:
            self.curr_map = evol.advect_project_sphere(self.curr_map, evol.RK4_proj,
                             t, self.dt, self.U, self.identity)

            XXn = utils.cart2sphere(evol.compose_maps(remaps, spts, current = interpolant))
            angs = [XXn[0].reshape([self.L+1,2*self.L]), XXn[1].reshape([self.L+1,2*self.L])]

            if self.rotating:
                omg_n = self.vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0],angs[1])
                zeta = omg_n - vel.rotating_frame(Phi,The)
                psi_lms = sph_tools.inv_Laplacian(zeta, L)

                # new velocity field
                u_n = sph_tools.project_stream(psi_lms, L, grid = self.grid_dict, Method = "MWSS")
                self.U = vel.velocity_interp_rotating(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, T = T)
                return
            else:
                omg_n = self.vorticity(angs[0], angs[1])
                psi_lms = sph_tools.inv_Laplacian(omg_n, L)

                # new velocity field
                u_n = sph_tools.project_stream(psi_lms, L, grid = self.grid_dict, Method = "MWSS")
                self.U = vel.velocity_interp(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, T = T)
                return

    def initialize(self):
        # void function to initialize first two steps of the simulation.
        t = 0
        #first take a lil' Euler Step
        int0 = evol.advect_project_sphere(self.map0, evol.Euler_step_proj, t, self.dt, self.U, identity = True)
        angs0 = utils.cart2sphere(int0.eval(spts))
        L = self.L
        #sample the vorticity and solve for stream function
        if self.rotating:
            angs0 = [angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])]
            zeta0 = self.vorticity(angs0[0], angs0[1]) + rotating_frame(angs0[0],angs0[1]) - rotating_frame(Phi,The)
            psi_lms1 = sph_tools.inv_Laplacian(zeta0, L)
            #new velocity field
            u1 = sph_tools.project_stream(psi_lms1, L, grid = self.grid_dict, Method = "MWSS")

            # append into the interpolant
            self.U.Vs.append(u1)
            # now repeat
            int1 = evol.advect_project_sphere(self.map0, evol.improved_Euler_proj, t, self.dt, self.U, identity = True)
            angs1 = utils.cart2sphere(int1(spts))
            angs1 = [angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])]
            zeta1 = self.vorticity(angs1[0], angs1[1]) + rotating_frame(angs1[0],angs1[1]) - rotating_frame(Phi,The)
            psi_lms2 = sph_tools.inv_Laplacian(zeta1, L)
            #new velocity field
            u2 = sph_tools.project_stream(psi_lms2, L, grid = self.grid_dict, Method = "MWSS")
            self.U.Vs[1] = u2

            # and again
            int2 = evol.advect_project_sphere(int1, evolv.RK4_proj, t + self.dt, self.dt, self.U)
            angs2 = utils.cart2sphere(int2(spts))
            angs2 = [angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])]
            zeta2 = self.vorticity(angs2[0], angs2[1]) + rotating_frame(angs2[0],angs2[1]) - rotating_frame(Phi,The)
            psi_lms3 = sph_tools.inv_Laplacian(zeta2, L)
            #new velocity field
            u3 = sph_tools.project_stream(psi_lms3, self.L, grid = self.grid_dict, Method = "MWSS")
            self.U.Vs.append(u3)
            self.curr_map = int2

            return
        else:
            psi_lms1 = sph_tools.inv_Laplacian(self.vorticity(angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])), L)
            #new velocity field
            u1 = sph_tools.project_stream(psi_lms1, self.L, grid = self.grid_dict, Method = "MWSS")

            # append into the interpolant
            self.U.Vs.append(u1)
            # now repeat
            int1 = evol.advect_project_sphere(self.map0, evol.improved_Euler_proj, t, self.dt, self.U, identity = True)
            angs1 = utils.cart2sphere(int1(spts))
            #sample the vorticity and solve for stream function
            psi_lms2 = sph_tools.inv_Laplacian(self.vorticity(angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])), L)
            #new velocity field
            u2 = sph_tools.project_stream(psi_lms2, self.L, grid = self.grid_dict, Method = "MWSS")
            self.U.Vs[1] = u2

            # and again
            int2 = evol.advect_project_sphere(int1, evolv.RK4_proj, t + self.dt, self.dt, self.U)
            angs2 = utils.cart2sphere(int2.eval(spts))
            #sample the vorticity and solve for stream function
            psi_lms3 = sph_tools.inv_Laplacian(self.vorticity(angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])), L)
            #new velocity field
            u3 = sph_tools.project_stream(psi_lms3, self.L, grid = self.grid_dict, Method = "MWSS")
            self.U.Vs.append(u3)
            self.curr_map = int2

        return



