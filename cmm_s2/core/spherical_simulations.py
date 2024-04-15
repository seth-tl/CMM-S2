#/----
"""
This script contains the spherical simulations 
"""
#/----
#imports
import pdb, time, scipy.io, igl, pickle
import numpy as np
import numba as nb
import pyssht as pysh
from . import utils
from . import evolution_functions as evol
from . import mesh_functions as meshes
from . import dynamical_fields as vel
from ..core.spherical_spline import sphere_diffeomorphism, composite_sphere_diffeomorphism, spline_interp_velocity
from . import spherical_harmonic_tools as sph_tools

#-------------- Spherical simulations --------------------------------------

def euler_simulation_sphere(L, Nt, T, mesh, vorticity, rot_rate = 2*np.pi):
    """
    Barotropic vorticity simulation without the submap decomposition
    Inputs (parameters of the simulation):
        - L (int): Band-limit defining the sampling of the vorticity
        - Nt (int): number of time steps
        - T (float): Final integration time
        - mesh (spherical_triangulation): mesh that the map is discretized on
        - vorticity (callable): function defining the initial vorticity
        - rot_rate: (float) defines rate of rotation (rad/s)    
    Outputs:
        - final map
    """
    #----------- Flow map interpolation set-up --------------
    jet_vals = np.zeros([3,4, len(mesh.x)])
    jet_vals[:,0,:] = np.array([mesh.x, mesh.y, mesh.z]); 
    jet_vals[:,1:,:] = np.array([utils.grad_x(mesh.vertices.T),
                    utils.grad_y(mesh.vertices.T),
                    utils.grad_z(mesh.vertices.T)])

    # largest radius for circumscribed triangle.
    # pre-allocate coefficient array
    coeffs = np.zeros([3,19, np.shape(mesh.vertices[np.array(mesh.simplices)])[0]])
    map0 = sphere_diffeomorphism(mesh, jet_vals, coeffs)
    curr_map = map0

    #---------  Initialize the velocity field--------------------------
    # define time span:
    tspan = np.linspace(0, T, Nt, endpoint = False)
    dt = tspan[1]-tspan[0]

    mesh_u = meshes.structure_spherical_triangulation(L = L)
    # (p = 3, x, 19, Nv) points needed to define interpolant pth-order time and space
    coeffs_u = np.zeros([3, 3, 19, np.shape(mesh_u.vertices[np.array(mesh_u.simplices)])[0]])
    vals_u = np.zeros([3, 3, 4, len(mesh_u.verticess)])

    # intialize the interpolant:
    U = spline_interp_velocity(mesh_u, vals_u, coeffs_u, ts = tspan[0:3])

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

    
    # this defines where the map will be sampled
    [Phi, The] = np.meshgrid(mesh_u.phi, mesh_u.theta)
    s_points = utils.sphere2cart(Phi,The)
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])]).T

    #obtain initial velocity field
    psi0 = vorticity(Phi, The)
    psi_lms = sph_tools.inv_Laplacian(psi0, L, D_inv)
    
    temp = 0*psi_lms[:] # temporary array for the projection step

    sph_tools.project_stream(psi_lms, L, L_plus, L_minus, L_z, U, 0, temp)

    # # Bootstrap first two steps:
    t = 0
    #first take a wee Euler Step
    int0 = evol.advect_project_sphere(map0, evol.Euler_step_proj, t, dt, U, identity = True)
    angs0 = utils.cart2sphere(int0(sample_pts))

    #sample the vorticity and solve for stream function
    angs0 = [angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])]
    zeta0 = vorticity(angs0[0], angs0[1]) + 2*rot_rate*np.cos(angs0[1]) - 2*rot_rate*np.cos(The)

    psi_lms1 = sph_tools.inv_Laplacian(zeta0, L, D_inv)
    #new velocity field
    sph_tools.project_stream(psi_lms1, L, L_plus, L_minus, L_z, U, 1, temp)
    U.nVs = 2
    # now repeat
    int1 = evol.advect_project_sphere(map0, evol.improved_Euler_proj, t, dt, U, identity = True)
    angs1 = utils.cart2sphere(int1(sample_pts))
    angs1 = [angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])]
    zeta1 = vorticity(angs1[0], angs1[1]) + 2*rot_rate*np.cos(angs1[1]) - 2*rot_rate*np.cos(The)
    psi_lms2 = sph_tools.inv_Laplacian(zeta1, L, D_inv)
    #new velocity field
    sph_tools.project_stream(psi_lms2, L, L_plus, L_minus, L_z, U, 1, temp)

    # and again
    int2 = evol.advect_project_sphere(int1, evol.RK4_proj, t + dt, dt, U)
    angs2 = utils.cart2sphere(int2(sample_pts))
    angs2 = [angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])]
    zeta2 = vorticity(angs2[0], angs2[1]) + 2*rot_rate*np.cos(angs2[1]) - 2*rot_rate*np.cos(The)
    psi_lms3 = sph_tools.inv_Laplacian(zeta2, L, D_inv)
    U.nVs = 3
    #new velocity field
    sph_tools.project_stream(psi_lms3, L, L_plus, L_minus, L_z, U, 2, temp)
    curr_map = int2

    # initialize the time-stepping:
    U.stepping = True
    
    for t in tspan[2:]:
        # evolution algorithm
        curr_map = evol.advect_project_sphere(curr_map, evol.RK4_proj, t, dt, U)
        XXn = curr_map(sample_pts)
        ansn = utils.cart2sphere(XXn)
        angs = [ansn[0].reshape([L+1,2*L]), ansn[1].reshape([L+1,2*L])]
        omg_n = vorticity(angs[0], angs[1]) + 2*rot_rate*np.cos(angs[1]) - 2*rot_rate*np.cos(The)
        psi_lms_n = sph_tools.inv_Laplacian(omg_n, L, D_inv)

        # new velocity field
        sph_tools.project_stream(psi_lms_n, L, L_plus, L_minus, L_z, U, 2, temp)

        #update the velocity field time_points
        U.ts = U.ts + dt

    return curr_map


def euler_simulation_sphere_remapping(L, Nt, T, mesh, vorticity, rot_rate = 2*np.pi, n_maps = 10):
    """
    Barotropic vorticity simulation without the submap decomposition
    Inputs (parameters of the simulation):
        - L (int): Band-limit defining the sampling of the vorticity
        - Nt (int): number of time steps
        - T (float): Final integration time
        - mesh (spherical_triangulation): mesh that the map is discretized on
        - vorticity (callable): function defining the initial vorticity
        - rot_rate: (float) defines rate of rotation (rad/s)    
    Outputs:
        - final map
    """
    #----------- Flow map interpolation set-up --------------

    # define the remapping points (in time)
    ns = [i for i in range(1,Nt) if i % n_maps == 0]

    # with the last map:
    ns.append(Nt)

    jet_vals = np.zeros([len(ns),3,4, len(mesh.x)])
    jet_vals[0,:,0,:] = np.array([mesh.x, mesh.y, mesh.z]); 
    jet_vals[0,:,1:,:] = np.array([utils.grad_x(mesh.vertices.T),
                    utils.grad_y(mesh.vertices.T),
                    utils.grad_z(mesh.vertices.T)])

    # largest radius for circumscribed triangle.
    # pre-allocate coefficient array
    coeffs = np.zeros([len(ns), 3, 19, np.shape(mesh.vertices[np.array(mesh.simplices)])[0]])
    map0 =  sphere_diffeomorphism(mesh, jet_vals[0].copy(), coeffs[0].copy())
    curr_map = map0

    submaps = composite_sphere_diffeomorphism(mesh, jet_vals, coeffs, ns)

    #---------  Initialize the velocity field--------------------------
    # define time span:
    tspan = np.linspace(0, T, Nt, endpoint = False)
    dt = tspan[1]-tspan[0]

    mesh_u = meshes.structure_spherical_triangulation(L = L)
    # (p = 3, x, 19, Nv) points needed to define interpolant pth-order time and space
    coeffs_u = np.zeros([3, 3, 19, np.shape(mesh_u.vertices[np.array(mesh_u.simplices)])[0]])
    vals_u = np.zeros([3, 3, 4, len(mesh_u.vertices)])

    # intialize the interpolant:
    U = spline_interp_velocity(mesh_u, vals_u, coeffs_u, ts = tspan[0:3])

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

    

    # this defines where the map will be sampled
    [Phi, The] = np.meshgrid(mesh_u.phi, mesh_u.theta)

    s_points = utils.sphere2cart(Phi,The)
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])]).T

    #obtain initial velocity field
    psi0 = vorticity(Phi, The)
    psi_lms = sph_tools.inv_Laplacian(psi0, L, D_inv)
    
    temp = 0*psi_lms[:] # temporary array for the projection step

    sph_tools.project_stream(psi_lms, L, L_plus, L_minus, L_z, U, 0, temp)

    # # Bootstrap first two steps:
    t = 0
    #first take a wee Euler Step
    int0 = evol.advect_project_sphere(map0, evol.Euler_step_proj, t, dt, U, identity = True)
    angs0 = utils.cart2sphere(int0(sample_pts))

    #sample the vorticity and solve for stream function
    angs0 = [angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])]
    zeta0 = vorticity(angs0[0], angs0[1]) + 2*rot_rate*np.cos(angs0[1]) - 2*rot_rate*np.cos(The)

    psi_lms1 = sph_tools.inv_Laplacian(zeta0, L, D_inv)
    #new velocity field
    sph_tools.project_stream(psi_lms1, L, L_plus, L_minus, L_z, U, 1, temp)
    U.nVs = 2
    # now repeat
    int1 = evol.advect_project_sphere(map0, evol.improved_Euler_proj, t, dt, U, identity = True)
    angs1 = utils.cart2sphere(int1(sample_pts))
    angs1 = [angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])]
    zeta1 = vorticity(angs1[0], angs1[1]) + 2*rot_rate*np.cos(angs1[1]) - 2*rot_rate*np.cos(The)
    psi_lms2 = sph_tools.inv_Laplacian(zeta1, L, D_inv)
    #new velocity field
    sph_tools.project_stream(psi_lms2, L, L_plus, L_minus, L_z, U, 1, temp)

    # and again
    int2 = evol.advect_project_sphere(int1, evol.RK4_proj, t + dt, dt, U)
    angs2 = utils.cart2sphere(int2(sample_pts))
    angs2 = [angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])]
    zeta2 = vorticity(angs2[0], angs2[1]) + 2*rot_rate*np.cos(angs2[1]) - 2*rot_rate*np.cos(The)
    psi_lms3 = sph_tools.inv_Laplacian(zeta2, L, D_inv)
    U.nVs = 3
    #new velocity field
    sph_tools.project_stream(psi_lms3, L, L_plus, L_minus, L_z, U, 2, temp)
    curr_map = int2

    # initialize the time-stepping:
    U.stepping = True
    identity = False
    count = 2

    for t in tspan[2:]:

        # evolution algorithm
        curr_map = evol.advect_project_sphere(curr_map, evol.RK4_proj, t, dt, U, identity = identity)
    
        XXn = submaps(curr_map(sample_pts).T).T
        ansn = utils.cart2sphere(XXn)

        angs = [ansn[0].reshape([L+1,2*L]), ansn[1].reshape([L+1,2*L])]
        omg_n = vorticity(angs[0], angs[1]) + 2*rot_rate*np.cos(angs[1]) - 2*rot_rate*np.cos(The)
        psi_lms_n = sph_tools.inv_Laplacian(omg_n, L, D_inv)

        # new velocity field
        sph_tools.project_stream(psi_lms_n, L, L_plus, L_minus, L_z, U, 2, temp)

        #update the velocity field time_points
        U.ts = U.ts + dt

        count += 1
        identity = False

        # initialize the next coefficients:
        if count in ns:
            idx = ns.index(count)
            submaps.Nc = idx + 1 # change the counter for how many submaps have been initializes
            
            # replace the submap at the index with the curr map values and initialize coefficients:
            submaps.assemble_coefficients(i = idx, vals = curr_map.vals[:])

            # reset current map to the identity
            curr_map = map0
            identity = True


    return submaps



def euler_simulation_sphere_remapping_video(L, Nt, T, mesh, vorticity, rot_rate, n_maps, file_name, save_steps):
    """
    Same as euler_simulation_sphere_remapping, except it also saves intermediate steps    
    Outputs:
        - final map
    """
    #----------- Flow map interpolation set-up --------------

    # define the remapping points (in time)
    ns = [i for i in range(1,Nt) if i % n_maps == 0]

    # with the last map:
    ns.append(Nt)

    jet_vals = np.zeros([len(ns),3,4, len(mesh.x)])
    jet_vals[0,:,0,:] = np.array([mesh.x, mesh.y, mesh.z]); 
    jet_vals[0,:,1:,:] = np.array([utils.grad_x(mesh.vertices.T),
                    utils.grad_y(mesh.vertices.T),
                    utils.grad_z(mesh.vertices.T)])

    # largest radius for circumscribed triangle.
    # pre-allocate coefficient array
    coeffs = np.zeros([len(ns), 3, 19, np.shape(mesh.vertices[np.array(mesh.simplices)])[0]])
    map0 =  sphere_diffeomorphism(mesh, jet_vals[0].copy(), coeffs[0].copy())
    curr_map = map0

    submaps = composite_sphere_diffeomorphism(mesh, jet_vals, coeffs, ns)

    #---------  Initialize the velocity field--------------------------
    # define time span:
    tspan = np.linspace(0, T, Nt, endpoint = False)
    dt = tspan[1]-tspan[0]

    mesh_u = meshes.structure_spherical_triangulation(L = L)
    # (p = 3, x, 19, Nv) points needed to define interpolant pth-order time and space
    coeffs_u = np.zeros([3, 3, 19, np.shape(mesh_u.vertices[np.array(mesh_u.simplices)])[0]])
    vals_u = np.zeros([3, 3, 4, len(mesh_u.vertices)])

    # intialize the interpolant:
    U = spline_interp_velocity(mesh_u, vals_u, coeffs_u, ts = tspan[0:3])

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

    

    # this defines where the map will be sampled
    [Phi, The] = np.meshgrid(mesh_u.phi, mesh_u.theta)

    s_points = utils.sphere2cart(Phi,The)
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])]).T

    #obtain initial velocity field
    psi0 = vorticity(Phi, The)
    psi_lms = sph_tools.inv_Laplacian(psi0, L, D_inv)
    
    temp = 0*psi_lms[:] # temporary array for the projection step

    sph_tools.project_stream(psi_lms, L, L_plus, L_minus, L_z, U, 0, temp)

    # # Bootstrap first two steps:
    t = 0
    #first take a wee Euler Step
    int0 = evol.advect_project_sphere(map0, evol.Euler_step_proj, t, dt, U, identity = True)
    angs0 = utils.cart2sphere(int0(sample_pts))

    #sample the vorticity and solve for stream function
    angs0 = [angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])]
    zeta0 = vorticity(angs0[0], angs0[1]) + 2*rot_rate*np.cos(angs0[1]) - 2*rot_rate*np.cos(The)

    psi_lms1 = sph_tools.inv_Laplacian(zeta0, L, D_inv)
    #new velocity field
    sph_tools.project_stream(psi_lms1, L, L_plus, L_minus, L_z, U, 1, temp)
    U.nVs = 2
    # now repeat
    int1 = evol.advect_project_sphere(map0, evol.improved_Euler_proj, t, dt, U, identity = True)
    angs1 = utils.cart2sphere(int1(sample_pts))
    angs1 = [angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])]
    zeta1 = vorticity(angs1[0], angs1[1]) + 2*rot_rate*np.cos(angs1[1]) - 2*rot_rate*np.cos(The)
    psi_lms2 = sph_tools.inv_Laplacian(zeta1, L, D_inv)
    #new velocity field
    sph_tools.project_stream(psi_lms2, L, L_plus, L_minus, L_z, U, 1, temp)

    # and again
    int2 = evol.advect_project_sphere(int1, evol.RK4_proj, t + dt, dt, U)
    angs2 = utils.cart2sphere(int2(sample_pts))
    angs2 = [angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])]
    zeta2 = vorticity(angs2[0], angs2[1]) + 2*rot_rate*np.cos(angs2[1]) - 2*rot_rate*np.cos(The)
    psi_lms3 = sph_tools.inv_Laplacian(zeta2, L, D_inv)
    U.nVs = 3
    #new velocity field
    sph_tools.project_stream(psi_lms3, L, L_plus, L_minus, L_z, U, 2, temp)
    curr_map = int2

    # initialize the time-stepping:
    U.stepping = True
    identity = False
    count = 2

    for t in tspan[2:]:
        print(t)
        # evolution algorithm
        curr_map = evol.advect_project_sphere(curr_map, evol.RK4_proj, t, dt, U, identity = identity)
    
        XXn = submaps(curr_map(sample_pts).T).T
        ansn = utils.cart2sphere(XXn)

        angs = [ansn[0].reshape([L+1,2*L]), ansn[1].reshape([L+1,2*L])]
        omg_n = vorticity(angs[0], angs[1]) + 2*rot_rate*np.cos(angs[1]) - 2*rot_rate*np.cos(The)
        psi_lms_n = sph_tools.inv_Laplacian(omg_n, L, D_inv)

        # new velocity field
        sph_tools.project_stream(psi_lms_n, L, L_plus, L_minus, L_z, U, 2, temp)

        #update the velocity field time_points
        U.ts = U.ts + dt

        count += 1
        identity = False

        if (count % save_steps == 0) and count not in ns:
            file = open(file_name + '%s'%count + '.txt', "wb")

            # track also the number of compositions for post-processing
            pickle.dump({"vals": curr_map.vals, "Nc": submaps.Nc}, file)
        

        # initialize the next coefficients:
        if count in ns:
            idx = ns.index(count)
            submaps.Nc = idx + 1 # change the counter for how many submaps have been initializes
            
            # replace the submap at the index with the curr map values and initialize coefficients:
            submaps.assemble_coefficients(i = idx, vals = curr_map.vals[:])

            # reset current map to the identity
            curr_map = map0
            identity = True

    file = open(file_name + '_all_maps.txt', "wb")
    pickle.dump(submaps, file)

    return 


