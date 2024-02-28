# ------------------------------------------------------------------------------
"""
Basic script to test the vorticity equation solver on the sphere
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time
import pyssht as pysh
from ..core.interpolants.spherical_spline import sphere_diffeomorphism
from ..core import spherical_simulations as sphere_sim
from ..core import spherical_harmonic_tools as sph_tools
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as vel
#--------------------------------- Setup --------------------------------------
# define grid to evaluate the error
N_pts = 500
N_pts = N_pts
phis = np.linspace(0, 2*np.pi, N_pts, endpoint = False)
thetas = np.linspace(0, np.pi, N_pts, endpoint = False)
XX = np.meshgrid(phis, thetas)

vorticity = vel.rossby_wave

omega_true = vorticity(XX[0], XX[1], t = 1)
# these are used to define the velocity field.
s_points = utils.sphere2cart(XX[0],XX[1]) # the sample points of size (3, L+1, 2*L)
eval_pts = np.array([s_points[0].reshape([N_pts*N_pts,]), s_points[1].reshape([N_pts*N_pts,]),
        s_points[2].reshape([N_pts*N_pts,])])

# omg_T_lms = pysh.forward(omega_true, N_pts, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
# enst0 = np.absolute(np.sum(omg_T_lms*omg_T_lms.conjugate()))

tri_size = []
l_inf = []
l_inf_map = []
error_over_time = []
Enst = []

resolutions = [16, 32, 64, 128, 256, 512, 1024]

for k in range(8):
    # parameters for the simulation
    ico_k = k+1
    L = resolutions[k] #2**(k+1) + 10
    T = 1
    t_res = L//2
    n_maps = 20

    # pre-compute all quantities derived from the mesh-----------------
    ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=ico_k)
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
    s_evals1D_vpts = verts0.copy()

    # for the stencil points
    edges_list_spts = np.zeros([7, 4*NN,3])  
    Bcc_spts = np.zeros([4*NN,3])
    nCs_spts = np.zeros([4*NN,6], dtype = 'int64')
    Cfs_spts = np.zeros([6,4*NN])
    s_evals1D_spts = spts.copy() #stencil points about the vertices

    s_evals = np.zeros([3,4,NN])

    # initialize coefficient arrays:
    Nv = v_pts.shape[0]
    cfs_map = np.zeros([3,Nv,3])
    grad_fx = np.zeros([3,3,Nv]); grad_fy = np.zeros([3,3,Nv]); grad_fz = np.zeros([3,3,Nv])

    coeffs0 = np.zeros([3, 19, v_pts.shape[0]])

    # largest radius for circumscribed triangle.
    tri_size.append(np.max(ico0.edge_lengths())) # note this is only suitable for the icosahedral discretization
    
    # initialize the coefficients defining the interpolant:
    coeffs0 = sphere_sim.assemble_coefficients(coeffs0, inds, vals, grad_vals, cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)
    coeffs = coeffs0.copy()

    # arrays used for the querying of the velocity field:
    # size based on vertices of the map grid
    trangs_qs_vpts = np.zeros([NN,], dtype = 'int64')
    verts_qs_vpts = np.zeros([NN, 3, 3])
    phi_lv_vpts = np.zeros([NN,], dtype = int)
    theta_lv_vpts = np.zeros([NN,], dtype = int)
    vel_vals_vpts = np.zeros([3,NN])

    # for the stencil points
    trangs_qs_spts = np.zeros([4*NN,], dtype = 'int64')
    verts_qs_spts = np.zeros([4*NN, 3, 3])
    phi_lv_spts = np.zeros([4*NN,], dtype = int)
    theta_lv_spts = np.zeros([4*NN,], dtype = int)
    vel_vals_spts = np.zeros([3,4*NN])


    # intialize the grid defining the velocity field:---------------------    
    [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
    angs0 = np.meshgrid(phis, thetas)
    s_points = np.array(utils.sphere2cart(angs0[0], angs0[1]))
    sample_pts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
            s_points[2].reshape([(L+1)*2*L,])])
    
    # needed for the evaluation of the map:
    sample_pts_out = sample_pts.copy()
    angs = np.array(angs0.copy())
    
    # all arrays needed from the mesh
    N, M = len(phis), len(thetas)
    XXn = np.meshgrid(phis, thetas[1:-1])
    ico_v = meshes.spherical_mesh(XXn[0], XXn[1], N, M-2)

    simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
    simplices = np.array(simplices) 
    msimplices = np.array(msimplices) 
    
    # pre-allocate arrays for the velocity field interpolants:
    U_verts = ico_v.points
    U_v_pts = U_verts[simplices]

    # vectors along the side of the PS split triangulation
    U_Hs, U_Gs, U_Es = sphere_sim.ps_split(U_v_pts)

    Nvv = U_v_pts.shape[0]
    U_cfs_v = np.zeros([3,Nvv,3])
    U_grad_fx_v = np.zeros([3,3,Nvv]); 
    U_grad_fy_v = np.zeros([3,3,Nvv]); 
    U_grad_fz_v = np.zeros([3,3,Nvv])
    
    # define the three coefficients array defining the interpolants of the velocity field
    U_coeffs0 = np.zeros([3,19, Nvv])
    U_coeffs1 = U_coeffs0.copy() 
    U_coeffs2 = U_coeffs0.copy()

    # define arrays for the actial
    # u0[i,j,:] gives values of D_j u^i on grid points
    U_u_coeffs0 = np.zeros([3, 4, L+1, 2*L])
    U_u_coeffs1 = U_u_coeffs0.copy()
    U_u_coeffs2 = U_u_coeffs0.copy()
    U_u0 = np.zeros([3, 4, (L-1)*2*L + 2])
    U_u1 = U_u0.copy()
    U_u2 = U_u0.copy()


    edges_list_samples = np.zeros([7,(L+1)*2*L,3])  
    Bcc_samples = np.zeros([(L+1)*2*L,3])
    nCs_samples = np.zeros([(L+1)*2*L,6], dtype = 'int64')
    Cfs_samples = np.zeros([6,(L+1)*2*L]) 

    u_lms = np.zeros([3, L, 2*L+1], dtype = 'complex128')

    outs = [u_lms[0,l,m] for l in range(0,L) for m in range(0,2*l+1)]

    L1 = len(outs) # \sum_{\ell = 0}^L {\sum_{|m| \leq \ell} 
    u_lms1d = np.zeros([3, 4, L1], dtype = 'complex128')
    
    #initialize angular momentum operator:
    L_plus = np.array(np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l-1)), (L,2*L+1), dtype = 'float64'))
    L_minus = np.array(np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l+1)), (L,2*L+1), dtype = 'float64'))
    L_minus[np.where(np.isnan(L_minus))] = 0.
    L_z = np.array(np.fromfunction(lambda l,m: m-l, (L, 2*L+1), dtype = 'float64'))

    # obtain stream function
    psi0 = vorticity(angs0[0], angs0[1])
    psi_lms = sphere_sim.inv_Laplacian(psi0, L)

    # compute spherical harmonic coefficients of velocity from stream function:

    # instantiate the velocity field interpolant:
    u_lms[:] = sphere_sim.angular_momentum(psi_lms, u_lms, L_plus, L_minus, L_z)

    U_u0[:] = sphere_sim.project_onto_S12_PS_vector(u_lms, u_lms1d, U_u_coeffs0, U_u0, L, L_plus, L_minus, L_z, s_points, Method = "MWSS")

    # assemble the coefficients for the first array:
    start, start_clock = time.perf_counter(), time.process_time()

    U_coeffs0[:] = sphere_sim.assemble_coefficients(U_coeffs0, simplices, U_u0[:,0,:], U_u0[:,1::,:], 
                        U_cfs_v, U_grad_fx_v, U_grad_fy_v, U_grad_fz_v, U_Hs, U_Gs, U_Es)

    # start the intergration, bootstrapping the first two steps:

    # bootstrap first two steps for the simulation:
    # specification for the class:
    tspan = np.linspace(0, T, t_res, endpoint = False)
    dt = tspan[1]-tspan[0]
    t = 0; t0 = 0

    # evaluate velocity field:
    u_num = sphere_sim.U_interp1(U_coeffs0, t, dt, spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)

    s_evals1D_spts = sphere_sim.div_norm(spts-dt*u_num)

    vals_n, grad_vals_n[0,:,:], grad_vals_n[1,:,:], grad_vals_n[2,:,:] = sphere_sim.spline_proj_sphere(s_evals1D_spts.reshape([3,ss[1], ss[2]]), gammas)

    #redefine the coefficients from the values:
    coeffs = sphere_sim.assemble_coefficients(coeffs, inds, vals_n, grad_vals_n, cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)   

    # define next velocity field:   

    # sample the interpolant at the MW sampling points
    bcc_samples0, trangs_samples0, vpts_samples0 = sphere_sim.query(sample_pts, vertices, inds, Bcc_samples)
    sample_pts_out = sphere_sim.eval_interp(coeffs, sample_pts, bcc_samples0, trangs_samples0, 
                    vpts_samples0, nCs_samples, Cfs_samples, edges_list_samples, Bcc_samples, sample_pts_out)
  
    angs = sphere_sim.cart2sphere(angs, sample_pts_out.reshape([3,L+1, 2*L]), L)
    zeta0 = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0], angs[1]) - vel.rotating_frame(angs0[0], angs0[1])
    psi_lms1 = sphere_sim.inv_Laplacian(zeta0, L)
    
    #new velocity field
    u_lms[:] = sphere_sim.angular_momentum(psi_lms1, u_lms, L_plus, L_minus, L_z)

    U_u1[:] = sphere_sim.project_onto_S12_PS_vector(u_lms, u_lms1d, U_u_coeffs1, U_u1, L, L_plus, L_minus, L_z, s_points, Method = "MWSS")
    U_coeffs1[:] = sphere_sim.assemble_coefficients(U_coeffs1, simplices, U_u1[:,0,:], U_u1[:,1::,:], 
                    U_cfs_v, U_grad_fx_v, U_grad_fy_v, U_grad_fz_v, U_Hs, U_Gs, U_Es)
    
    #backwards in time improved Euler scheme
    # first for the vertices:
    # # step 1:
    u_num_spts1 = sphere_sim.U_interp2(U_coeffs0, U_coeffs1, t0, t, dt, spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)
    step1_spts = sphere_sim.div_norm(spts-dt*u_num_spts1)

    # #step 2:
    u_num_spts2 = sphere_sim.U_interp2(U_coeffs0, U_coeffs1, t0, t+dt, dt, step1_spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)
    
    s_evals1D_spts = sphere_sim.div_norm(spts- (dt/2)*(u_num_spts1 + u_num_spts2))

    vals_n, grad_vals_n[0,:,:], grad_vals_n[1,:,:], grad_vals_n[2,:,:] = sphere_sim.spline_proj_sphere(s_evals1D_spts.reshape([3,ss[1], ss[2]]), gammas)

    #redefine the coefficients from the values:
    coeffs = sphere_sim.assemble_coefficients(coeffs, inds, vals_n, grad_vals_n, cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)   

    sample_pts_out = sphere_sim.eval_interp(coeffs, sample_pts, bcc_samples0, trangs_samples0, 
                    vpts_samples0, nCs_samples, Cfs_samples, edges_list_samples, Bcc_samples, sample_pts_out)
  
    angs = sphere_sim.cart2sphere(angs, sample_pts_out.reshape([3,L+1, 2*L]), L)
    zeta = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0], angs[1]) - vel.rotating_frame(angs0[0], angs0[1])
    psi_lms1 = sphere_sim.inv_Laplacian(zeta, L)
    
    #new velocity field
    u_lms[:] = sphere_sim.angular_momentum(psi_lms1, u_lms, L_plus, L_minus, L_z)

    U_u1[:] = sphere_sim.project_onto_S12_PS_vector(u_lms, u_lms1d, U_u_coeffs1, U_u1, L, L_plus, L_minus, L_z, s_points, Method = "MWSS")
    U_coeffs1[:] = sphere_sim.assemble_coefficients(U_coeffs1, simplices, U_u1[:,0,:], U_u1[:,1::,:], 
                    U_cfs_v, U_grad_fx_v, U_grad_fy_v, U_grad_fz_v, U_Hs, U_Gs, U_Es)
    

    # now step from dt: --------------
    t = dt

    # first for the vertices:
    # # step 1:
    u_num_spts1 = sphere_sim.U_interp2(U_coeffs0, U_coeffs1, t0, t+dt, dt, spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)
    u_num_vpts1 = sphere_sim.U_interp2(U_coeffs0, U_coeffs1, t0, t+dt, dt, verts0, phis, thetas, U_verts, msimplices, phi_lv_vpts, theta_lv_vpts, trangs_qs_vpts, verts_qs_vpts, s_evals1D_vpts, edges_list, nCs, Bcc, Cfs)
 
    step1_spts = sphere_sim.div_norm(spts-dt*u_num_spts1)
    step1_vpts = sphere_sim.div_norm(verts0-dt*u_num_vpts1)

    # #step 2:
    u_num_spts2 = sphere_sim.U_interp2(U_coeffs0, U_coeffs1, t0, t, dt, step1_spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)
    u_num_vpts2 = sphere_sim.U_interp2(U_coeffs0, U_coeffs1, t0, t, dt, step1_vpts, phis, thetas, U_verts, msimplices, phi_lv_vpts, theta_lv_vpts, trangs_qs_vpts, verts_qs_vpts, s_evals1D_vpts, edges_list, nCs, Bcc, Cfs)
 
    step2_spts = sphere_sim.div_norm(spts- (dt/4)*(u_num_spts1 + u_num_spts2))
    step2_vpts = sphere_sim.div_norm(verts0 - (dt/4)*(u_num_vpts1 + u_num_vpts2))

    # step 3
    u_num_spts3 = sphere_sim.U_interp2(U_coeffs0, U_coeffs1, t0, t + dt/2, dt, step2_spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)
    u_num_vpts3 = sphere_sim.U_interp2(U_coeffs0, U_coeffs1, t0, t + dt/2, dt, step2_vpts, phis, thetas, U_verts, msimplices, phi_lv_vpts, theta_lv_vpts, trangs_qs_vpts, verts_qs_vpts, s_evals1D_vpts, edges_list, nCs, Bcc, Cfs)
 
    s_evals1D_spts = sphere_sim.div_norm(spts- dt*(u_num_spts1/6 + u_num_spts2/6 + 2*u_num_spts3/3))
    s_evals1D_vpts = sphere_sim.div_norm(verts0 - dt*(u_num_vpts1/6 + u_num_vpts2/6 + 2*u_num_vpts3/3))

    # projection step:
    bcc, trangs, vpts = sphere_sim.query(s_evals1D_vpts, vertices, inds, Bcc)
    s_evals = sphere_sim.stencil_eval(coeffs, s_evals1D_spts.reshape(ss), bcc, trangs, vpts, nCs, Cfs, edges_list, Bcc, s_evals)
    vals_n, grad_vals_n[0,:,:], grad_vals_n[1,:,:], grad_vals_n[2,:,:] = sphere_sim.spline_proj_sphere(s_evals.reshape([3,ss[1], ss[2]]), gammas)

    #redefine the coefficients from the values:
    coeffs = sphere_sim.assemble_coefficients(coeffs, inds, vals_n, grad_vals_n, cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)   
    # -------------------------------------------------------------------

    # define next velocity field: ---------------------------------------
    sample_pts_out = sphere_sim.eval_interp(coeffs, sample_pts, bcc_samples0, trangs_samples0, 
                    vpts_samples0, nCs_samples, Cfs_samples, edges_list_samples, Bcc_samples, sample_pts_out)
  
    angs = sphere_sim.cart2sphere(angs, sample_pts_out.reshape([3,L+1, 2*L]), L)
    zeta = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0], angs[1]) - vel.rotating_frame(angs0[0], angs0[1])
    psi_lms = sphere_sim.inv_Laplacian(zeta, L)
    
    #new velocity field
    u_lms[:] = sphere_sim.angular_momentum(psi_lms, u_lms, L_plus, L_minus, L_z)

    U_u2[:] = sphere_sim.project_onto_S12_PS_vector(u_lms, u_lms1d, U_u_coeffs2, U_u2, L, L_plus, L_minus, L_z, s_points, Method = "MWSS")
    U_coeffs2[:] = sphere_sim.assemble_coefficients(U_coeffs2, simplices, U_u2[:,0,:], U_u2[:,1::,:], 
                    U_cfs_v, U_grad_fx_v, U_grad_fy_v, U_grad_fz_v, U_Hs, U_Gs, U_Es)
    
    # --------------------------------------------------------------------------
    
    for t in tspan[2::]:

        # first for the vertices:
        # # step 1:
        u_num_spts1 = sphere_sim.U_interp3(U_coeffs0, U_coeffs1, U_coeffs2, t0, t+dt, dt, spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)
        u_num_vpts1 = sphere_sim.U_interp3(U_coeffs0, U_coeffs1, U_coeffs2, t0, t+dt, dt, verts0, phis, thetas, U_verts, msimplices, phi_lv_vpts, theta_lv_vpts, trangs_qs_vpts, verts_qs_vpts, s_evals1D_vpts, edges_list, nCs, Bcc, Cfs)
    
        step1_spts = sphere_sim.div_norm(spts-dt*u_num_spts1)
        step1_vpts = sphere_sim.div_norm(verts0-dt*u_num_vpts1)

        # #step 2:
        u_num_spts2 = sphere_sim.U_interp3(U_coeffs0, U_coeffs1, U_coeffs2, t0, t, dt, step1_spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)
        u_num_vpts2 = sphere_sim.U_interp3(U_coeffs0, U_coeffs1, U_coeffs2, t0, t, dt, step1_vpts, phis, thetas, U_verts, msimplices, phi_lv_vpts, theta_lv_vpts, trangs_qs_vpts, verts_qs_vpts, s_evals1D_vpts, edges_list, nCs, Bcc, Cfs)
    
        step2_spts = sphere_sim.div_norm(spts- (dt/4)*(u_num_spts1 + u_num_spts2))
        step2_vpts = sphere_sim.div_norm(verts0 - (dt/4)*(u_num_vpts1 + u_num_vpts2))

        # step 3
        u_num_spts3 = sphere_sim.U_interp3(U_coeffs0, U_coeffs1, U_coeffs2, t0, t + dt/2, dt, step2_spts, phis, thetas, U_verts, msimplices, phi_lv_spts, theta_lv_spts, trangs_qs_spts, verts_qs_spts, s_evals1D_spts, edges_list_spts, nCs_spts, Bcc_spts, Cfs_spts)
        u_num_vpts3 = sphere_sim.U_interp3(U_coeffs0, U_coeffs1, U_coeffs2, t0, t + dt/2, dt, step2_vpts, phis, thetas, U_verts, msimplices, phi_lv_vpts, theta_lv_vpts, trangs_qs_vpts, verts_qs_vpts, s_evals1D_vpts, edges_list, nCs, Bcc, Cfs)
    
        s_evals1D_spts = sphere_sim.div_norm(spts- dt*(u_num_spts1/6 + u_num_spts2/6 + 2*u_num_spts3/3))
        s_evals1D_vpts = sphere_sim.div_norm(verts0 - dt*(u_num_vpts1/6 + u_num_vpts2/6 + 2*u_num_vpts3/3))

        # projection step:
        bcc, trangs, vpts = sphere_sim.query(s_evals1D_vpts, vertices, inds, Bcc)
        s_evals = sphere_sim.stencil_eval(coeffs, s_evals1D_spts.reshape(ss), bcc, trangs, vpts, nCs, Cfs, edges_list, Bcc, s_evals)
        vals_n, grad_vals_n[0,:,:], grad_vals_n[1,:,:], grad_vals_n[2,:,:] = sphere_sim.spline_proj_sphere(s_evals.reshape([3,ss[1], ss[2]]), gammas)

        #redefine the coefficients from the values:
        coeffs = sphere_sim.assemble_coefficients(coeffs, inds, vals_n, grad_vals_n, cfs_map, grad_fx, grad_fy, grad_fz, Hs, Gs, Es)   
        # -------------------------------------------------------------------

        # define next velocity field: ---------------------------------------
        sample_pts_out = sphere_sim.eval_interp(coeffs, sample_pts, bcc_samples0, trangs_samples0, 
                        vpts_samples0, nCs_samples, Cfs_samples, edges_list_samples, Bcc_samples, sample_pts_out)
    
        angs = sphere_sim.cart2sphere(angs, sample_pts_out.reshape([3,L+1, 2*L]), L)
        zeta = vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0], angs[1]) - vel.rotating_frame(angs0[0], angs0[1])
        psi_lms = sphere_sim.inv_Laplacian(zeta, L)
        
        #new velocity field
        u_lms[:] = sphere_sim.angular_momentum(psi_lms, u_lms, L_plus, L_minus, L_z)
        
        # shift the velocity field array down:
        U_u0[:] = U_u1; U_u_coeffs0[:] = U_u_coeffs1; U_coeffs0[:] = U_coeffs1
        U_u1[:] = U_u2; U_u_coeffs2[:] = U_u_coeffs2; U_coeffs2[:] = U_coeffs2

        U_u2[:] = sphere_sim.project_onto_S12_PS_vector(u_lms, u_lms1d, U_u_coeffs2, U_u2, L, L_plus, L_minus, L_z, s_points, Method = "MWSS")
        U_coeffs2[:] = sphere_sim.assemble_coefficients(U_coeffs2, simplices, U_u2[:,0,:], U_u2[:,1::,:], 
                        U_cfs_v, U_grad_fx_v, U_grad_fy_v, U_grad_fz_v, U_Hs, U_Gs, U_Es)
        
        t0 = t0 + dt


    mapping = sphere_diffeomorphism(mesh, vals_n, grad_vals_n)

    finish, finish_clock = time.perf_counter(), time.process_time()

    print("wall time (s):", finish - start)
    print("CPU time (s):", finish_clock - start_clock)

    evals = mapping(eval_pts)

    angs_f = np.array(utils.cart2sphere(evals)).reshape([2,N_pts, N_pts])
    omega_num = vorticity(angs_f[0], angs_f[1]) + vel.rotating_frame(angs_f[0],angs_f[1]) - vel.rotating_frame(XX[0],XX[1])

    # omg_n_lms = pysh.forward(omega_num, N_pts, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
    # enst_error = np.absolute(np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate())) - enst0)
    # Enst.append(enst_error/enst0)
    # print("Enstrophy Error:", Enst)

    l_inf_k = np.max(np.absolute(omega_true - omega_num))
    l_inf.append(l_inf_k)
    print("L-inf error:", l_inf_k)
