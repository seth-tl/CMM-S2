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
from ..core.spherical_spline import sphere_diffeomorphism
from . import spherical_harmonic_tools as sph_tools

#-------------- Spherical simulations --------------------------------------


# all functions needed for the evolution: =========================

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def grad_HBB(Cfs, bcc_n, Bs):
    bcc_x, bcc_y, bcc_z = Bs[0], Bs[1], Bs[2]

    dp_b1 = 2*Cfs[:,0]*bcc_n[:,0] + 2*Cfs[:,3]*bcc_n[:,1] + 2*Cfs[:,5]*bcc_n[:,2]

    dp_b2 = 2*Cfs[:,1]*bcc_n[:,1] + 2*Cfs[:,3]*bcc_n[:,0] + 2*Cfs[:,4]*bcc_n[:,2]

    dp_b3 = 2*Cfs[:,2]*bcc_n[:,2] + 2*Cfs[:,4]*bcc_n[:,1] + 2*Cfs[:,5]*bcc_n[:,0]

    outx = bcc_x[:,0]*dp_b1 +  bcc_x[:,1]*dp_b2 + bcc_x[:,2]*dp_b3
    outy = bcc_y[:,0]*dp_b1 +  bcc_y[:,1]*dp_b2 + bcc_y[:,2]*dp_b3
    outz = bcc_z[:,0]*dp_b1 +  bcc_z[:,1]*dp_b2 + bcc_z[:,2]*dp_b3

    return [outx, outy, outz]

def det_vec(v_1, v_2, v_3):
    det = v_1[:,0]*v_2[:,1]*v_3[:,2] + v_2[:,0]*v_3[:,1]*v_1[:,2] + v_1[:,1]*v_2[:,2]*v_3[:,0] - \
          (v_3[:,0]*v_2[:,1]*v_1[:,2] + v_3[:,1]*v_2[:,2]*v_1[:,0] + v_2[:,0]*v_1[:,1]*v_3[:,2])
    return det

def bary_coords(v_1,v_2,v_3,v, bcc_outs):
    """
    v1, v2, v3 define vertices of the containing triangle
    order counter-clockwise. v is the query point.

    # in-place operation on bcc_outs
    """
    denom = det_vec(v_1, v_2, v_3)
    bcc_outs[:,0] = det_vec(v, v_2, v_3)/denom
    bcc_outs[:,1] = det_vec(v_1,v,v_3)/denom
    bcc_outs[:,2] = det_vec(v_1, v_2, v)/denom

    return bcc_outs


#Coefficient list, this is needed for some silly indexing.
Cs = np.array([[[0,0,0,0,0,0], [18,1,13,7,6,16], [13,2,18,11,10,16]],
      [[18,14,0,17,5,4], [0,0,0,0,0,0], [14,18,2,17,10,9]],
       [[0,12,18,3,15,4],[1,18,12,7,15,8],[0,0,0,0,0,0]]])

edges_ps = np.array([[[0,0,0],[3,1,5], [5,2,3]],
                    [[3,6,0], [0,0,0],[6,3,2]],
                    [[0,4,3], [1,3,4], [0,0,0]]])

def div_norm2(x):
    #in-place operation
    Norm = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
    x[:,0] = x[:,0]/Norm
    x[:,1] = x[:,1]/Norm
    x[:,2] = x[:,2]/Norm

    return x

def ps_split_eval(v_pts_n, nCs, q_pts, coeffs, bcc, Cfs):
    # evaluation of the Powell-Sabin split
    inds = range(len(nCs))

    # for i in range(len(Cfs[0,:])):
    #     Cfs[:,i] = coeffs[nCs[i,:],i]
    Cfs[:] = coeffs[nCs[inds,:],inds]

    # update the bcc list
    bcc_n = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts, bcc)

    # formula for the evaluation of a quadratic Berstein-B{\'e}zier polynomial
    evals = Cfs[0,:]*(bcc_n[:,0]**2) + Cfs[1,:]*(bcc_n[:,1]**2) \
            + Cfs[2,:]*(bcc_n[:,2]**2) + 2*Cfs[3,:]*bcc_n[:,0]*bcc_n[:,1] \
            + 2*Cfs[4,:]*bcc_n[:,1]*bcc[:,2] + 2*Cfs[5,:]*bcc_n[:,0]*bcc_n[:,2]

    return evals

def eval_interp(coeffs, q_pts, bcc, trangs, v_pts, nCs, Cfs, EE, Bcc, eval_outs):

    #assign edge list
    EE[0] = v_pts[:,0,:]; EE[1] = v_pts[:,1,:]; EE[2] = v_pts[:,2,:]
    EE[3] = div_norm2((v_pts[:,0,:] + v_pts[:,1,:] + v_pts[:,2,:])/3) 
    EE[4] = div_norm2(v_pts[:,0,:]/2 + v_pts[:,1,:]/2) 
    EE[5] = div_norm2(v_pts[:,1,:]/2 + v_pts[:,2,:]/2) 
    EE[6] = div_norm2(v_pts[:,2,:]/2 + v_pts[:,0,:]/2)

    v_pts_n = np.zeros(v_pts.shape)
    b_min = np.argmin(bcc, axis = 1); b_max = np.argmax(bcc, axis = 1)

    ee = edges_ps[b_min, b_max] 
    v_pts_n[:,0,:] = EE[ee[0],:,:]
    v_pts_n[:,1,:] = EE[ee[1],:,:]
    v_pts_n[:,2,:] = EE[ee[2],:,:]
    nCs[:,:] = Cs[b_min, b_max]

    cfs_n = coeffs[:,:,trangs]
    for i in range(3):
        eval_outs[i] = ps_split_eval(v_pts_n, nCs, q_pts.T, coeffs = cfs_n[i,:,:], bcc = Bcc, Cfs = Cfs)

    return eval_outs

def stencil_eval(coeffs, st_pts, bcc, trangs, v_pts, nCs, Cfs, EE, Bcc, s_evals):
    # queries at q_pts and evaluates using small extrapolation at st_pts.

    #assign edge list
    EE[0] = v_pts[:,0,:]; EE[1] = v_pts[:,1,:]; EE[2] = v_pts[:,2,:]
    EE[3] = div_norm2((v_pts[:,0,:] + v_pts[:,1,:] + v_pts[:,2,:])/3) 
    EE[4] = div_norm2(v_pts[:,0,:]/2 + v_pts[:,1,:]/2) 
    EE[5] = div_norm2(v_pts[:,1,:]/2 + v_pts[:,2,:]/2) 
    EE[6] = div_norm2(v_pts[:,2,:]/2 + v_pts[:,0,:]/2)

    v_pts_n = np.zeros(v_pts.shape)
    b_min = np.argmin(bcc, axis = 1); b_max = np.argmax(bcc, axis = 1)

    ee = edges_ps[b_min, b_max] 
    v_pts_n[:,0,:] = EE[ee[0],:,:]
    v_pts_n[:,1,:] = EE[ee[1],:,:]
    v_pts_n[:,2,:] = EE[ee[2],:,:]
    nCs[:,:] = Cs[b_min, b_max]

    cfs_n = coeffs[:,:,trangs]

    # TODO: vectorize this operation
    for i in range(3):
        for j in range(4):
            s_evals[i,j,:] = ps_split_eval(v_pts_n, nCs, q_pts = st_pts[:,j,:].T, coeffs = cfs_n[i,:,:], bcc = Bcc, Cfs = Cfs)

    return s_evals

def epsilon_diff4(evals, gammas, eps):
    # form data for the integrated foot points
    # average stencilling
    vals = (evals[:,0,:] + evals[:,1,:] + evals[:,2,:] + evals[:,3,:])/4

    #partial derivative in pre-computed orthonormal basis
    df_dx1 = (evals[:,1,:] - evals[:,0,:] + evals[:,3,:] - evals[:,2,:])/(4*eps)
    df_dx2 = (evals[:,2,:] - evals[:,0,:] + evals[:,3,:] - evals[:,1,:])/(4*eps)

    # re-express in Cartesian coordinates and arrange appropriately
    g_val1 = (df_dx1[0][:,None]*gammas[0] + df_dx2[0][:,None]*gammas[1]).T
    g_val2 = (df_dx1[1][:,None]*gammas[0] + df_dx2[1][:,None]*gammas[1]).T
    g_val3 = (df_dx1[2][:,None]*gammas[0] + df_dx2[2][:,None]*gammas[1]).T

    return vals, g_val1, g_val2, g_val3

def spline_proj_sphere(s_pts, gammas, eps= 1e-5):
    # Projection step for each map, updates values and gradient_values of interpolant
    vals, g_val1, g_val2, g_val3 = epsilon_diff4(s_pts, gammas, eps)

    #normalize
    vals = div_norm(vals)
    #update the values of the interpolant:
    return vals, g_val1, g_val2, g_val3

@nb.jit(nopython = True)
def RK4_proj(t,dt,xn,V):
    #backwards RK4 combined with projection step
    tpdt = t + dt

    v1 = V(tpdt, dt, xn)
    step1 = div_norm(xn-dt*v1/2)

    v2 = V(t+dt/2, dt, step1)
    step2 = div_norm(xn - (dt/2)*v2)

    v3 = V(t+0.5*dt, dt, step2)

    step3 = div_norm(xn - dt*v3)
    v4 = V(t, dt, step3)

    xn1 = xn - (dt/6)*(v1 + 2*v2 + 2*v3 + v4)

    return div_norm(xn1)

@nb.jit(nopython = True)
def div_norm(x):
    # projection onto sphere for integrators, accepts more general arrays
    normalize = 1/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return normalize[None,:]*x

# @nb.jit(nopython = True)
def query(q_pts, vertices, simplices, Bcc):
    """
    Querying routine to find containing triangle for the query points
    Note: this is the optimized form I found given my resources
    and could/should be optimized further offering significant improvements
    to overall performance.
    TODO: Optimize the performance of this routine

    input:  q_pts - (N,3) array of query points
    output: bcc - (N,3) array of barycentric coordinates
            trangs - (N,) list of triangle indices in self.simplices
            vs - (N,3,3) array of corresponding vertex coordinates
    """
    # with nb.objmode(bcc = "Array(float64, 2, 'F', False, aligned=True)", trangs = "Array(int64, 1, 'C', False, aligned=True)",
    #                 vs = "Array(float64, 3, 'C', False, aligned=True)"):
    # obtain containing triangle from the igl package helper function
    tri_data = igl.point_mesh_squared_distance(q_pts.T, vertices, simplices)
    # second output is the containing triangle index
    trangs = tri_data[1]

    # compute other relevant quantities
    tri_out = simplices[trangs.reshape(-1)]
    vs = vertices[tri_out]
    bcc = bary_coords(vs[:,0,:],vs[:,1,:], vs[:,2,:], q_pts.T, Bcc)

    return bcc, trangs, vs

# @nb.jit(nopython = True)
# def query(mesh, q_pts):
#     with nb.objmode():
#         bcc, trangs, v_pts = mesh.query(q_pts)
#     return bcc, trangs, v_pts


@nb.jit(nopython = True)
def PS_split_coeffs(Hs, Gs, Es, c123, grad_f, coeffs):
    """
    Function which returns the 19 coefficients defining the S^1_2(T_{PS})
    spherical spline interpolant
    """
    g12, g21, g23, g32, g13, g31, g14, g24, g34, A = Gs[0], Gs[1], Gs[2], Gs[3], Gs[4], Gs[5], Gs[6], Gs[7], Gs[8], Gs[9]
    g_1, g_2, g_3 = Es[0], Es[1], Es[2]

    #obtain coefficients and combine appropriately
    coeffs[0,:] =  c123[:,0] 
    coeffs[1,:] =  c123[:,1]
    coeffs[2,:] =  c123[:,2]

    coeffs[3,:] = (dot(Hs[0], grad_f[0])/2 - g12[:,0]*c123[:,0])/g12[:,1]
    coeffs[4,:] = (dot(Hs[6], grad_f[0])/2 - g14[:,0]*c123[:,0])/g14[:,1]
    coeffs[5,:] = (dot(Hs[4], grad_f[0])/2 - g13[:,0]*c123[:,0])/g13[:,2]

    coeffs[6,:] = (dot(Hs[2], grad_f[1])/2 - g23[:,0]*c123[:,1])/g23[:,1]
    coeffs[7,:] = (dot(Hs[7], grad_f[1])/2 - g24[:,0]*c123[:,1])/g24[:,1]
    coeffs[8,:] = (dot(Hs[1], grad_f[1])/2 - g21[:,0]*c123[:,1])/g21[:,2]

    coeffs[9,:] = (dot(Hs[5], grad_f[2])/2 - g31[:,0]*c123[:,2])/g31[:,1]
    coeffs[10,:] = (dot(Hs[8], grad_f[2])/2 - g34[:,0]*c123[:,2])/g34[:,1]
    coeffs[11,:] = (dot(Hs[3], grad_f[2])/2 - g32[:,0]*c123[:,2])/g32[:,2]


    coeffs[12,:] = g_1[:,0]*coeffs[3,:] + g_1[:,1]*coeffs[8,:]
    coeffs[13,:] = g_2[:,1]*coeffs[6,:] + g_2[:,2]*coeffs[11,:]
    coeffs[14,:] = g_3[:,2]*coeffs[9,:] + g_3[:,0]*coeffs[5,:]
    coeffs[15,:] = g_1[:,0]*coeffs[4,:] + g_1[:,1]*coeffs[7,:]
    coeffs[16,:] = g_2[:,1]*coeffs[7,:] + g_2[:,2]*coeffs[10,:]
    coeffs[17,:] = g_3[:,0]*coeffs[4,:] + g_3[:,2]*coeffs[10,:]

    #barycentre coords of middle points:
    coeffs[18,:] = A[:,0]*coeffs[4,:] + A[:,1]*coeffs[7,:] + A[:,2]*coeffs[10,:]

    return coeffs


@nb.jit(nopython = True)
def assemble_coefficients(coeffs, inds, vals, grad_vals, cfs, grad_fx, grad_fy, grad_fz, Hs, Gs, Es):
    """
    assemble all the coefficients to perform the PS split
    interpolation.
    # TODO: find clever way to instatiate these arrays directly by slicing
    """
    #assign new values to the arrays
    for i in range(3):
        for j in range(len(inds[:,0])):
            grad_fx[i,:,j] =  grad_vals[0,:,inds[j,i]].T
            grad_fy[i,:,j] =  grad_vals[1,:,inds[j,i]].T
            grad_fz[i,:,j] =  grad_vals[2,:,inds[j,i]].T
            cfs[0,j,i] = vals[0, inds[j,i]]
            cfs[1,j,i] = vals[1, inds[j,i]]
            cfs[2,j,i] = vals[2, inds[j,i]]


    coeffs[0,:,:] = PS_split_coeffs(Hs, Gs, Es, c123 =  cfs[0,:,:], grad_f = grad_fx, coeffs = coeffs[0,:,:])
    
    coeffs[1,:,:] = PS_split_coeffs(Hs, Gs, Es, c123 =  cfs[1,:,:], grad_f = grad_fy, coeffs = coeffs[1,:,:])
    
    coeffs[2,:,:] = PS_split_coeffs(Hs, Gs, Es, c123 =  cfs[2,:,:], grad_f = grad_fz, coeffs = coeffs[2,:,:])


    return coeffs

# spectral-type functions =================================================================

def Laplacian(A,L):
    # computes the spherical Laplacian of the coefficient array A
    # A should be an (L,2L+1) array of coefficients, L is the band-limit
    els = np.arange(0,L)
    L = -els*(els + 1)
    return A*L[:,None]

def Poisson_Solve(f,L):
    # multiplies f in Fourier space by the inverse Laplacian.
    # input: f - (2,l_max + 1, l_max + 1) array
    els = np.arange(0,L)
    #zero out first mode
    els[0] = 1e16
    L_inv = -1/(els*(els + 1))

    return f*L_inv[:,None]

def inv_Laplacian(samples,L, Method = "MWSS", Spin = 0):
    # helper function for barotropic vorticity projection
    f_lm = pysh.forward(samples, L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 8)
    f_lm_coeffs = coeff_array(f_lm, L)
    L_flm = Poisson_Solve(f_lm_coeffs,L)

    return L_flm


def MW_sampling(L, method = "MWSS"):
    # Sampling defined by the sampling theorem of McEwen and Wiaux on the sphere
    # input: Bandlimit L
    # output: meshgrid
    [thetas, phis] = pysh.sample_positions(L, Method = method, Grid = False)
    [Phi, The] = np.meshgrid(phis, thetas)

    return [Phi, The]

def coeff_array(alms, L):
    # helper function for computations
    # input: 1-D array of coefficients
    # output: (L, 2L+1) array of coefficients
    outs = np.zeros([L, 2*L+1], dtype = 'complex128')
    for l in range(0,L):
        for m in range(0,2*l+1):
            #map m index to approriate index in array
            mm = m-l
            ind = pysh.elm2ind(l,mm)
            outs[l,m] = alms[ind]

    return outs

# def lm_array(coeffs,L):
#     # converts coefficient array back into approriate 1D array for SHHT
#     outs = [coeffs[l,m] for l in range(0,L) for m in range(0,2*l+1)]
#     return np.array([coeffs[l,m] for l in range(0,L) for m in range(0,2*l+1)])

# #Ladder operator computation
# # @nb.jit(nopython = True)
# def raise_op(alms, L, L_p):
#     # raising operator L_+ applied to the coefficients of a band-limited function
#     # with coefficients alm.
#     #Construct matrix based on band-limit L
#     pdb.set_trace()
#     outs = np.zeros([L, 2*L+1],  dtype = 'complex128')
#     outs[:,1::] = alms[:,0:-1]
        
#     return np.multiply(L_p, outs)

# # @nb.jit(nopython = True)
# def lower_op(alms, L, L_m):
#     # lowering operator L_- applied to the coefficients of a band-limited function
#     # with coefficients alm.
#     outs = np.zeros([L, 2*L+1],  dtype = 'complex128')
#     outs[:,0:-1] = alms[:,1::]

#     return np.multiply(L_m, outs)

@nb.jit(nopython = True)
def angular_momentum(alms, u_lms, L_plus, L_minus, L_z):
    
    temp1 = alms.copy()
    temp1[:,1::] = alms[:,0:-1]
    temp1[:,0] = 1j*0.

    Lp = np.multiply(L_plus, temp1)
    temp1[:,0:-1] = alms[:,1::].copy()
    temp1[:,-1] = 1j*0.

    Lm = np.multiply(L_minus, temp1)

    # angular momentum operator (r \times \nabla) applied to sph_harm coeffs
    u_lms[0,:,:] = 1j*(Lp + Lm)/2
    u_lms[1,:,:] = -(Lm - Lp)/2
    u_lms[2,:,:] = 1j*np.multiply(L_z, alms)

    return u_lms

@nb.jit(nopython = True)
def cross(output, A, B):
    # A X B into output
    output[0] = A[1]*B[2] - A[2]*B[1]
    output[1] = -A[0]*B[2] + A[2]*B[0]
    output[2] = A[0]*B[1] - A[1]*B[0]

    return output

def tan_proj(output, X, A):

    output[0] = (1-X[0]**2)*A[0] + (-X[0]*X[1])*A[1] + (-X[0]*X[2])*A[2]
    output[1] = (-X[0]*X[1])*A[0] + (1-X[1]**2)*A[1] + (-X[1]*X[2])*A[2]
    output[2] = (-X[0]*X[2])*A[0] + (-X[1]*X[2])*A[1] + (1-X[2]**2)*A[2]

    return output


def prune(A):
    # helper function for projection operation
    outs = A[1:-1,:].real
    outs = list(outs.reshape([len(outs[0,:])*len(outs[:,0]),]))
    outs.append(A[-1,0])
    outs.insert(0,A[0,0])

    return outs
# ================================================================================== 


def ps_split(v_pts):
        v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
        v4 = utils.div_norm((v1 + v2 + v3)/3).T
        e1, e2, e3 = utils.div_norm(v1/2+v2/2).T, utils.div_norm(v2/2+v3/2).T, utils.div_norm(v3/2 + v1/2).T

        # Hs correspond to tangential projection of vectors along the split
        h12, h21, h23 = meshes.sphere_tan_proj(v2-v1,v1).T, meshes.sphere_tan_proj(v1-v2,v2).T, meshes.sphere_tan_proj(v3-v2,v2).T
        h32, h13, h31 = meshes.sphere_tan_proj(v2-v3,v3).T, meshes.sphere_tan_proj(v3-v1,v1).T, meshes.sphere_tan_proj(v1-v3,v3).T
        h41, h42, h43 = meshes.sphere_tan_proj(v4-v1,v1).T, meshes.sphere_tan_proj(v4-v2,v2).T, meshes.sphere_tan_proj(v4-v3,v3).T

        # Gs barycentric coordinates within each split triangle
        g12, g21 = utils.bary_coords(v1,e1,v4,h12), utils.bary_coords(v2,v4,e1,h21)
        g23, g32 = utils.bary_coords(v2,e2,v4,h23), utils.bary_coords(v3,v4,e2,h32)
        g13, g31 = utils.bary_coords(v1,v4,e3,h13), utils.bary_coords(v3,e3,v4,h31)

        g14 = utils.bary_coords(v1,v4,e3, meshes.sphere_tan_proj(v4,v1).T)
        g24 = utils.bary_coords(v2,v4,e1, meshes.sphere_tan_proj(v4,v2).T)
        g34 = utils.bary_coords(v3,v4,e2, meshes.sphere_tan_proj(v4,v3).T)

        # barycentric coordinates of midpoint
        mid = utils.bary_coords(v1,v2,v3,v4)

        # barycentric coordinates of the edge vectors
        e_1, e_2, e_3 = utils.bary_coords(v1,v2,v3,e1), utils.bary_coords(v1,v2,v3,e2), utils.bary_coords(v1,v2,v3,e3)

        # organize data, naming subject to scrutiny
        Hs = np.array([h12.T, h21.T, h23.T, h32.T, h13.T, h31.T, h41.T, h42.T, h43.T])
        Gs = np.array([g12, g21, g23, g32, g13, g31, g14, g24, g34, mid])
        Es = np.array([e_1, e_2, e_3])

        return Hs, Gs, Es


def project_onto_S12_PS_vector(u_lms, u_lms1d, u_coeffs, u_out, L, L_plus, L_minus, L_z, s_points, Method = "MWSS"):
    """
    TODO: how to precompute for the spherical harmonic transform?? This should greatly reduce the computation time

    function which outputs a vector spline interpolant defined over a structured
    triangulation of the sphere allowing for O(1) querying.

    """
    # assign 1D arrays for the transform:
    count = 0
    for l in range(L):
        for m in range(2*l+1):
            u_lms1d[:, 0, count] = u_lms[:,l,m]

            count +=1 

    # compute the components and assign to u_coeffs
    temp_ulms = u_lms.copy()

    for i in range(3):

        # assign values
        u_coeffs[i,0,:,:] = pysh.inverse(u_lms1d[i,0,:], L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 7)
        
        # compute angular momentum
        tempx = angular_momentum(u_lms[i,:,:], temp_ulms, L_plus, L_minus, L_z)
        # convert to 1D array and assign indices 1-3 are for gradient
        # using loop since the function is jitted
        count = 0
        for l in range(L):
            for m in range(2*l+1):
                u_lms1d[i, 1::, count] = tempx[:,l,m]

                count +=1 

        # compute spherical harmonic transform of each components and assign to u_coeffs
        u_coeffs[i,1,:,:] = pysh.inverse(u_lms1d[i,1,:], L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 7)
        u_coeffs[i,2,:,:] = pysh.inverse(u_lms1d[i,2,:], L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 7)
        u_coeffs[i,3,:,:] = pysh.inverse(u_lms1d[i,3,:], L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 7)

        # rotate the values to obtain actual gradient:
        # TODO: verify this
        u_coeffs[i,1::,:,:] = tan_proj(u_coeffs[i,1::,:], s_points, cross(u_coeffs[i,1::,:,:], u_coeffs[i,1::,:,:].copy(), s_points))

        # assign arrays to the u_out:
        u_out[i,:,1:-1] = u_coeffs[i,:,1:-1,:].reshape(4, (L-1)*2*L)

        # then polar points:
        u_out[i,:,-1] = u_coeffs[i,:,-1,0]
        u_out[i,:,0] = u_coeffs[i,:,0,0]

    return u_out


@nb.jit(nopython = True)
def vector_spline_eval(out_vals, u_coeffs, q_pts, EE, nCs, Bcc, Cfs, bcc, trangs, v_pts):

    # bcc, trangs, v_pts = query_vector_spline(q_pts, Bcc, phis, thetas, points, msimplices, trangs, verts)
    
    #assign edge list
    EE[0] = v_pts[:,0,:]; EE[1] = v_pts[:,1,:]; EE[2] = v_pts[:,2,:]
    EE[3] = div_norm2((v_pts[:,0,:] + v_pts[:,1,:] + v_pts[:,2,:])/3) 
    EE[4] = div_norm2(v_pts[:,0,:]/2 + v_pts[:,1,:]/2) 
    EE[5] = div_norm2(v_pts[:,1,:]/2 + v_pts[:,2,:]/2) 
    EE[6] = div_norm2(v_pts[:,2,:]/2 + v_pts[:,0,:]/2)

    v_pts_n = np.zeros(v_pts.shape)
    N = len(bcc[:,0])

    for i in range(N):
        b_min = np.argmin(bcc[i,:]);  b_max = np.argmax(bcc[i,:])
        ee = edges_ps[b_min, b_max] 
        v_pts_n[i,0,:] = EE[ee[0],i,:]
        v_pts_n[i,1,:] = EE[ee[1],i,:]
        v_pts_n[i,2,:] = EE[ee[2],i,:]
        nCs[i,:] = Cs[b_min, b_max]

    cfs_n = u_coeffs[:,:,trangs]
    
    out_vals[0] = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_n[0,:,:], bcc = Bcc, Cfs = Cfs)
    out_vals[1] = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_n[1,:,:], bcc = Bcc, Cfs = Cfs)
    out_vals[2] = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_n[2,:,:], bcc = Bcc, Cfs = Cfs)

    return out_vals


@nb.jit(nopython = True)
def query_vector_spline(q_pts, phis, thetas, phi_l, theta_l, Bcc, points, msimplices, trangs, verts):
    phi_q = (np.arctan2(q_pts[1],q_pts[0]) + 2*np.pi) % (2*np.pi)
    theta_q  = np.arctan2(np.sqrt(q_pts[1]**2 + q_pts[0]**2), q_pts[2])

    # find which cell they land in:
    dphi = abs(phis[1]-phis[0]); dthe = abs(thetas[1]-thetas[0])

    n_phi = len(phis); n_the = len(thetas)
    for i in range(len(phi_l)):
        phi_l[i] = int((phi_q[i] -int(phis[0]))/dphi) % n_phi
        theta_l[i] = int((theta_q[i]-int(thetas[0]))/dthe)
    
    theta_l[theta_l == (len(thetas)-1)] = len(thetas)-2

    # need added check if query lands on theta_c = 0 or 1 lines. ==================
    # # #also compute position within the cell
    # end point indices
    # phi_p1 = (phi_l + 1) % (len(phis))
    # the_p1 = theta_l + 1
    # theta_c = (theta_q - thetas[theta_l])/dthe

    # # this checks if theta_c is close enough to the boundary
    # inds0 = np.where((theta_c <= 0.03) & (theta_l != 0))
    # inds1 = np.where((theta_c > 0.85) & (theta_l !=  n_the-2))

    # #in or outside spherical triangle for based on sign of n_vs

    # # bottom 
    # v_0 = sphere2cart(phis[phi_l[inds0]], thetas[theta_l[inds0]])
    # n_vs = cross(v_0.copy(), v_0, sphere2cart(phis[phi_p1[inds0]], thetas[theta_l[inds0]]))

    # q_pts0 = sphere2cart(phi_q[inds0], theta_q[inds0])

    # v_01 = sphere2cart(phis[phi_l[inds1]], thetas[the_p1[inds1]])
    # v_11 = sphere2cart(phis[phi_p1[inds1]], thetas[the_p1[inds1]])

    # n_vs2 = cross(v_01.copy(), v_01, v_11)
    # q_pts1 = sphere2cart(phi_q[inds1], theta_q[inds1])

    # s_inds1 = np.heaviside(-dot(n_vs2, q_pts1), 0)
    # s_inds0 = np.heaviside(utils.dot(n_vs, q_pts0), 0)  

    # for i in range(len(inds0)):
    #     theta_l[inds0[i]] = theta_l[inds0[i]] - int(s_inds0[i])
    
    # for i in range(len(inds1)):
    #     theta_l[inds1[i]] = theta_l[inds1[i]] + int(s_inds1[i])
    # ======================================================================
    s_inds = -dot(cross(q_pts.copy(), sphere2cart(q_pts.copy(), phis[phi_l], thetas[theta_l]),
                    sphere2cart(q_pts.copy(), phis[(phi_l + 1) % (n_phi)], thetas[theta_l + 1])), q_pts)
    # take heaviside of s_inds
    s_inds[np.where(s_inds <= 0.)] = 0.
    s_inds[np.where(s_inds > 0)] = 1

    for i in range(len(s_inds)):
        # find mesh element
        tri_out = msimplices[theta_l[i], phi_l[i], int(s_inds[i]),:]
        trangs[i] = tri_out[3]
        verts[i] = points[tri_out[0:3]]

    bcc = bary_coords(verts[:,0,:], verts[:,1,:], verts[:,2,:], q_pts.T, Bcc)

    return bcc, trangs, verts


# @nb.jit(nopython = True)
# def query_vector_spline(q_pts, Bcc, phi_l, theta_l, s_inds, points, msimplices, trangs, verts):

#     for i in range(len(s_inds)):
#         # find mesh element
#         tri_out = msimplices[theta_l[i], phi_l[i], int(s_inds[i]),:]
#         trangs[i] = tri_out[3]
#         verts[i] = points[tri_out[0:3]]

#     bcc = bary_coords(verts[:,0,:], verts[:,1,:], verts[:,2,:], q_pts.T, Bcc)

#     return bcc, trangs, verts

@nb.jit(nopython = True)
def sphere2cart(outs, phi, theta):
    outs[0] = np.sin(theta)*np.cos(phi)
    outs[1] = np.sin(theta)*np.sin(phi)
    outs[2] = np.cos(theta)

    return outs

@nb.jit(nopython = True)
def cart2sphere(angs, xyz, L):
    #This is modified to lambda \in [0,2pi)
    angs[0] = ((np.arctan2(xyz[1],xyz[0]) + 2*np.pi) % (2*np.pi))
    angs[1] = np.arctan2(np.sqrt(xyz[1]**2 + xyz[0]**2), xyz[2])
    return angs

# @nb.jit(nopython = True)
def U_interp1(U_coeffs, t, dt, xyz, phis, thetas, U_verts, msimplices, phi_l, theta_l, trangs, verts, s_evals1D, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts):
    # phi_l, theta_l : arrays used for the indices of the query points xyz
    # Bcc, trangs, verts: derived from the mesh

    # first query the mesh as these points:
    bcc0, trangs0, v_pts0 = query_vector_spline(xyz, phis, thetas, phi_l, 
                    theta_l, Bcc_qpts, U_verts, msimplices, trangs, verts)

    return vector_spline_eval(s_evals1D, U_coeffs, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 

# @nb.jit(nopython = True)
def U_interp2(U_coeffs1, U_coeffs2, t0, t, dt, xyz, phis, thetas, U_verts, msimplices, phi_l, theta_l, trangs, verts, s_evals1D, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts):
    # phi_l, theta_l : arrays used for the indices of the query points xyz
    # Bcc, trangs, verts: derived from the mesh

    # first query the mesh as these points:
    bcc0, trangs0, v_pts0 = query_vector_spline(xyz, phis, thetas, phi_l, 
                    theta_l, Bcc_qpts, U_verts, msimplices, trangs, verts)

    u_out1 = vector_spline_eval(s_evals1D, U_coeffs1, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 
    u_out2 = vector_spline_eval(s_evals1D, U_coeffs2, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 

    tau = (t - t0)/dt

    return (1-tau)*u_out1 + tau*u_out2

# @nb.jit(nopython = True)
def U_interp3(U_coeffs1, U_coeffs2, U_coeffs3, t0, t, dt, xyz, phis, thetas, U_verts, msimplices, phi_l, theta_l, trangs, verts, s_evals1D, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts):
    # phi_l, theta_l : arrays used for the indices of the query points xyz
    # Bcc, trangs, verts: derived from the mesh

    # first query the mesh as these points:
    bcc0, trangs0, v_pts0 = query_vector_spline(xyz, phis, thetas, phi_l, 
                    theta_l, Bcc_qpts, U_verts, msimplices, trangs, verts)

    u_out1 = vector_spline_eval(s_evals1D, U_coeffs1, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 
    u_out2 = vector_spline_eval(s_evals1D, U_coeffs2, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 
    u_out3 = vector_spline_eval(s_evals1D, U_coeffs3, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 

    tau0, tau1, tau2 = t0, t0 + dt, t0 + 2*dt
    l0 = (t-tau1)*(t-tau2)/(2*dt**2)
    l1 = (t-tau0)*(t-tau2)/(-dt**2)
    l2 = (t-tau0)*(t-tau1)/(2*dt**2)

    return l0*u_out1 + l1*u_out2 + l2*u_out3


# from numba import int32, float64   # import the types
# specs = [('V1', nb.float64[:]), 
#          ('V2', nb.float64[:]), 
#          ('V3', nb.float64[:]),
#          ('t0', nb.float64),
#          ('flag', nb.int32)]

# @nb.experimental.jitclass(specs)
class velocity_interp(object):
    """
    class to perform interpolation in time for velocity fields.
    """
    def __init__(self, L, t0 = 0, amount = 1):

        # mesh quantities:
        [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)        
        # create a dictionary for the grid
        N, M = len(phis), len(thetas)
        XX = np.meshgrid(phis, thetas[1:-1])
        ico_v = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

        simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
        self.phis = phis
        self.thetas = thetas
        self.simplices = np.array(simplices) 
        self.msimplices = np.array(msimplices)

        # pre-allocate arrays for the velocity field interpolants:
        self.verts = ico_v.points
        self.v_pts_v = self.verts[simplices]

        # vectors along the side of the PS split triangulation
        self.Hs, self.Gs, self.Es = ps_split(self.v_pts_v)
        
        # placeholder arrays:
        Nvv = self.v_pts_v.shape[0]
        self.cfs_v = np.zeros([3,Nvv,3])
        self.grad_fx_v = np.zeros([3,3,Nvv]); 
        self.grad_fy_v = np.zeros([3,3,Nvv]); 
        self.grad_fz_v = np.zeros([3,3,Nvv])
        
        # define the three coefficients array defining the interpolants of the velocity field
        self.coeffs0 = np.zeros([3,19, self.v_pts_v.shape[0]])
        self.coeffs1 = self.coeffs0.copy() 
        self.coeffs2 = self.coeffs0.copy()

        # define arrays for the actial
        # u0[i,j,:] gives values of D_j u^i on grid points
        self.u_coeffs0 = np.zeros([3, 4, L+1, 2*L])
        self.u_coeffs1 = self.u_coeffs0.copy()
        self.u_coeffs2 = self.u_coeffs0.copy()
        self.u0 = np.zeros([3, 4, (L-1)*2*L + 2])
        self.u1 = self.u0.copy()
        self.u2 = self.u0.copy()

        self.t0 = t0 #start time of the interpolant
        self.amount = amount

        return

    def coeffs_assemble0(self):
        self.coeffs0[:] = assemble_coefficients(self.coeffs0, self.simplices, self.u0[:,0,:], self.u0[:,1::,:], 
                        self.cfs_v, self.grad_fx_v, self.grad_fy_v, self.grad_fz_v, self.Hs, self.Gs, self.Es)

        return

    def coeffs_assemble1(self):
        self.coeffs1[:] = assemble_coefficients(self.coeffs1, self.simplices, self.u1[:,0,:], self.u1[:,1::,:], 
                        self.cfs_v, self.grad_fx_v, self.grad_fy_v, self.grad_fz_v, self.Hs, self.Gs, self.Es)

        return
    
    def coeffs_assemble2(self):
        self.coeffs2[:] = assemble_coefficients(self.coeffs2, self.simplices, self.u2[:,0,:], self.u2[:,1::,:], 
                        self.cfs_v, self.grad_fx_v, self.grad_fy_v, self.grad_fz_v, self.Hs, self.Gs, self.Es)

        return

    def __call__(self, t, dt, xyz, phi_l, theta_l, trangs, verts, s_evals1D, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts):
        # phi_l, theta_l : arrays used for the indices of the query points xyz
        # Bcc, trangs, verts: derived from the mesh

        # first query the mesh as these points:
        bcc0, trangs0, v_pts0 = query_vector_spline(xyz, self.phis, self.thetas, phi_l, 
                        theta_l, Bcc_qpts, self.verts, self.msimplices, trangs, verts)
        
        if self.amount == 3:
            u_out1 = vector_spline_eval(s_evals1D, self.coeffs0, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 
            u_out2 = vector_spline_eval(s_evals1D, self.coeffs1, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 
            u_out3 = vector_spline_eval(s_evals1D, self.coeffs2, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 

            tau0, tau1, tau2 = self.t0, self.t0 + dt, self.t0 + 2*dt
            l0 = (t-tau1)*(t-tau2)/(2*dt**2)
            l1 = (t-tau0)*(t-tau2)/(-dt**2)
            l2 = (t-tau0)*(t-tau1)/(2*dt**2)

            return l0*u_out1 + l1*u_out2 + l2*u_out3

        if self.amount == 1:
            u_out1 = vector_spline_eval(s_evals1D, self.coeffs0, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 

            return u_out1


        if self.amount == 2:
            u_out1 = vector_spline_eval(s_evals1D, self.coeffs0, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 
            u_out2 = vector_spline_eval(s_evals1D, self.coeffs1, xyz, edges_list_qpts, nCs_qpts, Bcc_qpts, Cfs_qpts, bcc0, trangs0, v_pts0) 

            tau = (t - self.t0)/dt

            return (1-tau)*u_out1 + tau*u_out2



def euler_simulation_rotating_sphere_functional(L, Nt, T, mesh, vorticity):
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


def euler_simulation_video_rotating_sphere(L, Nt, T, mesh, vorticity, save_steps, file_name, params):
    """
    Parameters:
        - L (int): Band-limit defining the sampling of the vorticity
        - Nt (int): number of time steps
        - T (float): Final integration time
        - mesh (spherical_triangulation): mesh that the map is discretized on
        - vorticity (callable): function defining the initial vorticity
        - save_steps (int): save the map every save_steps into file = file_name
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
    steps = 2
    saved_maps = []

    for t in tspan[2::]:
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

        steps +=1

        if steps % save_steps == 0:
            #save the output of the model into the file with the parameters of the model:
            saved_maps.append(curr_map)
            file = open(file_name + "_maps" + ".txt", "wb")
            pickle.dump([params, saved_maps], file)

    return 



def euler_simulation_video_rotating_sphere_remapping(L, Nt, T, n_maps, mesh, vorticity, save_steps, file_name, params):
    """
    Parameters:
        - L (int): Band-limit defining the sampling of the vorticity
        - Nt (int): number of time steps
        - T (float): Final integration time
        - mesh (spherical_triangulation): mesh that the map is discretized on
        - vorticity (callable): function defining the initial vorticity
        - save_steps (int): save the map every save_steps into file = file_name
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
    save_step_count = 0
    steps = 2
    remaps = []
    for t in tspan[2::]:
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
        steps +=1
        just_remapped = False

        if steps % n_maps == 0:
            remaps.append(curr_map)
            curr_map = map0
            just_remapped = True

        if steps % save_steps == 0:
            save_step_count +=1
            #save the output of the model into the file with the parameters of the model:
            file = open(file_name + "_maps_step%s" % save_step_count + ".txt", "wb")
            pickle.dump([params, remaps + [curr_map]], file)

    if just_remapped == False:
        remaps.append(curr_map)


    return 



def euler_simulation_static_sphere(L, Nt, T, mesh, vorticity):
    """
    Parameters
        - L (int): Band-limit defining the sampling of the vorticity
        - Nt (int): number of time steps
        - T (float): Final integration time
        - mesh (spherical_triangulation): mesh that the map is discretized on
        - vorticity (callable): function defining the initial vorticity
        
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
        curr_map = evol.advect_project_sphere(curr_map, evol.RK4_proj,
                         t, dt, U, just_remapped)

        XXn = utils.cart2sphere(evol.compose_maps(remaps, sample_pts, current = [curr_map]))
        angs = [XXn[0].reshape([L+1,2*L]), XXn[1].reshape([L+1,2*L])]
        omg_n = vorticity(angs[0], angs[1])
        psi_lms = sph_tools.inv_Laplacian(omg_n, L)

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


def euler_simulation_static_sphere_remapping(L, Nt, T, n_maps, mesh, vorticity):
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
        omg_n = vorticity(angs[0], angs[1]) 
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



# class euler_simulation_sphere_parallel(object):

#     def __init__(self, L, T, mesh, vorticity, rotating = False):
#         """
#         sets up an iterable to perform the Euler simulation with parallelization
#         of the map evaluation.
#         Parameters
#             - L (int): Band-limit defining the sampling of the vorticity
#             - T (float): Final integration time
#             - mesh (spherical_triangulation): mesh that the map is discretized on
#             - vorticity (callable): function defining the initial vorticity
#         """
#         #TODO: include functionality for adaptive time-stepping
#         #--------- Velocity Field set-up -------------------------
#         self.L = L
#         [thetas, phis] = pysh.sample_positions(L, Method = "MWSS", Grid = False)
#         [Phi, The] = np.meshgrid(phis, thetas)
#         s_points = utils.sphere2cart(Phi,The)
#         spts = np.array([s_points[0].reshape([(L+1)*2*L,]), s_points[1].reshape([(L+1)*2*L,]),
#                 s_points[2].reshape([(L+1)*2*L,])])
#         self.spts = spts
#         #obtain stream function
#         psi0 = vorticity(Phi, The)
#         psi_lms = sph_tools.inv_Laplacian(psi0, L)

#         #create a dictionary for the grid
#         N, M = len(phis), len(thetas)
#         XX = np.meshgrid(phis, thetas[1:-1])
#         ico = meshes.spherical_mesh(XX[0], XX[1], N, M-2)

#         simplices, msimplices = meshes.full_assembly(len(phis), len(thetas))
#         self.grid_dict = {"phis": phis, "thetas": thetas, "simplices": simplices,
#                      "msimplices": msimplices, "mesh": ico, "sample_points": s_points}

#         u0 = sph_tools.project_stream(psi_lms, L, grid = self.grid_dict, Method = "MWSS")
#         self.rotating = rotating

#         if rotating:
#             self.U = vel.velocity_interp_rotating(Vs = [u0], t0 = 0, T = T)
#         else:
#             self.U = vel.velocity_interp(Vs = [u0], t0 = 0)
#         #----------- Flow map interpolation set-up ---------------
#         vals = [mesh.x, mesh.y, mesh.z]
#         grad_vals = [utils.grad_x(mesh.vertices.T),
#                      utils.grad_y(mesh.vertices.T),
#                      utils.grad_z(mesh.vertices.T)]

#         # define initial discretization of the map
#         self.map0 = sphere_diffeomorphism(mesh = mesh, vals = vals,
#                                         grad_vals = grad_vals)
#         self.curr_map = self.map0
#         #----------------------------------------------------------
#         self.tspan = np.linspace(0, T, t_res, endpoint = False)
#         self.dt = tspan[1]-tspan[0]

#         self.remaps = []
#         self.t_ind = 2
#         # # Bootstrap first two steps:
#         # initializes the current map
#         self.initialize(0, self.dt, self.map0, vorticity, self.U, self.grid_dict, spts, Phi, The, L, T)

#         return

#     def step(self, Nt = 1):
#         # void function which steps through the current state
#         # remapping criterion is meant to be checked outside of this function
#         for t in self.tspan[self.t_ind:Nt]:
#             self.curr_map = evol.advect_project_sphere(self.curr_map, evol.RK4_proj,
#                              t, self.dt, self.U, self.identity)

#             XXn = utils.cart2sphere(evol.compose_maps(remaps, spts, current = interpolant))
#             angs = [XXn[0].reshape([self.L+1,2*self.L]), XXn[1].reshape([self.L+1,2*self.L])]

#             if self.rotating:
#                 omg_n = self.vorticity(angs[0], angs[1]) + vel.rotating_frame(angs[0],angs[1])
#                 zeta = omg_n - vel.rotating_frame(Phi,The)
#                 psi_lms = sph_tools.inv_Laplacian(zeta, L)

#                 # new velocity field
#                 u_n = sph_tools.project_stream(psi_lms, L, grid = self.grid_dict, Method = "MWSS")
#                 self.U = vel.velocity_interp_rotating(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, T = T)
#                 return
#             else:
#                 omg_n = self.vorticity(angs[0], angs[1])
#                 psi_lms = sph_tools.inv_Laplacian(omg_n, L)

#                 # new velocity field
#                 u_n = sph_tools.project_stream(psi_lms, L, grid = self.grid_dict, Method = "MWSS")
#                 self.U = vel.velocity_interp(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, T = T)
#                 return

#     def initialize(self):
#         # void function to initialize first two steps of the simulation.
#         t = 0
#         #first take a lil' Euler Step
#         int0 = evol.advect_project_sphere(self.map0, evol.Euler_step_proj, t, self.dt, self.U, identity = True)
#         angs0 = utils.cart2sphere(int0.eval(spts))
#         L = self.L
#         #sample the vorticity and solve for stream function
#         if self.rotating:
#             angs0 = [angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])]
#             zeta0 = self.vorticity(angs0[0], angs0[1]) + rotating_frame(angs0[0],angs0[1]) - rotating_frame(Phi,The)
#             psi_lms1 = sph_tools.inv_Laplacian(zeta0, L)
#             #new velocity field
#             u1 = sph_tools.project_stream(psi_lms1, L, grid = self.grid_dict, Method = "MWSS")

#             # append into the interpolant
#             self.U.Vs.append(u1)
#             # now repeat
#             int1 = evol.advect_project_sphere(self.map0, evol.improved_Euler_proj, t, self.dt, self.U, identity = True)
#             angs1 = utils.cart2sphere(int1(spts))
#             angs1 = [angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])]
#             zeta1 = self.vorticity(angs1[0], angs1[1]) + rotating_frame(angs1[0],angs1[1]) - rotating_frame(Phi,The)
#             psi_lms2 = sph_tools.inv_Laplacian(zeta1, L)
#             #new velocity field
#             u2 = sph_tools.project_stream(psi_lms2, L, grid = self.grid_dict, Method = "MWSS")
#             self.U.Vs[1] = u2

#             # and again
#             int2 = evol.advect_project_sphere(int1, evolv.RK4_proj, t + self.dt, self.dt, self.U)
#             angs2 = utils.cart2sphere(int2(spts))
#             angs2 = [angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])]
#             zeta2 = self.vorticity(angs2[0], angs2[1]) + rotating_frame(angs2[0],angs2[1]) - rotating_frame(Phi,The)
#             psi_lms3 = sph_tools.inv_Laplacian(zeta2, L)
#             #new velocity field
#             u3 = sph_tools.project_stream(psi_lms3, self.L, grid = self.grid_dict, Method = "MWSS")
#             self.U.Vs.append(u3)
#             self.curr_map = int2

#             return
#         else:
#             psi_lms1 = sph_tools.inv_Laplacian(self.vorticity(angs0[0].reshape([L+1,2*L]), angs0[1].reshape([L+1,2*L])), L)
#             #new velocity field
#             u1 = sph_tools.project_stream(psi_lms1, self.L, grid = self.grid_dict, Method = "MWSS")

#             # append into the interpolant
#             self.U.Vs.append(u1)
#             # now repeat
#             int1 = evol.advect_project_sphere(self.map0, evol.improved_Euler_proj, t, self.dt, self.U, identity = True)
#             angs1 = utils.cart2sphere(int1(spts))
#             #sample the vorticity and solve for stream function
#             psi_lms2 = sph_tools.inv_Laplacian(self.vorticity(angs1[0].reshape([L+1,2*L]), angs1[1].reshape([L+1,2*L])), L)
#             #new velocity field
#             u2 = sph_tools.project_stream(psi_lms2, self.L, grid = self.grid_dict, Method = "MWSS")
#             self.U.Vs[1] = u2

#             # and again
#             int2 = evol.advect_project_sphere(int1, evolv.RK4_proj, t + self.dt, self.dt, self.U)
#             angs2 = utils.cart2sphere(int2.eval(spts))
#             #sample the vorticity and solve for stream function
#             psi_lms3 = sph_tools.inv_Laplacian(self.vorticity(angs2[0].reshape([L+1,2*L]), angs2[1].reshape([L+1,2*L])), L)
#             #new velocity field
#             u3 = sph_tools.project_stream(psi_lms3, self.L, grid = self.grid_dict, Method = "MWSS")
#             self.U.Vs.append(u3)
#             self.curr_map = int2

#         return



