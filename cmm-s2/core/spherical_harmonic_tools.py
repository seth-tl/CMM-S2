#/---
"""
Collection of functions used in spherical harmonic coefficient space
"""
#/---
import numpy as np
import pdb, stripy, time
from ..core import utils
import pyssht as pysh
from scipy.special import sph_harm
from ..core.spherical_spline import spline_interp_vec, spline_interp_structured

# Transform tools

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

def lm_array(coeffs,L):
    # converts coefficient array back into approriate 1D array for SHHT
    outs = [coeffs[l,m] for l in range(0,L) for m in range(0,2*l+1)]
    return np.array(outs)

#Ladder operator computation
def raise_op(alms, L):
    # raising operator L_+ applied to the coefficients of a band-limited function
    # with coefficients alm.

    #Construct matrix based on band-limit L
    L_p = np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l-1)), (L,2*L+1), dtype = 'float64')
    outs = np.zeros([L, 2*L+1],  dtype = 'complex128')
    outs[:,1::] = alms[:,0:-1]
    return np.multiply(L_p, outs)


def lower_op(alms, L):
    # lowering operator L_- applied to the coefficients of a band-limited function
    # with coefficients alm.
    L_m = np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l+1)), (L,2*L+1), dtype = 'float64')
    L_m[np.where(np.isnan(L_m))] = 0.
    outs = np.zeros([L, 2*L+1],  dtype = 'complex128')
    outs[:,0:-1] = alms[:,1::]

    return np.multiply(L_m, outs)

def angular_momentum(alms, L):
    # angular momentum operator (r \times \nabla) applied to sph_harm coeffs
    out_x = (raise_op(alms, L) + lower_op(alms,L))/2
    out_y = 1j*(lower_op(alms,L) - raise_op(alms,L))/2

    L_z = np.fromfunction(lambda l,m: m-l, (L, 2*L+1), dtype = 'float64')

    return [1j*out_x, 1j*out_y, 1j*np.multiply(L_z,alms)]

#--------------Projection onto spline space: -----------------------------------
def prune(A):
    # helper function for projection operation
    outs = A[1:-1,:].real
    outs = list(outs.reshape([len(outs[0,:])*len(outs[:,0]),]))
    outs.append(A[-1,0])
    outs.insert(0,A[0,0])

    return outs


def project_stream(alms, L, grid, Method = "MWSS"):
    """
    function which outputs a spline interpolant defined over a structured
    triangulation of the sphere allowing for O(1) querying.

    alms should be a coefficient array of the stream function to be projected.
    """
    #computes gradient at grid points defined by bandlimit L
    #if vector == False:
    u_lms = angular_momentum(alms,L)

    return project_onto_S12_PS_vector(u_lms, L, grid, Method = Method)


def project_onto_S12_PS_vector(alms, L, grid, Method = "MWSS"):
    """
    function which outputs a spline interpolant defined over a structured
    triangulation of the sphere allowing for O(1) querying.

    alms should be a coefficient array of the stream function to be projected.
    """
    #computes gradient at grid points defined by bandlimit L
    [X,Y,Z] = grid["sample_points"]


    values = []
    grad_values = []
    for flm in alms:
        u_lms = [lm_array(A,L) for A in angular_momentum(flm,L)]
        u_num = [pysh.inverse(u_lms[i], L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 7) for i in range(3)]
        # gradient operator
        grad_vals = tan_proj(utils.cross(u_num,[X,Y,Z]), [X,Y,Z])
        vals = pysh.inverse(lm_array(flm, L), L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 7)

        # vals and grad_vals should 1-D arrays
        # perform pruning to the arrays
        vals_n = prune(vals)
        grad_vals_n = [prune(B) for B in grad_vals]

        values.append(np.array(vals_n))
        grad_values.append(np.array(grad_vals_n))
    

    # arrange gradient
    interpolant = spline_interp_vec(grid = grid["mesh"], simplices = grid["simplices"],
                 msimplices = grid["msimplices"], phi = grid["phis"], theta = grid["thetas"],
                 vals = values, grad_vals = grad_values)

    return interpolant

def project_onto_S12_PS(alms, L, grid, Method = "MWSS"):
    """
    function which outputs a spline interpolant defined over a structured
    triangulation of the sphere allowing for O(1) querying.

    alms should be a coefficient array of the stream function to be projected.
    """
    #computes gradient at grid points defined by bandlimit L
    [X,Y,Z] = grid["sample_points"]

    u_lms = [lm_array(A,L) for A in angular_momentum(alms,L)]
    u_num = []
    for i in [0,1,2]:
        u_num.append(pysh.inverse(u_lms[i], L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 5))

    # gradient operator
    grad_vals = tan_proj(utils.cross(u_num,[X,Y,Z]), [X,Y,Z])
    vals = pysh.inverse(lm_array(alms, L), L, Spin = 0, Method = Method, Reality = True, backend = 'ducc', nthreads = 5)

    # vals and grad_vals should 1-D arrays
    # perform pruning to the arrays
    vals_n = prune(vals)
    grad_vals_n = np.array([prune(B) for B in grad_vals])

    # perform projection
    interpolant = spline_interp_structured(grid = grid["mesh"], simplices = grid["simplices"],
                 msimplices = grid["msimplices"], phi = grid["phis"], theta = grid["thetas"],
                 vals = np.array(vals_n), grad_vals = grad_vals_n)
    
    return interpolant


def proj(a,b):
    #project a on to b numpy arrays
    scl = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])/np.sqrt(b[0]**2 + b[1]**2 +b[2]**2)
    return [scl*b[0], scl*b[1], scl*b[2]]

def tan_proj(U,R):
    pU = proj(U,R)
    return [U[0]-pU[0], U[1]-pU[1], U[2]-pU[2]]


# ------------- Post-processing functions --------------------------------------
# Power Spectrum calculation
def energy_spectrum(alms,L):
    # alms should be the vorticity spherical harmonic coefficients
    outs = [sum(alms[0,:]*alms[0,:].conjugate())]
    for ell in range(1,L):
        outs.append(sum(alms[ell,:]*alms[ell,:].conjugate())/(ell*(ell+1)))

    return np.array(outs)
