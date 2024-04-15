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
from ..core.spherical_spline import spherical_spline

# Transform tools
def inv_Laplacian(samples, L, L_inv):
    # helper function for barotropic vorticity projection
    f_lm = pysh.forward(samples, L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 8)
    flm_coeffs = coeff_array(f_lm, L)*L_inv[:,None]

    return flm_coeffs

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

def angular_momentum(alms, L_plus, L_minus, L_z, temp):
    # angular momentum operator (r \times \nabla) applied to sph_harm coeffs
    temp *= 0 
    temp[:,1::] = alms[:,0:-1]
    out_plus = np.multiply(L_plus, temp)
    
    temp *= 0
    temp[:,0:-1] = alms[:,1::]
    out_minus = np.multiply(L_minus, temp)

    out_x = (out_plus + out_minus)/2
    out_y = 1j*(out_minus - out_plus)/2

    return [1j*out_x, 1j*out_y, 1j*np.multiply(L_z,alms)]

#--------------Projection onto spline space: -----------------------------------

def project_stream(alms, L, L_plus, L_minus, L_z, U, n, temp):
    """
    function which computes I_h^3[L \psi]
    projects the psi function onto the velocity interpolation space
    at the n-th point in time
    """

    if U.stepping:
        # shift all the values over and initialize the first interpolant
        U.coeffs[0:2] = U.coeffs[1:3]
        U.vals[0:2] = U.vals[1:3]

    s_pts = U.mesh.grid_points
    u_lms = angular_momentum(alms, L_plus, L_minus, L_z, temp)

    for i in range(3):
        # differentiate once again:

        u_temp = pysh.inverse(lm_array(u_lms[i], L), L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 8)
        U.vals[n,i,0,1:-1] = u_temp[1:-1].reshape((L-1)*2*L).real
        # replace pole points
        U.vals[n,i,0,-1] = u_temp[-1,0].real
        U.vals[n,i,0,0] = u_temp[0,0].real

        flm_ang = angular_momentum(u_lms[i], L_plus, L_minus, L_z, temp)
        u_grad = np.array([pysh.inverse(lm_array(flm_ang[i],L), L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 8) for i in range(3)])
        
        u_grad[:] = tan_proj(utils.cross(u_grad,s_pts), s_pts) 
        # pdb.set_trace()

        # intermediate gradient replacement
        U.vals[n,:,i+1,1:-1] = u_grad[:,1:-1,:].reshape([3,(L-1)*2*L]).real

        # replace pole points
        U.vals[n,:,i+1,-1] = u_grad[:,-1,0].real
        U.vals[n,:,i+1,0] = u_grad[:,0,0].real

    U.init_coeffs(n) 

    return 


def project_onto_S12(alms, L, mesh):
    """
    function which computes I_h^3[f] from the spherical harmonic coefficients of f
    """
    vals = np.zeros([4,len(mesh.vertices)])

    L_plus = np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l-1)), (L,2*L+1), dtype = 'float64')
    L_minus = np.fromfunction(lambda l,m: np.sqrt(l*(l+1)-(m-l)*(m-l+1)), (L,2*L+1), dtype = 'float64')
    L_plus[np.where(np.isnan(L_plus))] = 0. 
    L_minus[np.where(np.isnan(L_minus))] = 0. 
    L_z = np.fromfunction(lambda l,m: m-l, (L, 2*L+1), dtype = 'float64')
    temp = 0*alms

    s_pts = mesh.grid_points
    u_lms = angular_momentum(alms, L_plus, L_minus, L_z, temp)
    u_grad = np.array([pysh.inverse(lm_array(u_lms[i], L), L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 8) for i in range(3)])
    
    u_grad[:] = tan_proj(utils.cross(u_grad,s_pts), s_pts) # rotate

    vals[1::,1:-1] = u_grad[:,1:-1].reshape(3, (L-1)*2*L).real
    # replace pole points
    vals[1::,-1] = u_grad[:,-1,0].real
    vals[1::,0] = u_grad[:,0,0].real
    
    # replace the values:
    vals_temp = pysh.inverse(lm_array(alms, L), L, Spin = 0, Method = "MWSS", Reality = True, backend = 'ducc', nthreads = 8)

    vals[0,1:-1] = vals_temp[1:-1,:].reshape((L-1)*2*L).real
    # replace pole points
    vals[0,-1] = vals_temp[-1,0].real
    vals[0,0] = vals_temp[0,0].real

    coeffs = np.zeros([19, np.shape(mesh.vertices[np.array(mesh.simplices)])[0]])

    return spherical_spline(mesh, vals, coeffs)


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
