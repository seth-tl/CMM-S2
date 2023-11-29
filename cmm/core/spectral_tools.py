#/--------
"""
A collection of tools to perform operations in frequency space
"""
#/-------

import numpy as np
#import pyfftw
from numpy.fft import fftshift, ifftshift, fft2, ifft2, fft, ifft
from scipy.linalg import solve_banded, solve
from .interpolants import torus_interpolants as interps
from . import mesh_functions as meshes
from scipy import sparse
import scipy.sparse.linalg

from numpy.linalg import inv
import pdb


# channel flow Poisson solve tools:------------------------------------
def Mmul(A,b):
    return  np.matmul(A,b)

def Laplace(N):
    M = np.ones([3,N])
    M[1,:] = -2
    return sparse.spdiags(M, (-1,0,1), N,N)

def Poisson_solve_cylinder(omg, dy, b_l, b_u):
    # solves Poisson equation on cylinder
    N = len(omg)
    psi = np.zeros([N,N], dtype = 'complex128')
    freqs = -ifftshift(np.linspace(-N//2, N//2, N+1)**2)
    Omg = np.zeros([N,N], dtype = 'complex128')
    L_h = Laplace(N-2)

    #modify for compact finite difference - fourth order
    Lh = Laplace(N).toarray()
    for i in range(N):
        Omg[:,i] = Mmul(Lh,omg[:,i])/12 + omg[:,i]

    Omg = (dy**2)*Omg; Omg[1,:] += -b_l; Omg[N-2,:] += -b_u;

    for i in range(N):
        Omg[i,:] = fft(Omg[i,:])

    for j in range(N):
        k_Id = sparse.spdiags((dy**2)*np.ones(N-2)*freqs[j], (0), N-2, N-2)
        psi[1:-1,j] = sparse.linalg.spsolve(L_h + k_Id, Omg[1:-1,j])

    for i in range(1,N-1):
        psi[i,:] = ifft(psi[i,:])

    psi[0,:] = b_l
    psi[N-1,:] = b_u

    return psi.real


def Biot_Savart_Cylinder(omg, dy, psi_u, psi_l, xs, ys, Lx, Ly):

    psi = Poisson_solve_cylinder(omg, dy, psi_u, psi_l)
    #then differentiate and return an interpolant
    # psi_kx = fft2(psi, axes = (1,))
    # psi_x = ifft2(psi_kx, axes = (1,))
    #
    # #interior point derivatives
    # N = len(psi[:,0])
    # #centred difference stencil
    # psi_y_int = (psi[2::,:]-psi[0:N-2,:])/(2*dy)
    #
    # #at the boundary
    # psi_y_b0 = (psi[1,:]-psi[0,:])/dy
    # psi_y_b1 = -(psi[N-1,:] - psi[N-2,:])/dy
    #
    # psi_y = np.zeros([N,len(psi[0,:])])
    # psi_y[1:-1,:] = psi_y_int; psi_y[0,:] = psi_y_b0; psi_y[N-1,:] = psi_y_b1;

    return Bilinear_Cylinder(psi, xs, ys, Lx, Ly)

#---------------------------------------------------------------------
def Poisson_Solve(f_k):
    # Poisson solve on torus
    """
    f_k in fourier space 2D
    """
    m = len(f_k[0,:])
    n_freqs = ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])
    m_freqs = -(np.linspace(-m/2, m/2, m + 1)[:-1])**2

    U_k = np.zeros([m,m], dtype = "complex128")
    j = 0

    for n in n_freqs:
        if n != 0:
            D_m = np.diag(1/(-n**2 + m_freqs), k=0)
            U_k[:,j] = ifftshift(Mmul(D_m, fftshift(f_k[:,j])))
        else:
            freqs = m_freqs.copy()
            freqs[int(m/2)] = np.inf
            D_m = np.diag(1/freqs, k= 0)
            U_k[:,j] = ifftshift(Mmul(D_m, fftshift(f_k[:,j])))
        j+=1


    return U_k


def partial_x(u_k):
    # assuming L = 2*pi
    u1 = u_k.copy()
    N = len(u_k[:,0])
    freqs = np.arange(-N//2, N//2)
    freqs[0] = 0
    freqs = 1j*ifftshift(freqs)
    # freqs[0] = 1
    for j in range(N):
        u1[:,j] = freqs[j]*u1[:,j]
        j+=1
    return u1

def partial_xx(u_k):
    # assuming L = 2*pi
    u1 = u_k.copy()
    N = len(u_k[:,0])
    freqs = -ifftshift(np.arange(-N//2, N//2)**2)
    for j in range(N):
        u1[:,j] = freqs[j]*u1[:,j]
        j+=1
    return u1


def partial_y(u_k):
    u2 = u_k.copy()
    N = len(u_k[:,0])
    freqs = np.arange(-N//2, N//2)
    freqs[0] = 0
    freqs = 1j*ifftshift(freqs)
    # freqs[0] = 1
    for j in range(N):
        u2[j,:] = freqs[j]*u2[j,:]
    return u2

def partial_yy(u_k):
    u2 = u_k.copy()
    N = len(u_k[:,0])
    freqs = -ifftshift(np.arange(-N//2, N//2)**2)
    for j in range(N):
        u2[j,:] = freqs[j]*u2[j,:]
    return u2


def diff_map_k(map):
    ## Jacobian of map in Fourier
    return [1 + ifft2(partial_x(map[0])), ifft2(partial_y(map[0])), ifft2(partial_x(map[1])), 1 + ifft2(partial_y(map[1]))]


def project_gradient_field(psi_k, psi_objects, L):
    """
    Fourier Space projector for gradient vector field
    onto the space of Hermite cubics.
    Input: psi_k in Fourier space
    """
    #regularize input
    # psi_k = zero_pad(f_k.copy())

    psi_xk = partial_x(psi_k)
    psi_yk = partial_y(psi_k)
    psi_xxk = partial_x(psi_xk)
    psi_yyk = partial_y(psi_yk)
    psi_xyk = partial_y(psi_xk)
    psi_xxyk = partial_y(psi_xxk)
    psi_yxyk = partial_y(psi_xyk)

    psi_objects[1](psi_xk)
    psi_objects[2](psi_yk)
    psi_objects[3](psi_xxk)
    psi_objects[4](psi_yyk)
    psi_objects[5](psi_xyk)
    psi_objects[6](psi_xxyk)
    psi_objects[7](psi_yxyk)

    xs = np.linspace(0, 2*pi, L, endpoint = False)
    ys = np.linspace(0, 2*pi, L, endpoint = False)

    u_x = Hermite_T2(phi = xs, theta = ys,
                     f = (psi_objects[1]().real).copy(),
                     f_x = (psi_objects[3]().real).copy(),
                     f_y = (psi_objects[5]().real).copy(),
                     f_xy = (psi_objects[6]().real).copy())

    u_y = Hermite_T2(phi = xs, theta = ys,
                     f = (psi_objects[2]().real).copy(),
                     f_x = (psi_objects[5]().real).copy(),
                     f_y = (psi_objects[4]().real).copy(),
                     f_xy = (psi_objects[7]().real).copy())


    return [u_x, u_y]


def Biot_Savart(omega_k, L):
    psi_k = Poisson_Solve(omega_k)
    return project_solenoidal_field(psi_k, L)


def project_solenoidal_field(psi_k, L):
    """
    Fourier Space projector for gradient vector field
    onto the space of Hermite cubics.
    Input: psi_k in Fourier space
    """
    # psi = ifft2(psi_k)
    # psi_k = zero_pad(f_k.copy())
    # x- component:
    # TODO: consider parallelizing all of these transforms
    #(x, y, xx, yy, xy, xxy, yxy)

    psi_xk = partial_x(psi_k)
    psi_yk = partial_y(psi_k)
    psi_xxk = partial_x(psi_xk)
    psi_yyk = partial_y(psi_yk)
    psi_xyk = partial_y(psi_xk)
    psi_xxyk = partial_y(psi_xxk)
    psi_yxyk = partial_y(psi_xyk)

    mesh = meshes.torus_mesh(L,L)
    #update the values of the psi_k objects and perform the inverse transforms
    u_y = interps.Hermite_T2(mesh = mesh,
                     f = ifft2(psi_xk).real,
                     f_x = ifft2(psi_xxk).real,
                     f_y = ifft2(psi_xyk).real,
                     f_xy = ifft2(psi_xxyk).real)

    u_x = interps.Hermite_T2(mesh = mesh,
                     f = -1*ifft2(psi_yk).real,
                     f_x = -1*ifft2(psi_xyk).real,
                     f_y = -1*ifft2(psi_yyk).real,
                     f_xy = -1*ifft2(psi_yxyk).real)

    return interps.Hermite_velocity(u_x, u_y)

def zero_pad(f_k):
    # zero_pad fourier series to double its size
    # have to centre the array
    N = len(f_k)
    f_kn = np.zeros([2*N,2*N], dtype = "complex128")
    f_kn[N//2:-N//2, N//2:-N//2] = 4*fftshift(f_k)

    return ifftshift(f_kn)


# def dealias(f_k, Kmax):
#     f_kn = np.zeros([N,N], dtype = "complex128")


def zero_pad_regularize(f_k):
    # zero_pad fourier series to double its size
    # have to centre the array
    N = len(f_k)
    f_kn = np.zeros([2*N,2*N], dtype = "complex128")
    f_kn[N//2+2:-N//2-2, N//2+2:-N//2-2] = 4*fftshift(f_k)[2:-2,2:-2]

    return ifftshift(f_kn)

def project_hermite(u_k, L):

    u_xk = partial_x(u_k)
    u_yk = partial_y(u_k)
    u_xyk = partial_x(u_yk)

    xs = np.linspace(0, 2*pi,L, endpoint = False)
    ys = np.linspace(0, 2*pi,L, endpoint = False)

    return Hermite_T2(phi = xs, theta = ys, f = ifft2(u_k).real,
                     f_x = ifft2(u_xk).real, f_y = ifft2(u_yk).real,
                     f_xy = ifft2(u_xyk).real)

def IFFT2(map):
    return np.array([ifft2(map[0]), ifft2(map[1])])

def FFT2(map):
    return np.array([fft2(map[0]), fft2(map[1])])

def L_arc(f_k):
    #computes the L operator
    #approximate volume using the zero mode
    int_f = f_k[0,0]
    return np.arccos(int_f/(4*pi**2))/np.sqrt(1- (int_f/(4*pi**2))**2)

def information_gradient_alt(W_phi, W_1, gW_1, rho_sq, sigma, XX, L):
    """
    form the vector field resulting from gradient of E(\varphi)
    Variables:
    W_phi = W(\varphi_*\mu_0) -- precompute
    W_1 = W(\mu_1) -- given
    gW_1 = grad(W(\mu_1)) -- given
    rho = J_\mu(\varphi^-1)
    sigma -- parameter in energy functional
    XX -- Fourier grid
    """
    #first compute the integrals
    rho_12_k = fft2(rho_sq)
    w1_wphi_k = fft2(W_phi*W_1)
    c1 = L_arc(rho_12_k)
    c2 = L_arc(w1_wphi_k)
    #compute gradient of rho
    gr_k = [partial_x(rho_12_k), partial_y(rho_12_k)]
    #other portion
    gp_k = [partial_x(w1_wphi_k) - fft2(2*gW_1[0]*W_phi),
            partial_y(w1_wphi_k) - fft2(2*gW_1[1]*W_phi)]
    #Combine
    u_k = [Poisson_Solve(sigma*c1*gr_k[0] + c2*gp_k[0]),
           Poisson_Solve(sigma*c1*gr_k[1] + c2*gp_k[1])]

    return [project_hermite(u_k[0],L), project_hermite(u_k[1],L)]
    #then return Hermite velocity

def information_gradient(W_phi, W_1, gW_1, rho, sigma, XX, L):
    """
    form the vector field resulting from gradient of E(\varphi)
    Variables:
    W_phi = W(\varphi_*\mu_0) -- precompute
    W_1 = W(\mu_1) -- given
    gW_1 = grad(W(\mu_1)) -- given
    rho = J_\mu(\varphi^-1)
    sigma -- parameter in energy functional
    XX -- Fourier grid
    """
    #first compute the integrals
    rho_12_k = fft2(rho)
    w1_wphi_k = fft2(W_phi)
    # c1 = L_arc(rho_12_k)
    # c2 = L_arc(w1_wphi_k)
    #compute gradient of rho
    gr_k = [partial_x(rho_12_k), partial_y(rho_12_k)]
    #other portion
    gp_k = [fft2(W_1*ifft2(partial_x(w1_wphi_k))) - fft2(gW_1[0]*W_phi),
            fft2(W_1*ifft2(partial_y(w1_wphi_k))) - fft2(gW_1[1]*W_phi)]
    #Combine
    u_k = [Poisson_Solve(sigma*gr_k[0] + gp_k[0]),
           Poisson_Solve(sigma*gr_k[1] + gp_k[1])]

    return [project_hermite(u_k[0],L), project_hermite(u_k[1],L)]
    #then return Hermite velocity

def smoothed_density(density, XX,L):
    rho_k = fft2(density(XX[0], XX[1]))
    return project_hermite(rho_k, L)


def energy(f_k):
    # assuming unnormalized DFT
    s = np.shape(f_k)
    F = f_k.reshape([s[0]*s[1],])
    energy = 0
    for i in range(len(F)):
        energy += abs(F[i]*F[i].conj())
    return (4*np.pi**2)*energy/(s[0]**2*s[1]**2)


def energy_spectrum(f_k):
    size = int(np.shape(f_k)[0]/2)
    Ek = np.zeros((size))
    for i in range(size):
        for j in range(size):
            k = int(np.floor(np.sqrt(i**2 + j**2)))
            if (k < size) and (k > 0):
                Ek[k] += (np.sqrt(f_k[i,j]*f_k[i,j].conjugate()))/(k**2)
    return Ek
