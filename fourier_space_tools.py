# Code to perform the double Fourier transform
import numpy as np
#import pyfftw
from scipy.fft import fftshift, ifftshift, fft2, ifft2
from scipy.linalg import solve_banded, solve
from numpy.linalg import inv
from InterpT2 import Hermite_T2
import pdb

#Oft benutzt functions, expression, quantities----------------------------------
pi = np.pi

def sin(x):
    return np.sin(x)
def cos(x):
    return np.cos(x)
def Mmul(A,b):
    return  np.matmul(A,b)

#------------------------------------------------------------------------------

def Poisson_Solve(f_k):
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
    u1 = u_k.copy()
    m = len(u_k[:,0])
    n_freqs = 1j*ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])
    # n_freqs[0] = 1#0.
    j = 0
    for n in n_freqs:
        u1[:,j] = n*u1[:,j]
        j+=1
    return u1


def partial_y(u_k):
    u2 = u_k.copy()
    m = len(u_k[:,0])
    n_freqs = 1j*ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])
    # n_freqs[0] = 1#0.
    j = 0
    for n in n_freqs:
        u2[j,:] = n*u2[j,:]
        j+=1
    return u2


def partial_yy(u_k):
    u2 = u_k.copy()
    m = len(u_k[:,0])
    n_freqs = -ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])**2
    # n_freqs[0] = 1#0.
    j = 0
    for n in n_freqs:
        u2[j,:] = n*u2[j,:]
        j+=1
    return u2


def partial_xx(u_k):
    u1 = u_k.copy()
    m = len(u_k[:,0])
    n_freqs = -ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])**2
    # n_freqs[0] = 1#0.
    j = 0
    for n in n_freqs:
        u1[:,j] = n*u1[:,j]
        j+=1
    return u1


def project_gradient_field(psi_k, L):
    """
    Fourier Space projector for gradient vector field
    onto the space of Hermite cubics.
    Input: psi_k in Fourier space
    """
    # psi = ifft2(psi_k)
    # x- component:
    psi_xk = partial_x(psi_k)
    psi_xxk = partial_x(psi_xk)
    psi_xyk = partial_y(psi_xk)
    psi_xxyk = partial_y(psi_xxk)

    # y- component
    psi_yk = partial_y(psi_k)
    psi_yxk = partial_x(psi_yk)
    psi_yyk = partial_y(psi_yk)
    psi_yyxk = partial_x(psi_yyk)


    xs = np.linspace(0, 2*pi, L, endpoint = False)
    ys = np.linspace(0, 2*pi, L, endpoint = False)

    u_x = Hermite_T2(phi = xs, theta = ys, f = ifft2(psi_xk).real,
                     f_x = ifft2(psi_xxk).real, f_y = ifft2(psi_xyk).real,
                     f_xy = ifft2(psi_xxyk).real)

    u_y = Hermite_T2(phi = xs, theta = ys, f = ifft2(psi_yk).real,
                     f_x = ifft2(psi_yxk).real, f_y = ifft2(psi_yyk).real,
                     f_xy = ifft2(psi_yyxk).real)

    return [u_x, u_y]


def project_solenoidal_field(psi_k, L):
    """
    Fourier Space projector for gradient vector field
    onto the space of Hermite cubics.
    Input: psi_k in Fourier space
    """
    # psi = ifft2(psi_k)
    # x- component:
    psi_xk = partial_x(psi_k)
    psi_xxk = partial_x(psi_xk)
    psi_xyk = partial_y(psi_xk)
    psi_xxyk = partial_y(psi_xxk)

    # y- component
    psi_yk = partial_y(psi_k)
    psi_yxk = partial_x(psi_yk)
    psi_yyk = partial_y(psi_yk)
    psi_yyxk = partial_x(psi_yyk)


    xs = np.linspace(0, 2*pi, L, endpoint = False)
    ys = np.linspace(0, 2*pi, L, endpoint = False)

    #switch components
    u_y = Hermite_T2(phi = xs, theta = ys, f = ifft2(psi_xk).real,
                     f_x = ifft2(psi_xxk).real, f_y = ifft2(psi_xyk).real,
                     f_xy = ifft2(psi_xxyk).real)

    u_x = Hermite_T2(phi = xs, theta = ys, f = -1*ifft2(psi_yk).real,
                     f_x = -1*ifft2(psi_yxk).real, f_y = -1*ifft2(psi_yyk).real,
                     f_xy = -1*ifft2(psi_yyxk).real)

    return [u_x, u_y]
#
# def upsample(f_k, N_u):
#     # extend Fourier series by zero to length N_u series
#     # have to centre the array
#     f_kn = np.zeros([N_u, N_u], dtype = "complex128")
#     S = np.shape(f_k)
#     h = int((N_u - S[0])/2)
#     # broadcast to centre of array then ifftshift back
#     f_kn[h:-h, h:-h] = fftshift(f_k.copy())
#
#     return ifftshift(f_kn)

def project_hermite(u_k, L):

    u_xk = partial_x(u_k)
    u_yk = partial_y(u_k)
    u_xyk = partial_x(u_yk)


    xs = np.linspace(0, 2*pi,L, endpoint = False)
    ys = np.linspace(0, 2*pi,L, endpoint = False)

    return Hermite_T2(phi = xs, theta = ys, f = ifft2(u_k).real,
                     f_x = ifft2(u_xk).real, f_y = ifft2(u_yk).real,
                     f_xy = ifft2(u_xyk).real)

def Biot_Savart(omega_k, L):
    psi_k = Poisson_Solve(omega_k)
    return project_solenoidal_field(psi_k, L)

def Biot_Savart_scalar(omega_k, L):
    psi_k = Poisson_Solve(omega_k)
    return project_hermite(psi_k, L)

def L_arc(f_k):
    #computes the L operator from manuscript
    #approximate volume using the zero mode
    int_f = f_k[0,0]
    return np.arccos(int_f/(4*pi**2))/np.sqrt(1- (int_f/(4*pi**2))**2)

def information_gradient_alternative(W_phi, W_1, gW_1, rho, sigma, XX, L):
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
    rho_12_k = fft2(np.sqrt(rho(XX[0], XX[1])))
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

    return interpolate_velocity_field(u_k,L)
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

    return interpolate_velocity_field(u_k,L)
    #then return Hermite velocity


def energy(f_k):
    # assuming unnormalized transformation
    s = np.shape(f_k)
    F = f_k.reshape([s[0]*s[1],])
    energy = 0
    for i in range(len(F)):
        energy += abs(F[i]*F[i].conj())
    return (4*pi**2)*energy/(s[0]**2*s[1]**2)


def energy_spectrum(omega_k):
    size = int(np.shape(omega_k)[0]/2)
    Ek = np.zeros((size))
    for i in range(size):
        for j in range(size):
            k = int(np.floor(np.sqrt(i**2 + j**2)))
            if (k < size) and (k > 0):
                Ek[k] += (np.absolute(omega_k[i,j])**2)/(k**2)
    return Ek
