# Code to perform the double Fourier transform
import numpy as np
#import pyfftw
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
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
    I = np.identity(m)
    n_freqs = ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])
    m_freqs = -(np.linspace(-m/2, m/2, m + 1)[:-1])**2

    def D_m(n):
        if n != 0.:
            freqs = -n**2 + m_freqs
            return np.diag(1/freqs, k = 0)

        if n == 0.:
            freqs = m_freqs.copy()
            freqs[int(m/2)] = 1.
            return np.diag(1/freqs, k= 0)

    U_k = np.empty([m,m], dtype = "complex128")
    j = 0

    for n in n_freqs:
        U_k[:,j] = ifftshift(Mmul(D_m(n), fftshift(f_k[:,j])))
        j+=1


    return U_k


def partial_x(u_k):
    # uu = fft2(u)
    uu = u_k.copy()
    m = len(u_k[:,0])
    n_freqs = 1j*ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])
    n_freqs[0] = 1.
    j = 0
    for n in n_freqs:
        uu[:,j] = n*uu[:,j]
        j+=1
    return uu

def partial_y(u_k):
    uu = u_k.copy()
    m = len(u_k[:,0])
    n_freqs = 1j*ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])
    n_freqs[0] = 1.
    j = 0
    for n in n_freqs:
        uu[j,:] = n*uu[j,:]
        j+=1
    return uu


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


    xs = np.linspace(0, 2*pi,L, endpoint = False)
    ys = np.linspace(0, 2*pi,L, endpoint = False)

    u_x = Hermite_T2(phi = xs, theta = ys, f = ifft2(psi_xk).real,
                     f_x = ifft2(psi_xxk).real, f_y = ifft2(psi_xyk).real,
                     f_xy = ifft2(psi_xxyk).real)

    u_y = Hermite_T2(phi = xs, theta = ys, f = ifft2(psi_yk).real,
                     f_x = ifft2(psi_yxk).real, f_y = ifft2(psi_yyk).real,
                     f_xy = ifft2(psi_yyxk).real)

    return [u_x, u_y]

def project_hermite(u_k, L):

    u_xk = partial_x(u_k)
    u_yk = partial_y(u_k)
    u_xyk = partial_x(u_yk)


    xs = np.linspace(0, 2*pi,L, endpoint = False)
    ys = np.linspace(0, 2*pi,L, endpoint = False)

    return Hermite_T2(phi = xs, theta = ys, f = ifft2(u_k).real,
                     f_x = ifft2(u_xk).real, f_y = ifft2(u_yk).real,
                     f_xy = ifft2(u_xyk).real)

def Biot_Savart(omega_k, xs, ys):
    psi_k = Poisson_Solve(omega_k)
    L = len(xs)
    return [project_hermite(partial_y(psi_k),L), project_hermite(partial_x(-psi_k),L)]

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


def energy_spectrum(omega_k):
    size = int(np.shape(omega_k)[0]/2)
    Ek = np.zeros((size))
    for i in range(size):
        for j in range(size):
            k = int(np.floor(np.sqrt(i**2 + j**2)))
            if (k < size) and (k > 0):
                Ek[k] += (np.absolute(omega_k[i,j])**2)/(k**2)
    return Ek

# #----------------------------------------------------------------------------
# class Poisson_Solver():
#     """
#     Methods:
#     eval(): Solves an equation of the form \Delta u = g,
#     Outputs the harmonic coefficients of u based on the DFS method or the
#     the function in spatial domain.
#     Input harmonic transformed g.
#
#     n = longitudinal wavenumber
#     m = latitudinal wavenumber
#
#     partial_x, partial_y
#
#     Laplace():
#
#     """
#     def __init__(self, m, n):
#         """
#         One first initialization of the matrices needed to solve Poisson eq
#         """
#         # first the T_sin^2(theta) matrix
#         self.m = m
#         P0 = np.identity(m)
#         P0[0,0] = 0.5
#         x0 = np.zeros([1,m])
#         P = np.vstack((P0,x0))
#         P[-1,0] = 0.5
#         self.P = P.copy()
#         self.n_freqs = ifftshift(np.linspace(-n/2, n/2, n + 1)[:-1])
#         self.m_freqs = np.linspace(-m/2, m/2, m + 1)
#         D_m1 = 1j*np.diag(self.m_freqs, k = 0)
#         Q = np.identity(m+1)
#         Q[0,-1] = 1
#         self.Q = Q[:-1, :].copy()
#         self.D_m = Mmul(self.Q, Mmul(D_m1, P)) # will be the same as D_n
#         self.D_2m = Mmul(D_m1, D_m1)[:-1,:-1]
#         x1 = np.ones((m+1+4,))
#         M_s2 = np.diag((1/2)*x1, k = 0) + np.diag((-1/4)*x1[:-2], k = 2) + np.diag((-1/4)*x1[:-2], k = -2)
#         Q2 = np.hstack((np.zeros([m,2]), self.Q, np.zeros([m,2])))
#         self.T_s2 = Mmul(Q2, Mmul(M_s2[:, 2:-2], P))
#         self.T_s2_inv = inv(Mmul(Q2, Mmul(M_s2[:, 2:-2], P)))
#         M_cs = np.diag(0*x1, k = 0) + np.diag((1j/4)*x1[:-2], k = 2) + np.diag((-1j/4)*x1[:-2], k = -2)
#         self.T_cs = Mmul(Q2, Mmul(M_cs[:, 2:-2], P))
#
#         #Need matrix representing 1/sin(theta):
#         x3 = np.ones((m+1+2,))
#         Sin = np.diag(x3*0, k = 0) + np.diag((1j/2)*x3[:-1], k = -1) + np.diag((-1j/2)*x3[:-1], k = 1)
#         Q3 = np.hstack((np.zeros([m,1]), self.Q, np.zeros([m,1])))
#         Cos = np.diag(x3*0, k = 0) - np.diag((1/2)*x3[:-1], k = -1) - np.diag((1/2)*x3[:-1], k = 1)
#         self.T_sin = Mmul(Q3, Mmul(Sin[:,1:-1], P))
#         self.T_cos = Mmul(Q3, Mmul(Cos[:,1:-1], P))
#
#         return
#
#
#     def L_n(self, n):
#         #multiplied by T_s2
#         return Mmul(self.T_s2, self.D_2m) +  Mmul(self.T_cs, self.D_m) - (n**2)*np.identity(self.m)
#
#     def Sin_inv(self, X, which = "col"):
#         # uu = X.copy()
#         # if which == "col":
#         #     for n in range(len(X[:,0])):
#         #         uu[:,n] = ifftshift(Mmul(self.T_s2_inv, Mmul(self.T_s, fftshift(uu[:,n]))))
#         #     return ifft2(uu)
#         # else:
#         #     for n in range(len(X[:,0])):
#         #         uu[n,:] = ifftshift(Mmul(self.T_s2_inv, Mmul(self.T_s, fftshift(uu[n,:]))))
#         #     return ifft2(uu)
#         return ifftshift(Mmul(self.T_s2_inv, Mmul(self.T_sin, fftshift(X))))
#
#     def Tmul(self, A, B, axis = "col"):
#         out = B.copy()
#         N = len(B[:,0]) # Assuming everything is square
#         if axis == "col":
#             for n in range(N):
#                 out[:,n] = ifftshift(Mmul(A, fftshift(out[:,n])))
#         if axis == "row":
#             for n in range(N):
#                 out[n, :] = ifftshift(Mmul(A, fftshift(out[n, :])))
#
#         return out
#
#     def Cart_Grad(self, u_k):
#         # Grad\dot e_x
#         uu_1 = self.Tmul(self.T_cos,
#                          self.Tmul(self.D_m, u_k.copy(), axis = "col"), axis = "col")
#         uu_1 = self.Tmul(self.T_cos, uu_1, axis = "row")
#
#         uu_2 = self.Tmul(self.T_sin,
#                          self.Tmul(self.D_m, u_k.copy(), axis = "row"), axis = "row")
#
#         uu_2 = self.Tmul(self.T_s2_inv, self.Tmul(self.T_sin, uu_2, axis = "col"), axis = "col")
#
#         out_x = -ifft2(uu_2) + ifft2(uu_1)
#
#         #-----------
#         uy_1 = self.Tmul(self.T_cos,
#                          self.Tmul(self.D_m, u_k.copy(), axis = "col"), axis = "col")
#         uy_1 = self.Tmul(self.T_sin, uy_1, axis = "row")
#
#         uy_2 = self.Tmul(self.T_cos,
#                          self.Tmul(self.D_m, u_k.copy(), axis = "row"), axis = "row")
#
#         uy_2 = self.Tmul(self.T_s2_inv, self.Tmul(self.T_sin, uy_2, axis = "col"), axis = "col")
#
#         out_y = ifft2(uy_2) + ifft2(uy_1)
#
#         #------------
#
#         uu_1_z = self.Tmul(self.T_sin, self.Tmul(self.D_m, u_k.copy(), axis = "col"), axis = "col")
#         out_z = -ifft2(uu_1_z)
#
#         # # # #fix the poles
#         N = len(out_x[:,0])
#         M = len(out_x[0,:])
#
#         # # # #South pole
#         # out_x[0,:] = (1/N)*np.sum(out_x[0,:])
#         # out_y[0,:] = (1/N)*np.sum(out_y[0,:])
#         # out_z[0,:] = (1/N)*np.sum(out_z[0,:])
#         # #
#         # # #North Pole
#         # out_x[int(N/2),:] = (1/N)*np.sum(out_x[int(N/2),:])
#         # out_y[int(N/2),:] = (1/N)*np.sum(out_y[int(N/2),:])
#         # out_z[int(N/2),:] = (1/N)*np.sum(out_z[int(N/2),:])
#
#         return [out_x, out_y, out_z]
#
#
#     def partial_x(self, u_k, freq = False):
#         uu = u_k.copy()
#         m = len(u_k[:,0])
#         nfreqs = 1j*fftshift(np.linspace(-m/2, m/2, m + 1)[:-1])
#         nfreqs[0] = 0.
#         j = 0
#         for n in nfreqs:
#             uu[:,j] = n*uu[:,j]
#             j+=1
#
#         if freq == False:
#             return ifft2(uu)
#         else:
#             return uu
#
#     def partial_y(self, u_k, freq = False):
#         uu = u_k.copy()
#         m = len(u_k[:,0])
#         n_freqs = 1j*ifftshift(np.linspace(-m/2, m/2, m + 1)[:-1])
#         n_freqs[0] = 0.#1.
#         j = 0
#         for n in n_freqs:
#             uu[j,:] = n*uu[j,:]
#             j+=1
#         if freq == False:
#             return ifft2(uu)
#         else:
#             return uu
#
#     def eval(self, f_k):
#         g = f_k.copy()
#         u = np.zeros(np.shape(g), dtype = "complex128")
#         i = 0
#         for n in self.n_freqs:
#
#             if n == 0:
#
#                 L_0 = self.L_n(0)
#
#                 def u_j(k):
#
#                     if k**2 != 1.0:
#                         return (1 + np.exp(1j*k*pi))/(1-k**2)
#                     else:
#                         return 0.
#
#                 uj0 = []
#                 for k in self.m_freqs[:-1]:
#                     uj0.append(2*pi*u_j(k))
#                     # if int(k) != -1 and int(k) != 1:
#                     #     uj0.append(2*pi*u_j(k))
#                     # else:
#                     #     uj0.append(0)
#
#                 #uj0[0] = uj0[0].real
#                 uj0[0] = 0.5*uj0[0]
#                 uj0[-1] = 0.5*uj0[-1]
#
#                 g_0 = fftshift(g[:,0])
#                 g_mean = 0
#                 for ii in range(self.m):
#                     g_mean += uj0[ii]*g_0[ii]
#
#                 #g_0[int(self.m/2)] = 0. #g_mean
#                 L_0[int(self.m/2), :] = uj0
#
#                 u[:, 0] = ifftshift(solve(L_0, Mmul(self.T_s2, g_0)))
#
#
#             if n != 0:
#
#                 u[:, i] = ifftshift(solve(self.L_n(n), Mmul(self.T_s2, fftshift(g[:,i]))))
#
#             i += 1
#
#         return u
#
#     # def well_define(self, f_k):
#     #     #in-place imposition of BMC condition on the coefficients.
#     #     N = len(f_k[0,:]) #! must be even
#     #     coeffs = list(np.arange(0,int(N/2))) + list(np.arange(-int(N/2),0))
#     #     for n in range(1,N): # so the zero component doesn't change
#     #         col_temp = fftshift(f_k[:,n]).copy()
#     #         col_temp[int(N/2)+1::] = (-1)**(abs(coeffs[n]))*col_temp[1:int(N/2)]
#     #
#     #         f_k[:,n] = ifftshift(col_temp)
#     #
#     #     return f_k
#
#
#     def Laplace(self, f_k):
#         # Laplace operator in Fourier space on f_k
#         g = f_k.copy()
#         u = np.zeros(np.shape(g), dtype = "complex128")
#         i = 0
#         for n in self.n_freqs:
#
#             if n == 0:
#
#                 L_0 = self.L_n(0)
#
#                 # def u_j(k):
#                 #
#                 #     if k**2 != 1.0:
#                 #         return (1 + np.exp(1j*k*pi))/(1-k**2)
#                 #     else:
#                 #         return 0.
#                 #
#                 # uj0 = []
#                 # for k in self.m_freqs[:-1]:
#                 #     if int(k) != -1 and int(k) != 1:
#                 #         uj0.append(2*pi*u_j(k))
#                 #     else:
#                 #         uj0.append(0)
#                 #
#                 # #uj0[0] = uj0[0].real
#                 # uj0[0] = 0.5*uj0[0]
#                 # uj0[-1] = 0.5*uj0[-1]
#                 #
#                 g_0 = fftshift(g[:,0])
#                 # g_mean = 0
#                 # for ii in range(self.m):
#                 #     g_mean += uj0[ii]*g_0[ii]
#                 #
#                 # g_0[int(self.m/2)] = 0. #g_mean
#                 # L_0[int(self.m/2), :] = uj0
#
#                 u[:, 0] = ifftshift(Mmul(self.T_s2_inv, Mmul(L_0, g_0)))
#
#             if n != 0:
#                 u[:, i] = ifftshift(Mmul(self.T_s2_inv, Mmul(self.L_n(n), fftshift(g[:,i]))))
#
#             i += 1
#
#         return u
#
#     def Implicit_Heat(self, f_k, dt, alpha, order = "linear"):
#         # Laplace operator in Fourier space on f_k
#         I = np.identity(self.m)
#         if order == "linear":
#             g = f_k.copy()
#             u = np.zeros(np.shape(g), dtype = "complex128")
#             i = 0
#             for n in self.n_freqs:
#
#                 if n == 0:
#
#                     L_0 = self.L_n(0)
#
#                     g_0 = fftshift(g[:,0])
#
#                     R0 = I - dt*alpha*Mmul(self.T_s2_inv,L_0)
#
#                     #u[:, 0] = ifftshift(Mmul(self.T_s2_inv, Mmul(L_0, g_0)))
#                     u[:, 0] = ifftshift(solve(R0, g_0))
#
#
#                 if n != 0:
#                     R = I - dt*alpha*Mmul(self.T_s2_inv, self.L_n(n))
#                     u[:, i] = ifftshift(solve(R, fftshift(g[:,i])))
#
#                 i += 1
#
#             return u
#         # if order == "third":
#         #     # Note that f_k should be a list in this instance
#         #     # Corresponding to f_k^n, f_k^{n-1}
#         #     g_n = f_k[0].copy()
#         #     g_nm1 = f_k[1].copy()
#         #     u = np.zeros(np.shape(g_n), dtype = "complex128")
#         #     i = 0
#         #     for n in self.n_freqs:
#         #         if n == 0:
#         #             L_0 = self.L_n(0)
#         #             g_0 = (4/3)*fftshift(g_n[:,0]) - (1/3)*fftshift(g_nm1[:,0])
#         #             R0 = I - (2/3)*dt*alpha*Mmul(self.T_s2_inv,L_0)
#         #             u[:, 0] = ifftshift(solve(R0, g_0))
#         #
#         #         if n != 0:
#         #             R = I - (2/3)*dt*alpha*Mmul(self.T_s2_inv, self.L_n(n))
#         #             g_nn = (4/3)*fftshift(g_n[:,i]) - (1/3)*fftshift(g_nm1[:,i])
#         #             u[:, i] = ifftshift(solve(R, g_nn))
#         #
#         #         i += 1
#         #
#         #     return u
#         #
#         if order == "third":
#             # Second-Order Crank-Nicolson Scheme
#             # Note that f_k should be a list in this instance
#             # Corresponding to f_k^n, f_k^{n-1}
#             g_n = f_k.copy()
#             u = np.zeros(np.shape(g_n), dtype = "complex128")
#             i = 0
#             for n in self.n_freqs:
#                 if n == 0:
#                     L_0 = self.L_n(0)
#                     g_0 = fftshift(g_n[:,0]) + \
#                      (1/2)*dt*alpha*Mmul(Mmul(self.T_s2_inv,L_0), g_n[:,0])
#
#                     R0 = I - (1/2)*dt*alpha*Mmul(self.T_s2_inv,L_0)
#                     u[:, 0] = ifftshift(solve(R0, g_0))
#
#                 if n != 0:
#                     R = I - (1/2)*dt*alpha*Mmul(self.T_s2_inv, self.L_n(n))
#                     g_nn = fftshift(g_n[:,i]) + \
#                      (1/2)*dt*alpha*Mmul(Mmul(self.T_s2_inv,self.L_n(n)), fftshift(g_n[:,i]))
#
#                     u[:, i] = ifftshift(solve(R, g_nn))
#
#                 i += 1
#
#             return u
#
#     def Allen_Cahn(self, f, dt, eps):
#
#         I = np.identity(self.m)
#
#         g = fft2(f.copy())
#         u = np.zeros(np.shape(g), dtype = "complex128")
#         i = 0
#         for n in self.n_freqs:
#         #Perform an operator splitting
#
#             if n == 0:
#
#                 L_0 = self.L_n(0)
#
#                 g_0 = fftshift(g[:,0])
#
#                 R0 = I - eps*dt*Mmul(self.T_s2_inv,L_0)
#
#                 #u[:, 0] = ifftshift(Mmul(self.T_s2_inv, Mmul(L_0, g_0)))
#                 gg0 = g_0 #+ dt*(g_0 - g_0**3)
#                 u[:, 0] = ifftshift(solve(R0, gg0))
#
#
#             if n != 0:
#                 R = I - eps*dt*Mmul(self.T_s2_inv, self.L_n(n))
#                 b = g[:,i] #+ dt*(g[:,i] - g[:,i]**3)
#                 u[:, i] = ifftshift(solve(R, fftshift(b)))
#
#             i += 1
#
#         uu = ifft2(u).real  # intermediate solution
#
#         temp = np.sqrt((np.exp(-2*dt) + (uu**2)*(1-np.exp(-2*dt))))
#         u_n1 = uu/temp
#
#         return u_n1
#
