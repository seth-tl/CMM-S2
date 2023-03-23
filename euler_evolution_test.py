import numpy as np
from numpy import sin, cos, pi
from InterpT2 import Hermite_T2, hermite_density, Hermite_Map
from evolution_functions import Advect, OneStepSO, compose_maps, velocity_interp, Initialize_Euler
import pickle
import pdb
import fourier_space_tools as fspace
import scipy.io
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

from datetime import datetime

def identity(phi,theta):
    return np.ones(np.shape(phi))

#------------------------------------------------------------------------------
k = 2.
T = 5

# def U(t,XY):
#     x = XY[0]; y = XY[1];
#     ux = 2*k*sin(y)*sin(x)**2
#     uy = k*sin(2*x)*cos(y)
#     return cos(pi*t/T)*np.array([ux, uy])

def omega_0(X):
    x = X[0]; y = X[1];
    return cos(x) + cos(y) + 0.6*cos(2*x) + 0.2*cos(3*x)

def tracer(x,y):
    return cos(x*10)*sin(y*7)
#--------------------------------- Setup ---------------------------------------

# finer grid for evaluation
phi_finer = np.linspace(0, 2*pi, 500, endpoint = False)
theta_finer = np.linspace(0, 2*pi, 500, endpoint = False)
# np.random.seed(303)
# phi_rand = np.random.uniform(-pi, pi, 1000)
# the_rand = np.random.uniform(-pi, pi, 1000)
# XX_rand = np.meshgrid(phi_rand, the_rand)
XX = np.meshgrid(phi_finer, theta_finer)
[Phi, Theta] = XX
# ------------------------------
# Sampling grid
L = 256
phi_F = np.linspace(0, 2*pi, L, endpoint = False)
theta_F = np.linspace(0, 2*pi, L, endpoint = False)
XX_F = np.meshgrid(phi_F, theta_F)


Ns = np.array([16, 32, 64, 128, 256, 512])

L_inf = []
l_2 = []
#----------- Initialize Algorithm -------------------------------------------


u_k0 = fspace.energy_spectrum(fft2(omega_0(XX)))

vort_init = omega_0(XX_F)
u0 = fspace.Biot_Savart(fft2(vort_init), phi_F, theta_F)

L_inf = []
l_2 = []

for N in Ns[2::]:
    # define grid for the map
    xs = np.linspace(0, 2*pi, N, endpoint = False)
    ys = np.linspace(0, 2*pi, N, endpoint = False)
    dx = abs(xs[1] - xs[0])
    Phi, Theta = np.meshgrid(xs, ys)
    X0 = [Phi, Theta]

    #initial displacement- CMM
    Inverse_Map = Hermite_Map(xs, ys, identity = True)
    # set up parameters
    tspan = np.linspace(0, T, 2*N, endpoint = False)
    dt = abs(tspan[1]-tspan[0])
    # Solve advection equation:
    y0 = X0.copy()
    startTime = datetime.now()
    eps = 1e-5
    spts = [np.array([y0[0] - eps, y0[1] - eps]), np.array([y0[0] - eps, y0[1] + eps]),
            np.array([y0[0] + eps, y0[1] - eps]), np.array([y0[0] + eps, y0[1] + eps])]

    # Initialize velocity field
    U, inv_map_n = Initialize_Euler(u0, omega_0, dt, Inverse_Map, XX_F, y0, spts, eps, phi_F, theta_F)
    ii = 0
    for t in tspan[2::]:
        print(t)
        inv_map_n = Advect(inv_map_n, U, t, dt, y0, spts, OneStepSO, eps = eps)

        u_n = fspace.Biot_Savart(fft2(omega_0(inv_map_n(XX_F))),phi_F, theta_F)

        U = velocity_interp(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, dt = dt)

        if N == 64:
            omega_num = omega_0(inv_map_n(XX))
            scipy.io.savemat("./data/euler_simulations/four_modes_T%s_L%s_N%s_iteration_%s.mat" %(T,L,N,ii), {"omg_n": omega_num})
        ii +=1
    #
    omega_num = omega_0(inv_map_n(XX))
    u_k = fspace.energy_spectrum(fft2(omega_num))

    error = np.absolute(np.sum(u_k0) - np.sum(u_k))/np.sum(u_k0)

    print(error)
    l_2.append(error)



# l2 = np.array(l_2)
#
# orders = np.log(l2[1::]/l2[0::-1])/np.log(Ns[1::]/Ns[0::-1])
