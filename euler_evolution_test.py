import numpy as np
from numpy import sin, cos, pi
from InterpT2 import Hermite_T2, hermite_density, Hermite_Map
from evolution_functions import Advect, OneStepSO, compose_maps, velocity_interp, Initialize_Euler, velocity_interp_curl
import pickle
import pdb
import fourier_space_tools as fspace
import scipy.io
from scipy.fft import fftshift, ifftshift, fft2, ifft2

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

name = "four_modes"
def omega_0(X):
    return cos(X[0]) + cos(X[1]) + 0.6*cos(2*X[0]) + 0.2*cos(3*X[0])

def tracer(x,y):
    return cos(x*10)*sin(y*7)
#--------------------------------- Setup ---------------------------------------

# finer grid for evaluation
phi_finer = np.linspace(0, 2*pi, 1024, endpoint = False)
theta_finer = np.linspace(0, 2*pi, 1024, endpoint = False)
# np.random.seed(303)
# phi_rand = np.random.uniform(-pi, pi, 1000)
# the_rand = np.random.uniform(-pi, pi, 1000)
# XX_rand = np.meshgrid(phi_rand, the_rand)
XX = np.meshgrid(phi_finer, theta_finer)
# ------------------------------
# Sampling grid
L = 512
phi_F = np.linspace(0, 2*pi, L, endpoint = False)
theta_F = np.linspace(0, 2*pi, L, endpoint = False)
XX_F = np.meshgrid(phi_F, theta_F)


Ns = np.array([16, 32, 64, 128, 256, 512])

L_inf = []
l_2 = []
#----------- Initialize Algorithm -------------------------------------------


omg_k0 = fspace.energy(fft2(omega_0(XX)))

vort_init = omega_0(XX_F)
u0 = fspace.Biot_Savart(fft2(vort_init), L)

l_inf = []
l_2 = []

for N in Ns:
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
    U, inv_map_n = Initialize_Euler(u0, omega_0, dt, Inverse_Map, XX_F, y0, spts, eps, L)
    ii = 1
    remaps = []
    for t in tspan[2::]:
        print(t)
        inv_map_n = Advect(inv_map_n, U, t, dt, y0, spts, OneStepSO, eps = eps)

        omg_n = omega_0(inv_map_n(XX_F))

        # if len(remaps) == 0:
        #     omg_n = omega_0(inv_map_n(XX_F))
        # else:
        #     omg_n = omega_0(compose_maps(remaps + [inv_map_n], XX_F))

        u_n = fspace.Biot_Savart(fft2(omg_n), L)
        U = velocity_interp(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, dt = dt)
        ii +=1

        # if ii % 10 == 0:
        #     print("remap")
        #     remaps.append(inv_map_n)
        #     inv_map_n = Hermite_Map(xs, ys, identity = True)
        #     jr = True

        if N == 64:
            scipy.io.savemat("./data/euler_simulations/%s_T%s_L%s_N%s_iteration_%s.mat" %(name,T,L,N,ii), {"omg_n": omg_n})


    # if jr == False:
    #     omg_n = omega_0(compose_maps(remaps + [inv_map_n], XX))
    # else:
    #     omg_n = omega_0(compose_maps(remaps, XX))

    omg_n_fine = omega_0(inv_map_n(XX))
    omg_k = fspace.energy(fft2(omg_n_fine))

    error = np.absolute(omg_k0 - omg_k) #/np.sum(u_k0)
    linf_err = np.max(np.absolute(omg_n-vort_init))

    l_2.append(error)
    l_inf.append(linf_err)
    print(l_2, l_inf)




# l2 = np.array(l_2)
#
# orders = np.log(l2[1::]/l2[0::-1])/np.log(Ns[1::]/Ns[0::-1])
