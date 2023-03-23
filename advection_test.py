import numpy as np
from numpy import sin, cos, pi
from InterpT2 import Hermite_T2, hermite_density, Hermite_Map
from evolution_functions import Advect, OneStepSO, compose_maps, velocity_interp, Euler_Step, Improved_Euler
import fourier_space_tools as fspace
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

import pickle
import pdb

import scipy.io
from datetime import datetime

#------------------------------------------------------------------------------
T = 1

def U(t,XY):
    u_max = 10
    x = XY[0] ; y = XY[1] ;
    ux = -u_max*sin(y)*cos(pi*t/T)
    uy = u_max*sin(x)*cos(pi*t/T) #+ 2*pi/T
    return np.array([ux, uy])


def tracer(X):
    return cos(X[0]*10)*sin(X[1]*8)
#--------------------------------- Setup ---------------------------------------
# finer grid for evaluation
phi_finer = np.linspace(0, 2*pi, 500, endpoint = False)
theta_finer = np.linspace(0, 2*pi, 500, endpoint = False)
# np.random.seed(303)
# phi_rand = np.random.uniform(0, 2*pi, 1000)
# the_rand = np.random.uniform(0, 2*pi, 1000)
# XX_rand = np.meshgrid(phi_rand, the_rand)
XX = np.meshgrid(phi_finer, theta_finer)
[Phi, Theta] = XX

def identity(phi,theta):
    return np.ones(np.shape(phi))

Ns = np.array([16, 32, 64, 128, 256, 512])

L_inf = []
l_2 = []

# ===== Integration with interpolated velocity field ===========================
tracer_true = tracer(XX)
#
def omega_0(x,y):
    return cos(x) + cos(y)

# def stream(x,y):
#
#     return -omega_0(x,y)
#
def velocity(t,xy):
    ux = -sin(xy[1])*cos(pi*t/T)
    uy = sin(xy[0])*cos(pi*t/T)
    return [ux, uy]

# Sampling grid
L = 512
phi_F = np.linspace(0, 2*pi, L, endpoint = False)
theta_F = np.linspace(0, 2*pi, L, endpoint = False)
XX_F = np.meshgrid(phi_F, theta_F)


for N in Ns:
    # an evenly spaced grid first
    xs = np.linspace(0, 2*pi, N, endpoint = False) # for interpolant
    ys = np.linspace(0, 2*pi, N, endpoint = False) # for interpolant
    dx = abs(xs[1] - xs[0])
    Phi, Theta = np.meshgrid(xs, ys)
    X0 = [Phi, Theta]

    #initial displacement- CMM
    inv_map = Hermite_Map(xs, ys, identity = True)
    tspan = np.linspace(0, T, 2*N, endpoint = False)

    dt = abs(tspan[1]-tspan[0])

    # Solve advection equation:
    y0 = X0.copy()
    startTime = datetime.now()
    eps = 1e-5
    spts = [np.array([y0[0] - eps, y0[1] - eps]), np.array([y0[0] - eps, y0[1] + eps]),
            np.array([y0[0] + eps, y0[1] - eps]), np.array([y0[0] + eps, y0[1] + eps])]

    # uu0 = velocity(0,XX_F)
    # u0 = [fspace.project_hermite(fft2(uu0[0]),L), fspace.project_hermite(fft2(uu0[1]),L)]
    #
    # U = velocity_interp(Vs = [u0], t0 = 0, dt = dt)
    #
    # uu1 = velocity(dt,XX_F)
    # u1 = [fspace.project_hermite(fft2(uu1[0]),L), fspace.project_hermite(fft2(uu1[1]),L)]
    # U.Vs.append(u1)
    #
    # uu2 = velocity(2*dt,XX_F)
    # u2 = [fspace.project_hermite(fft2(uu2[0]),L), fspace.project_hermite(fft2(uu2[1]),L)]
    #
    # U.Vs.append(u2)

    remaps = []
    ii = 0
    for t in tspan:
        jr = False
        inv_map = Advect(inv_map, U, t, dt, y0, spts, OneStepSO, eps = eps)

        # with remapping

        if ii % 10 == 0:
            remaps.append(inv_map)
            inv_map = Hermite_Map(xs, ys, identity = True)
            jr = True
        # #resample velocity
        # if ii > 2:
        #     uun = velocity(t,XX_F)
        #
        #     u_n = [fspace.project_hermite(fft2(uun[0]),L), fspace.project_hermite(fft2(uun[1]),L)]
        #     U = velocity_interp(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, dt = dt)

        if N == 64:
            tracer_num = tracer(compose_maps(remaps + [inv_map],XX))
            # tracer_num = tracer(inv_map(XX))
            scipy.io.savemat("./data/euler_simulations/advection_T%s_N%s_iteration_%s.mat" %(T,N,ii), {"omg_n": tracer_num})

        ii +=1

    if jr == False:
        tracer_num = tracer(compose_maps(remaps + [inv_map],XX))

    else:
        tracer_num = tracer(compose_maps(remaps,XX))

    # tracer_num = tracer(inv_map(XX))
    linf = np.absolute(tracer_num - tracer_true)
    scipy.io.savemat("./data/euler_simulations/advection_T%s_N%s_final.mat" %(T,N), {"omg_n": tracer_num, "omg_true": tracer_true})

    # rho = hermite_density(H_phi, H_the)
    # scipy.io.savemat("./data/density_correction_advection.mat", {"dens_i": tracer_true, "dens_f": rho(XX[0], XX[1]),
    #                                                                      "rho_f": tracer_num})

    #----------------------
    L_inf.append(np.max(linf))
    print(L_inf)
