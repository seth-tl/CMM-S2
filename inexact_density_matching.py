import numpy as np
from numpy import sin, cos, pi
from InterpT2 import Hermite_T2, hermite_density
from evolution_functions import Advect_Project, horizontal_velocity_interp, Euler_Step, Improved_Euler, OneStepSO
import fourier_space_tools as fourier
import pickle
import pdb

import scipy.io
from datetime import datetime


def identity(phi,theta):
    return np.ones(np.shape(phi))

def identity_density(x,y):
    return 0*x + 1
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# define the velocity field

# define intial density to match

# def rho(x,y):
#     return 1 + 0.01*sin(10*x)*cos(2*y)
#
# #define angle
# angle = np.arccos(39.4783/(4*pi**2))

def mu_0(x,y):
    return 1 + 0.01*cos(x)

def mu_1(x,y):
    return 1 + 0.01*cos(4*y)*sin(2*x)

def gmu_1(x,y):
    return [0.01*2*cos(4*y)*cos(2*x), -0.01*4*sin(4*y)]

#--------------------------------- Setup ---------------------------------------

# finer grid for evaluation
L = 256
phi_finer = np.linspace(-pi, pi, L, endpoint = False)
theta_finer = np.linspace(-pi, pi, L, endpoint = False)

XX = np.meshgrid(phi_finer, theta_finer)
[Phi, Theta] = XX

Ns = np.array([16, 32, 64, 128, 256, 512])


#----------- Initialize Algorithm -------------------------------------------

from evolution_functions import gradient_field, horizontal_velocity_interp, Initialize_horizontal_vector_field, OneStepSO

L_inf = []
l_2 = []

for N in Ns:
    # an evenly spaced grid first
    phi_grid = np.linspace(-pi, pi, N, endpoint = False) # for interpolant
    theta_grid = np.linspace(-pi, pi, N, endpoint = False) # for interpolant
    dx = abs(phi_grid[1] - phi_grid[0])
    Phi, Theta = np.meshgrid(phi_grid, theta_grid)
    X0 = [Phi, Theta]

    #initial displacement- CMM
    ident = identity(Phi, Theta)
    H_phi0 = Hermite_T2(phi = phi_grid, theta = theta_grid,
                          f = 0*ident, f_x = 0*ident,
                          f_y = 0*ident, f_xy = 0*ident)

    H_the0 = Hermite_T2(phi = phi_grid, theta = theta_grid,
                          f = 0*ident, f_x = 0*ident,
                          f_y = 0*ident, f_xy = 0*ident)

    # set up parameters
    tspan = np.linspace(0, 1, N, endpoint = False)
    dt = abs(tspan[1]-tspan[0])
    # Solve advection equation:
    y0 = X0.copy()
    startTime = datetime.now()
    eps = 1e-5
    spts = [np.array([y0[0] - eps, y0[1] - eps]), np.array([y0[0] - eps, y0[1] + eps]),
            np.array([y0[0] + eps, y0[1] - eps]), np.array([y0[0] + eps, y0[1] + eps])]

    # Initialize velocity field
    W_phi = np.sqrt(mu_0(XX[0], XX[1]))
    W_1 = np.sqrt(mu_1(XX[0], XX[1]))
    gradw = gmu_1(XX[0], XX[1])
    gW_1 = [0.5*gradw[0]/W_1, 0.5*gradw[1]/W_1]
    sigma = 0.01
    u0 = fourier.information_gradient(W_phi, W_1, gW_1, identity_density(XX[0], XX[1]), sigma, XX, L)
    U = horizontal_velocity_interp(Vs = [u0], t0 = 0, dt = dt)

    # step 1:
    H_phi1, H_the1 = Advect_Project(H_phi0, H_the0, U, 0, dt, y0, spts, Euler_Step, eps = eps)

    qpts1 = [XX[0] + H_phi1.eval(XX[0], XX[1]), XX[1] + H_the1.eval(XX[0], XX[1])]
    rho = hermite_density(H_phi1 ,H_the1)
    rho_sq =  np.sqrt(rho(XX[0], XX[1]))
    W_phi = rho_sq*np.sqrt(mu_0(qpts1[0], qpts1[1]))
    u1 = fourier.information_gradient(W_phi, W_1, gW_1, rho_sq, sigma, XX, L)
    U.Vs.append(u1)

    #step 2
    H_phi2, H_the2 = Advect_Project(H_phi0, H_the0, U, 0, dt, y0, spts, Improved_Euler, eps = eps)
    qpts2 = [XX[0] + H_phi2.eval(XX[0], XX[1]), XX[1] + H_the2.eval(XX[0], XX[1])]

    rho = hermite_density(H_phi2,H_the2)
    rho_sq =  np.sqrt(rho(XX[0], XX[1]))
    W_phi = rho_sq*np.sqrt(mu_0(qpts2[0], qpts2[1]))
    u2 = fourier.information_gradient(W_phi, W_1, gW_1, rho_sq, sigma, XX, L)
    U.Vs[1] = u2

    #step 3
    H_phi_n1, H_the_n1 = Advect_Project(H_phi2, H_the2, U, dt, dt, y0, spts, OneStepSO, eps = eps)
    qpts3 = [XX[0] + H_phi_n1.eval(XX[0], XX[1]), XX[1] + H_the_n1.eval(XX[0], XX[1])]

    rho = hermite_density(H_phi_n1, H_the_n1)
    rho_sq =  np.sqrt(rho(XX[0], XX[1]))
    W_phi = rho_sq*np.sqrt(mu_0(qpts3[0], qpts3[1]))
    u3 = fourier.information_gradient(W_phi, W_1, gW_1, rho_sq, sigma, XX, L)
    U.Vs.append(u3)

    for t in tspan[2::]:
        print(t)
        H_phi, H_the = Advect_Project(H_phi_n1, H_the_n1, U, t, dt, y0, spts, OneStepSO, eps = eps)
        H_phi_n1 = H_phi
        H_the_n1 = H_the

        qpts_n = [XX[0] + H_phi.eval(XX[0], XX[1]), XX[1] + H_the.eval(XX[0], XX[1])]

        #reconstruct velocity field
        rho = hermite_density(H_phi_n1, H_the_n1)
        rho_sq = np.sqrt(rho(XX[0], XX[1]))
        W_phi = rho_sq*np.sqrt(mu_0(qpts_n[0], qpts_n[1]))
        u_n = fourier.information_gradient(W_phi, W_1, gW_1, rho_sq, sigma, XX, L)
        U = horizontal_velocity_interp(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, dt = dt)

    file = open('./data/density_correction_inexact_matching_N%s.txt' %N, "wb")
    pickle.dump([H_phi, H_the], file)
    #------ Check Quality of Solution -----
    rho_s = rho(qpts_n[0], qpts_n[1])


    print("error:", np.max(np.absolute(rho_s*mu_0(qpts_n[0], qpts_n[1]) - mu_1(XX[0], XX[1]))))
