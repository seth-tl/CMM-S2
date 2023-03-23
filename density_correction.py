import numpy as np
from numpy import sin, cos, pi
from InterpT2 import Hermite_T2
from evolution_functions import Advect_Project
import pickle
import pdb

import scipy.io
from datetime import datetime


def identity(phi,theta):
    return np.ones(np.shape(phi))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# define the velocity field

# define intial density to match

def rho(x,y):
    return 1 + 0.01*sin(10*x)*cos(2*y)

#define angle
angle = np.arccos(39.4783/(4*pi**2))

#--------------------------------- Setup ---------------------------------------

# finer grid for evaluation
L = 1024
phi_finer = np.linspace(-pi, pi, L, endpoint = False)
theta_finer = np.linspace(-pi, pi, L, endpoint = False)

XX = np.meshgrid(phi_finer, theta_finer)
[Phi, Theta] = XX


Ns = np.array([16, 32, 64, 128, 256, 512])


#----------- Initialize Algorithm -------------------------------------------

from evolution_functions import gradient_field, horizontal_velocity_interp, Initialize_horizontal_vector_field, OneStepSO

u0 = gradient_field(angle, rho, s = 0, q_pts = XX, L = L)


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
    tspan = np.linspace(0, 1, 2*N, endpoint = False)
    dt = abs(tspan[1]-tspan[0])

    # Solve advection equation:
    y0 = X0.copy()
    startTime = datetime.now()
    eps = 1e-5
    spts = [np.array([y0[0] - eps, y0[1] - eps]), np.array([y0[0] - eps, y0[1] + eps]),
            np.array([y0[0] + eps, y0[1] - eps]), np.array([y0[0] + eps, y0[1] + eps])]

    # Initialize velocity field
    U, [H_phi_n1, H_the_n1] = Initialize_horizontal_vector_field(u0, dt, [H_phi0, H_the0], XX, y0, spts, eps, angle, rho, L)


    for t in tspan[2::]:
        print(t)
        H_phi, H_the = Advect_Project(H_phi_n1, H_the_n1, U, t, dt, y0, spts, OneStepSO, eps = eps)
        H_phi_n1 = H_phi
        H_the_n1 = H_the

        qpts_n = [XX[0] + H_phi.eval(XX[0], XX[1]), XX[1] + H_the.eval(XX[0], XX[1])]

        u_n = gradient_field(angle, rho, s = t + dt, q_pts = qpts_n, L = L)
        U = horizontal_velocity_interp(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt, dt = dt)

    file = open('./data/density_correction_N%s.txt' %N, "wb")
    pickle.dump([H_phi, H_the], file)
    #------ Check Quality of Solution -----
    dChi = [1 + H_phi.eval(XX[0], XX[1], deriv = "phi"),
            H_phi.eval(XX[0], XX[1], deriv = "theta"),
            H_the.eval(XX[0], XX[1], deriv = "phi"),
            H_the.eval(XX[0], XX[1], deriv = "theta") + 1]

    J_c = np.absolute(dChi[0]*dChi[3] - dChi[1]*dChi[2])
    rho_s = rho(qpts_n[0], qpts_n[1])


    print("error:", np.max(np.absolute(rho_s*J_c - 1)))
