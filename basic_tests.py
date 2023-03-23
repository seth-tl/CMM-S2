from InterpT2 import Hermite_T2
import numpy as np
import pdb
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from evolution_functions import Hermite_proj_eps, epsilon_diff

import fourier_space_tools as fspace

from numpy import cos, sin, pi
# Testing Script ===============================================================


Ns = [16, 32, 64, 128, 256, 512, 1024]


def F(x,y):
    return cos(10*x)*sin(4*y)

def dFx(x,y):

    return sin(4*y)*-sin(10*x)*10

def dFy(x,y):

    return cos(10*x)*4*cos(4*y)

def dFxy(x,y):

    return -10*sin(10*x)*4*cos(4*y)



# define fine grid for evaluation
phis = np.linspace(0,2*pi, 1000, endpoint = False)
thes = np.linspace(0,2*pi, 1000, endpoint = False)

XX_fine = np.meshgrid(phis, thes)


## Interpolation Test =======================================================
# u_true = F(XX_fine[0], XX_fine[1])
#
# for N in Ns:
#     #define mesh
#     xs = np.linspace(0,2*pi, N, endpoint = False)
#     ys = np.linspace(0,2*pi, N, endpoint = False)
#
#     XX = np.meshgrid(xs,ys)
#     #define interpolation data
#     interpolant = Hermite_T2(phi = xs, theta = ys, f = F(XX[0],XX[1]),
#                              f_x = dFx(XX[0], XX[1]), f_y = dFy(XX[0], XX[1]),
#                              f_xy = dFxy(XX[0], XX[1]))
#     u_num = interpolant.eval(XX_fine[0], XX_fine[1])
#
#     print(np.max(np.absolute(u_num - u_true)))
#
#     du_dx = interpolant.eval(XX_fine[0], XX_fine[1], deriv = "theta")
#     print(np.max(np.absolute(du_dx - dFy(XX_fine[0], XX_fine[1]))))

# ============== Poisson Solve Test ===========================================

#F will be solution, meaning source is
def Laplace_F(x,y):
    return -100*cos(10*x)*sin(4*y) - 16*cos(10*x)*sin(4*y)


for N in Ns:

    xs = np.linspace(0, 2*pi, N, endpoint = False)
    ys = np.linspace(0, 2*pi, N, endpoint = False)
    XX = np.meshgrid(xs,ys)

    source = Laplace_F(XX[0], XX[1])

    solution  = ifft2(fspace.Poisson_Solve(fft2(source)))
    solution_true = F(XX[0], XX[1])
    print(np.max(np.absolute(solution - solution_true)))

# ============ Hermite Projector Convergence ===================================

# #
# phis = np.linspace(0, 2*pi, 1000, endpoint = False)
# thes = np.linspace(0, 2*pi, 1000, endpoint = False)
#
# XX_fine = np.meshgrid(phis, thes)
#
# u_true = F(XX_fine[0], XX_fine[1])
# grad_y = dFy(XX_fine[0], XX_fine[1])
#
# for N in Ns:
#     #define mesh
#     xs = np.linspace(0, 2*pi, N, endpoint = False)
#     ys = np.linspace(0, 2*pi, N, endpoint = False)
#
#     XX = np.meshgrid(xs,ys)
#     #define interpolation data
#     # sample the solution
#     samples = F(XX[0], XX[1])
#     interpolant = fspace.project_hermite(fft2(samples), N)
#     u_num = interpolant.eval(XX_fine[0], XX_fine[1])
#
#     u_num_y = interpolant.eval(XX_fine[0], XX_fine[1], deriv = "theta")
#     print(np.max(np.absolute(u_num_y - grad_y)))


# ======= Biot_Savart Test ===================================================
# velocity from

# #F will be solution, meaning source is
# def Laplace_F(x,y):
#     return -100*cos(10*x)*sin(4*y) - 16*cos(10*x)*sin(4*y)
#
# def velocity(x,y):
#     return np.array([dFy(x,y), -dFx(x,y)])
#
# u_true = velocity(XX_fine[0], XX_fine[1])
#
# for N in Ns:
#
#     xs = np.linspace(-pi,pi, N, endpoint = False)
#     ys = np.linspace(-pi,pi, N, endpoint = False)
#     XX = np.meshgrid(xs,ys)
#
#     source = Laplace_F(XX[0], XX[1])
#
#     u_num = fspace.velocity_from_omega(fft2(source), xs, ys)
#     unum = np.array([u_num[0].eval(XX_fine[0], XX_fine[1]),
#                      u_num[1].eval(XX_fine[0], XX_fine[1])])
#
#     print(np.max(np.absolute(unum - u_true)))

# ====== Epsilon - Difference Test ============================================
#
# u_true = F(XX_fine[0], XX_fine[1])
#
# for N in Ns:
#     #define mesh
#     xs = np.linspace(0,2*pi, N, endpoint = False)
#     ys = np.linspace(0,2*pi, N, endpoint = False)
#
#     XX = np.meshgrid(xs,ys)
#     #define interpolation data
#     interpolant = Hermite_T2(phi = xs, theta = ys, f = F(XX[0],XX[1]),
#                              f_x = dFx(XX[0], XX[1]), f_y = dFy(XX[0], XX[1]),
#                              f_xy = dFxy(XX[0], XX[1]))
#
#     y0 = XX_fine.copy()
#     eps = 1e-5
#     spts = [np.array([y0[0] - eps, y0[1] - eps]), np.array([y0[0] - eps, y0[1] + eps]),
#             np.array([y0[0] + eps, y0[1] - eps]), np.array([y0[0] + eps, y0[1] + eps])]
#
#     Hs = Hermite_proj_eps(interpolant, spts[0], spts[1], spts[2], spts[3], y0)
#     outs = epsilon_diff(Hs[0], Hs[1], Hs[2], Hs[3], eps)
#
#
#     print(np.max(np.absolute(outs[0] - u_true)))
#
#     print(np.max(np.absolute(outs[2] - dFy(XX_fine[0], XX_fine[1]))))
