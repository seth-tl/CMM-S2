import numpy as np
import pdb, pickle
from InterpT2 import Hermite_T2, Hermite_Map
import fourier_space_tools as fspace
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

from datetime import datetime
import pdb
import sys
np.set_printoptions(threshold=sys.maxsize)

#-----------------------------------------------------------------------------

"""
What does this script do?

This script is implements the Characteristic mapping method on the sphere

"""

#------------------------------------------------------------------------------

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

pi = np.pi

#---------------------------- Routines ----------------------------------------
# ALgorithms for Characteristic Mapping Method
# Some useful integrators
def Euler_Step(V, y0, t, dt):
    x_n = y0.copy()
    return x_n-dt*V(t, x_n)


def Improved_Euler(V, y0, t, dt):
    xn = np.array(y0.copy()); tpdt = t + dt
    # step1:
    v1 = V(t,xn)
    step1 =  xn-dt*v1
    #project
    v2 = V(t+dt,np.array(step1))
    return xn-(dt/2)*(v1 + v2)


def OneStepSO(V, y0, t, dt):

    yn = y0.copy()  # Don't destroy initial conditions
    tpdt = t + dt
    #backwards RK3-TVD
    step1 = yn - dt*V(tpdt, yn)
    step2 = yn - dt*((1/4)*V(tpdt, yn) + \
                     (1/4)*V(t, step1))
    yn1 = yn - dt*((1/6)*V(tpdt, yn) + \
                   (1/6)*V(t, step1) + \
                   (2/3)*V(t + 0.5*dt, step2))

    return yn1

def epsilon_diff(H1, H2, H3, H4, eps = 1e-5, eps_c = False):

    """
    H1-H4 can be lists of stencil points. If it is a list then the stencil
    pts correspond to the stencil for the gradient and mixed derivatives
    separately.
    """

    # if eps_c == False:
    f = (1/4)*(H1 + H2 + H3 + H4)
    f_p = (1/(4*eps))*((H3 - H1) + (H4 - H2))
    f_t = (1/(4*eps))*((H2 - H1) + (H4 - H3))
    f_pt = (1/(4*(eps**2)))*(H4 - H3 - H2 + H1)
    # else:
    #     f = (1/4)*(H1[0] + H2[0] + H3[0] + H4[0])
    #     f_p = (1/(4*eps))*((H3[0] - H1[0]) + (H4[0] - H2[0]))
    #     f_t = (1/(4*eps))*((H2[0] - H1[0]) + (H4[0] - H3[0]))
    #     f_pt = (1/(4*(eps_c**2)))*(H4[1] - H3[1] - H2[1] + H1[1])


    return [f, f_p, f_t, f_pt]

def Hermite_stencil_eval(interpolant, eps1, eps2, eps3, eps4, yn):

    H1 = interpolant.stencil_eval(phi = yn[0], theta = yn[1],
                          phi_s = eps1[0], theta_s = eps1[1])
    H2 = interpolant.stencil_eval(phi = yn[0], theta = yn[1],
                          phi_s = eps2[0], theta_s = eps2[1])
    H3 = interpolant.stencil_eval(phi = yn[0], theta = yn[1],
                          phi_s = eps3[0], theta_s = eps3[1])
    H4 = interpolant.stencil_eval(phi = yn[0], theta = yn[1],
                          phi_s = eps4[0], theta_s = eps4[1])

    # pdb.set_trace()
    # without extension
    # H1 = interpolant.eval(eps1[0], eps1[1])
    # H2 = interpolant.eval(eps2[0], eps2[1])
    # H3 = interpolant.eval(eps3[0], eps3[1])
    # H4 = interpolant.eval(eps4[0], eps4[1])


    return [H1, H2, H3, H4]


def Advect(Chi, V, t, dt, y0, spts, Integrator, eps = 1e-5, deriv = False):

    #integrate for foot points, one step map
    eps1 = Integrator(V, spts[0], t, dt)
    eps2 = Integrator(V, spts[1], t, dt)
    eps3 = Integrator(V, spts[2], t, dt)
    eps4 = Integrator(V, spts[3], t, dt)
    yn = Integrator(V, y0, t, dt)

    # only evaluated the displacement component
    Chi_x_n1 = Hermite_stencil_eval(Chi.Chi_x, eps1, eps2, eps3, eps4, yn)
    Chi_y_n1 = Hermite_stencil_eval(Chi.Chi_y, eps1, eps2, eps3, eps4, yn)

    # Displacement
    dX1 = eps1 - spts[0]; dX2 = eps2 - spts[1]; dX3 = eps3 - spts[2]; dX4 = eps4 - spts[3]
    # pdb.set_trace()
    # dX1 = -spts[0]; dX2 = -spts[1]; dX3 = -spts[2]; dX4 = -spts[3]
    H_phi_n = Chi_x_n1 + np.array([dX1[0], dX2[0], dX3[0], dX4[0]])
    H_the_n = Chi_y_n1 + np.array([dX1[1], dX2[1], dX3[1], dX4[1]])

    F_phi = epsilon_diff(H_phi_n[0], H_phi_n[1], H_phi_n[2], H_phi_n[3], eps)
    F_the = epsilon_diff(H_the_n[0], H_the_n[1], H_the_n[2], H_the_n[3], eps)

    H_phi_new = Hermite_T2(Chi.Chi_x.phi, Chi.Chi_x.theta, F_phi[0], F_phi[1], F_phi[2], F_phi[3])
    H_the_new = Hermite_T2(Chi.Chi_y.phi, Chi.Chi_y.theta, F_the[0], F_the[1], F_the[2], F_the[3])

    return Hermite_Map(H_phi_new, H_the_new, identity = False)

def Initialize_Euler(u0, omega_0, dt, int0, XX, y0, spts, eps, L):

    U = velocity_interp(Vs = [u0], t0 = 0, dt = dt)
    inv_map1 = Advect(int0, U, 0, dt, y0, spts, Euler_Step, eps = eps)

    u1 = fspace.Biot_Savart(fft2(omega_0(inv_map1(XX))),L)
    U.Vs.append(u1)

    inv_map2 = Advect(int0, U, 0, dt, y0, spts, Improved_Euler, eps = eps)

    u2 = fspace.Biot_Savart(fft2(omega_0(inv_map2(XX))),L)

    U.Vs[1] = u2

    #then one more
    inv_map3 = Advect(inv_map2, U, dt, dt, y0, spts, OneStepSO, eps = eps)

    u3 = fspace.Biot_Savart(fft2(omega_0(inv_map3(XX))),L)

    U.Vs.append(u3)

    return U, inv_map3

def Initialize_horizontal_vector_field(u0, dt, int0, XX, y0, spts, eps, angle, rho, L):

    U = velocity_interp(Vs = [u0], t0 = 0, dt = dt)
    inv_map1 = Advect(int0, U, 0, dt, y0, spts, Euler_Step, eps = eps)

    u1 = gradient_field(angle, rho, s = dt, q_pts = inv_map1(XX), L = L)
    U.Vs.append(u1)

    inv_map2 = Advect(int0, U, 0, dt, y0, spts, Improved_Euler, eps = eps)

    u2 = gradient_field(angle, rho, s = dt, q_pts =  inv_map2(XX), L = L)
    U.Vs[1] = u2

    #then one more
    inv_map3 = Advect(inv_map2, U, dt, dt, y0, spts, OneStepSO, eps = eps)

    u3 = gradient_field(angle, rho, s = 2*dt, q_pts = inv_map3(XX), L = L)
    U.Vs.append(u3)

    return U, inv_map3

def gradient_field(theta, rho, s, q_pts, L):
    """
    gradient vector field to perform volume preserving projection
    theta = \int \sqrt{rho}/vol(M)
    s = time parameter
    """
    s_rho = np.sqrt(rho(q_pts[0], q_pts[1]))
    sigma = sin((1-s)*theta)/sin(theta) + (sin(s*theta)/sin(theta))*s_rho
    sigma_dot = -theta*cos((1-s)*theta)/sin(theta) + theta*(cos(s*theta)/sin(theta))*s_rho
    f_samples_k = fspace.Poisson_Solve(fft2(2*sigma_dot/sigma))


    return fspace.project_gradient_field(f_samples_k,L)


def eval_vector_Hermite(Hs, x,y):
    return np.array([Hs[0].eval(x,y), Hs[1].eval(x,y)])


class velocity_interp():
    """
    class to perform interpolation in time for gradient vector fields.
    The primary object is a list of potentials, evaluation is performed taking
    the gradient of the Hermite basis.
    """


    def __init__(self, Vs, t0, dt):

        self.Vs = Vs #list of Hermite_T2 objects
        self.t0 = t0 #start time of the interpolant
        self.dt = dt
        return

    def __call__(self, t, xy):

        if len(self.Vs) == 1:
            #self.Vs[0](t,dt,l_n) #
            return eval_vector_Hermite(self.Vs[0], xy[0], xy[1])
            #return list(map(lambda x: cos(pi*t/T)*x.real, v_out))

        if len(self.Vs) == 2:
            dt = self.dt
            #input l_n should be in spherical coordinates
            tau = (t - self.t0)/dt

            # evaluate in space:
            v0 = eval_vector_Hermite(self.Vs[0], xy[0], xy[1])
            v1 = eval_vector_Hermite(self.Vs[1], xy[0], xy[1])

            return (1-tau)*v0 + tau*v1

        if len(self.Vs) == 3:
            dt = self.dt
            tau0, tau1, tau2 = self.t0, self.t0 + dt, self.t0 + 2*dt
            l0 = (t-tau1)*(t-tau2)/(2*dt**2)
            l1 = (t-tau0)*(t-tau2)/(-dt**2)
            l2 = (t-tau0)*(t-tau1)/(2*dt**2)

            v0 = eval_vector_Hermite(self.Vs[0], xy[0], xy[1])
            v1 = eval_vector_Hermite(self.Vs[1], xy[0], xy[1])
            v2 = eval_vector_Hermite(self.Vs[2], xy[0], xy[1])

            return l0*v0 + l1*v1 + l2*v2


class velocity_interp_curl():
    """
    class to perform interpolation in time for gradient vector fields.
    The primary object is a list of potentials, evaluation is performed taking
    the gradient of the Hermite basis.
    """


    def __init__(self, Vs, t0, dt):

        self.Vs = Vs #list of Hermite_T2 objects
        self.t0 = t0 #start time of the interpolant
        self.dt = dt
        return

    def __call__(self, t, xy):

        if len(self.Vs) == 1:
            #self.Vs[0](t,dt,l_n) #
            return self.Vs[0].eval_curl(xy[0], xy[1])
            #return list(map(lambda x: cos(pi*t/T)*x.real, v_out))

        if len(self.Vs) == 2:
            dt = self.dt
            #input l_n should be in spherical coordinates
            tau = (t - self.t0)/dt

            # evaluate in space:
            v0 = self.Vs[0].eval_curl(xy[0], xy[1])
            v1 = self.Vs[1].eval_curl(xy[0], xy[1])

            return (1-tau)*v0 + tau*v1

        if len(self.Vs) == 3:
            dt = self.dt
            tau0, tau1, tau2 = self.t0, self.t0 + dt, self.t0 + 2*dt
            l0 = (t-tau1)*(t-tau2)/(2*dt**2)
            l1 = (t-tau0)*(t-tau2)/(-dt**2)
            l2 = (t-tau0)*(t-tau1)/(2*dt**2)

            v0 = self.Vs[0].eval_curl(xy[0], xy[1])
            v1 = self.Vs[1].eval_curl(xy[0], xy[1])
            v2 = self.Vs[2].eval_curl(xy[0], xy[1])

            return l0*v0 + l1*v1 + l2*v2




def compose_maps(remaps, X):

    remaps_rev = remaps[::-1]
    XX_n1 = remaps_rev[0](X)

    for map in remaps_rev[1::]:
        XX_n = map(XX_n1)
        XX_n1 = XX_n.copy()
    return XX_n1
