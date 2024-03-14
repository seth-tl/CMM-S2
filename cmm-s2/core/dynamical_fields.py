#/----
"""
This scripts provides the base classes for the spatio-temporal interpolants
defining the velocity fields. Along with defining the test velocity fields and
initial voriticity distributions used in the experimentation.
"""
#/----
# imports
import numpy as np
import pdb
from scipy.special import sph_harm 
from . import utils

# ------ Velocity field interpolants -------------------------------------------
class velocity_interp():
    """
    class to perform interpolation in time for velocity fields.
    """


    def __init__(self, Vs, t0):

        self.Vs = Vs #list of spline_interp_vec objects
        self.t0 = t0 #start time of the interpolant

        return

    def __call__(self, t, dt, xyz):
        # TODO: get rid of these if statements

        if len(self.Vs) == 3:
            tau0, tau1, tau2 = self.t0, self.t0 + dt, self.t0 + 2*dt
            l0 = (t-tau1)*(t-tau2)/(2*dt**2)
            l1 = (t-tau0)*(t-tau2)/(-dt**2)
            l2 = (t-tau0)*(t-tau1)/(2*dt**2)

            v0 = self.Vs[0].eval(xyz)
            v1 = self.Vs[1].eval(xyz)
            v2 = self.Vs[2].eval(xyz)

            return l0*v0 + l1*v1 + l2*v2


        if len(self.Vs) == 1:
            v_out = self.Vs[0].eval(xyz)

            return v_out

        if len(self.Vs) == 2:
            tau = (t - self.t0)/dt
            v0 = self.Vs[0].eval(xyz)
            v1 = self.Vs[1].eval(xyz)

            return (1-tau)*v0 + tau*v1


# ------------------------------------------------------------------------------

#-----------Linear advection test velocity fields-------------------------------
pi = np.pi
T = 1
alpha = 0 # default parameter for the solid body rotation

def u_sbr(t,dt,X):
    # solid body rotation
    c = -2*pi/T # full rotation for every t = 1 unit of time.

    return np.array(utils.cross(X, [c*np.sin(alpha), 0, c*np.cos(alpha)]))

def u_deform(t,dt,X):
    # deformational flow test
    k = 2
    c1 = 2*k*np.cos(pi*t/T)
    psi_x = 0*X[0]; psi_y = c1*X[1]; psi_z = 0*X[2];

    return np.array(utils.cross(X, [psi_x, psi_y, psi_z]))

def u_deform_rot(t,dt,X):
    # transform the query points into the rotating frame
    tau = 2*pi*t
    X_r = utils.rotate(utils.Mmul(utils.Rot_z(tau).T, utils.Rot_y(alpha)), X)
    u_s = u_deform(t,dt,X_r)

    #now rotate back into static frame and add with sbr
    u_sr = utils.rotate(utils.Mmul(utils.Rot_y(alpha).T, utils.Rot_z(tau)), u_s)
    u_rot = u_sbr(t,dt,X)

    return u_sr +  u_rot

rho_0 = 3. # constant defining vortex
def sech(x):
    return 1/np.cosh(x)

def omega_r(theta):

    c_the = np.sin(theta)
    outs = 0*theta
    outs[np.where(theta != 0.)] = (2*pi)*np.sqrt(3)*(3/2)*(1/(rho_0*c_the[np.where(theta != 0)]))\
    *np.tanh(rho_0*c_the[np.where(theta != 0)])*(sech(rho_0*c_the[np.where(theta!=0)])**2)

    return outs

def u_static_vortex(t, dt, X):
    # static vortex test case
    phi, theta = utils.cart2sphere(X)
    c_the = np.sin(theta)
    omg = 0*theta
    omg[np.where(theta != 0.)] = (2*pi)*np.sqrt(3)*(3/2)*(1/(rho_0*c_the[np.where(theta != 0)]))\
    *np.tanh(rho_0*c_the[np.where(theta != 0)])*(sech(rho_0*c_the[np.where(theta!=0)])**2)

    return np.array([-X[1]*omg, X[0]*omg, 0*X[2]])

def u_rotating_vortex(t, dt, X):
    #in rotated system
    tau = 2*pi*t/T
    #transform into rotating system
    X_r = utils.rotate(utils.Mmul(utils.Rot_x(pi/2), utils.Rot_z(tau).T), X)
    u_prime = u_static_vortex(t,dt, X_r)
    u_sr =  utils.rotate(utils.Mmul(utils.Rot_z(tau), utils.Rot_x(pi/2).T), u_prime)
    u_rot = u_sbr(t,dt,X)

    return u_sr + u_rot

def equator_vortex(t, dt, X):
    #transform into rotating system
    X_r = utils.rotate(utils.Rot_x(pi/2), X)
    u_prime = u_static_vortex(t,dt, X_r)
    u_sr =  utils.rotate(utils.Rot_x(pi/2).T, u_prime)
    return np.array(u_sr)

def static_vort_IC(t,xyz):
    phi, theta = utils.cart2sphere(xyz)
    rho = rho_0*np.sin(theta)
    return 1-np.tanh((rho/5)*np.sin(phi-omega_r(theta)*t))

def rotating_vort_IC(t,xyz):
    tau = 0 #2*pi/T
    X_r = utils.rotate(utils.Mmul(utils.Rot_x(pi/2),utils.Rot_z(tau).T), xyz)
    phi, theta = utils.cart2sphere(X_r)
    rho = rho_0*np.sin(theta)
    return 1-np.tanh((rho/5)*np.sin(phi-omega_r(theta)*t))

def u_div(t,dt, X):
    [phi, theta] = utils.cart2sphere(X)
    u = -(np.sin(phi/2)**2)*np.sin(2*theta)*np.sin(theta)*np.cos(pi*t/T)
    v = (-1/2)*np.sin(phi)*(np.sin(theta)**3)*np.cos(pi*t/T)

    u_x = np.cos(theta)*np.cos(phi)*v - np.sin(theta)*np.sin(phi)*u
    u_y = np.cos(theta)*np.sin(phi)*v + np.sin(theta)*np.cos(phi)*u
    u_z = -np.sin(theta)*v

    return np.array(utils.tan_proj(X,[u_x, u_y, u_z]))

# Initial tracer distributions ------------------------------------------------

h_0 = 1.
# for deformational flows
phi_c1, the_c1 = pi + pi/6, pi/2
phi_c2, the_c2 = pi - pi/6, pi/2

# #for divergent flow
# phi_c1, the_c1 = 5*pi/4, pi/2
# phi_c2, the_c2 = 3*pi/4, pi/2

R = 1/2

# ##cosine bell initial condition
def cosine_bell(phi,theta):
    #cosine bell on sphere
    arg1 = np.sin(the_c1-pi/2)*np.sin(theta-pi/2) + np.cos(the_c1-pi/2)*np.cos(theta-pi/2)*np.cos(phi-phi_c1)
    arg2 = np.sin(the_c2-pi/2)*np.sin(theta-pi/2) + np.cos(the_c2-pi/2)*np.cos(theta-pi/2)*np.cos(phi-phi_c2)

    r_p = np.arccos(arg1)
    inds1 = np.where(r_p < R)
    out = 0*theta
    out[inds1] = (h_0/2)*(1 + np.cos(r_p[inds1]*pi/R))

    r_p2 = np.arccos(arg2)
    inds2 = np.where(r_p2 < R)
    out2 = 0*theta
    out2[inds2] = (h_0/2)*(1 + np.cos(r_p2[inds2]*pi/R))

    return 0.9*(out + out2) + 0.1

def IC_SC(phi,theta):
    #Slotted Cylinder initial condition
    arg1 = np.sin(the_c1-pi/2)*np.sin(theta-pi/2) + np.cos(the_c1-pi/2)*np.cos(theta-pi/2)*np.cos(phi-phi_c1)
    arg2 = np.sin(the_c2-pi/2)*np.sin(theta-pi/2) + np.cos(the_c2-pi/2)*np.cos(theta-pi/2)*np.cos(phi-phi_c2)

    r_p = np.arccos(arg1)
    # inds1_0 = np.where(r_p <= R & np.absolute(phi - phi_c1) >= R/6)
    # inds1 = np.where(r_p <= R & np.absolute(phi - phi_c1) < R/6 & (theta - theta_c1 - pi/2) < -5*R/12)
    out = 0*theta
    #phi, theta = Mod(phi, theta)
    #phi_mod = (phi + pi) % 2*pi - pi
    # phi_c1, phi_c2 = -pi/6, pi/6

    out[(r_p <= R) & (np.absolute(phi - phi_c1) >= R/6)] = 0.9
    out[(r_p <= R) & (np.absolute(phi - phi_c1) < R/6) & (theta < (-5*R/12+pi/2))] = 0.9

    r_p2 = np.arccos(arg2)

    out2 = 0*theta
    out2[(r_p2 <= R) & (np.absolute(phi - phi_c2) >= R/6) ] = 0.9
    out2[(r_p2 <= R) & (np.absolute(phi - phi_c2) < R/6) & (theta > (5*R/12+pi/2))] = 0.9

    return out + out2 + 0.1


np.random.seed(42)
coeffs = np.random.uniform(-1,1, 33**2)
def random_initial_condition(xyz):
    [phi,theta] = utils.cart2sphere(xyz)
    ells = np.arange(0,33)
    out = 0
    lm = 0
    for l in ells:
        for m in range(-l,l):
            out += coeffs[lm]*sph_harm(m, l, phi, theta).real
            lm += 1

    return out

# ==============================================================================

# Initial vorticity distributions
def rotating_frame(phi, theta):
    # this is the 2*planetary vorticity with t = 1 being a full rotation
    return 4*pi*np.cos(theta)


def zonal_jet(phi, theta):
    #unperturbed Zonal Jet
    beta2 = 12**2
    theta_c = pi/4
    u_lt = (pi/2)*np.exp(-2*beta2*(1-np.cos(-theta+pi/2-theta_c)))
    return (np.cos(-theta+pi/2)*(2*beta2*(np.cos(theta_c)*np.sin(-theta+pi/2) - np.sin(theta_c)*np.cos(-theta+pi/2))) + np.sin(pi/2 - theta))*u_lt

pert = 0.01
def perturbed_zonal_jet(phi, theta):
    beta2 = 12**2
    theta_c = pi/4 + 0.01*np.cos(pert*phi)
    u_lt = (pi/2)*np.exp(-2*beta2*(1-np.cos(-theta+pi/2-theta_c)))
    return (np.cos(-theta+pi/2)*(2*beta2*(np.cos(theta_c)*np.sin(-theta+pi/2) - np.sin(theta_c)*np.cos(-theta+pi/2))) + np.sin(pi/2 - theta))*u_lt

def gaussian_vortex(phi, theta):
    [x,y,z] = utils.sphere2cart(phi,theta)
    sigma, C = 1/16, 4*pi
    return C*np.exp(-((x-1)**2 + y**2 + z**2)/sigma)

def rossby_wave(phi, theta, t = 0):
    # Rossby Haurwitz wave (\ell,m) = (5,4)
    ell = 5
    # C = 1 #(ell*(ell + 1))/((ell*(ell + 1)) -2)
    alpha = 1/(ell*(ell+1))
    Omega = 2*pi
    return 30*np.cos(theta)*np.cos(4*(phi + 2*Omega*alpha*t))*np.sin(theta)**4 #+ C*4*pi*cos(theta)

# perturbed Rossby-Haurwitz wave
def perturbed_rossby_wave(phi, theta):
    return 30*np.cos(theta)*np.cos(4*phi)*np.sin(theta)**4 + 0.1*np.sin(theta)*np.cos(30*theta)

