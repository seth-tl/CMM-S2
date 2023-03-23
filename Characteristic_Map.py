import numpy as np
import pdb, pickle
from MappingS2 import MapS2
from InterpS2 import Hermite_S2

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

    if eps_c == False:
        f = (1/4)*(H1 + H2 + H3 + H4)
        f_p = (1/(4*eps))*((H3 - H1) + (H4 - H2))
        f_t = (1/(4*eps))*((H2 - H1) + (H4 - H3))
        f_pt = (1/(4*(eps**2)))*(H4 - H3 - H2 + H1)
    else:
        f = (1/4)*(H1[0] + H2[0] + H3[0] + H4[0])
        f_p = (1/(4*eps))*((H3[0] - H1[0]) + (H4[0] - H2[0]))
        f_t = (1/(4*eps))*((H2[0] - H1[0]) + (H4[0] - H3[0]))
        f_pt = (1/(4*(eps_c**2)))*(H4[1] - H3[1] - H2[1] + H1[1])


    return [f, f_p, f_t, f_pt]

def Hermite_proj_eps(interpolant, eps1, eps2, eps3, eps4, yn):

    H1 = interpolant.eval(phi = yn[0], theta = yn[1],
                          phi_s = eps1[0], theta_s = eps1[1])
    H2 = interpolant.eval(phi = yn[0], theta = yn[1],
                          phi_s = eps2[0], theta_s = eps2[1])
    H3 = interpolant.eval(phi = yn[0], theta = yn[1],
                          phi_s = eps3[0], theta_s = eps3[1])
    H4 = interpolant.eval(phi = yn[0], theta = yn[1],
                          phi_s = eps4[0], theta_s = eps4[1])

    return np.array([H1, H2, H3, H4])


def Displacement(H_phi, H_the, V, t, dt, y0, eps = 1e-5, eps_c = False, deriv = False):

    # Compute displacement map for each coordinate
    phi_grid = H_phi.phi
    the_grid = H_phi.theta

    eps1 = OneStepSO(V, [y0[0] - eps, y0[1] - eps], t, dt)
    eps2 = OneStepSO(V, [y0[0] - eps, y0[1] + eps], t, dt)
    eps3 = OneStepSO(V, [y0[0] + eps, y0[1] - eps], t, dt)
    eps4 = OneStepSO(V, [y0[0] + eps, y0[1] + eps], t, dt)
    yn = OneStepSO(V, y0, t, dt)



    H_phi_n0 = Hermite_proj_eps(H_phi, eps1, eps2, eps3, eps4, yn)
    H_the_n0 = Hermite_proj_eps(H_the, eps1, eps2, eps3, eps4, yn)

    # Displacement

    if eps_c != False:
        eps1_c = OneStepSO(V, [y0[0] - eps_c, y0[1] - eps_c], t, dt)
        eps2_c = OneStepSO(V, [y0[0] - eps_c, y0[1] + eps_c], t, dt)
        eps3_c = OneStepSO(V, [y0[0] + eps_c, y0[1] - eps_c], t, dt)
        eps4_c = OneStepSO(V, [y0[0] + eps_c, y0[1] + eps_c], t, dt)
        yn = OneStepSO(V, y0, t, dt)

        H_phi_n0_c = Hermite_proj_eps(H_phi, eps1_c, eps2_c, eps3_c, eps4_c, yn)
        H_the_n0_c = Hermite_proj_eps(H_the, eps1_c, eps2_c, eps3_c, eps4_c, yn)

        # Displacement
        dX1_c = eps1_c - np.array([y0[0] - eps_c, y0[1] - eps_c])
        dX2_c = eps2_c - np.array([y0[0] - eps_c, y0[1] + eps_c])
        dX3_c = eps3_c - np.array([y0[0] + eps_c, y0[1] - eps_c])
        dX4_c = eps4_c - np.array([y0[0] + eps_c, y0[1] + eps_c])

        H_phi_n_c = H_phi_n0_c + np.array([dX1_c[0], dX2_c[0], dX3_c[0], dX4_c[0]])
        H_the_n_c = H_the_n0_c + np.array([dX1_c[1], dX2_c[1], dX3_c[1], dX4_c[1]])

        F_phi = epsilon_diff([H_phi_n_c[0], H_phi_n_c[0]], [H_phi_n_c[1], H_phi_n_c[1]], \
                              [H_phi_n_c[2], H_phi_n_c[2]], [H_phi_n_c[3], H_phi_n_c[3]], eps, eps_c)
        F_the = epsilon_diff([H_the_n_c[0], H_phi_n_c[0]], [H_the_n_c[1], H_phi_n_c[1]], \
                              [H_the_n_c[2], H_phi_n_c[2]], [H_the_n_c[3], H_phi_n_c[3]], eps, eps_c)

        H_phi_new = Hermite_S2(phi_grid, the_grid, F_phi[0], F_phi[1], F_phi[2], F_phi[3])
        H_the_new = Hermite_S2(phi_grid, the_grid, F_the[0], F_the[1], F_the[2], F_the[3])

    else:
        dX1 = eps1 - np.array([y0[0] - eps, y0[1] - eps])
        dX2 = eps2 - np.array([y0[0] - eps, y0[1] + eps])
        dX3 = eps3 - np.array([y0[0] + eps, y0[1] - eps])
        dX4 = eps4 - np.array([y0[0] + eps, y0[1] + eps])

        H_phi_n = H_phi_n0 + np.array([dX1[0], dX2[0], dX3[0], dX4[0]])
        H_the_n = H_the_n0 + np.array([dX1[1], dX2[1], dX3[1], dX4[1]])

        F_phi = epsilon_diff(H_phi_n[0], H_phi_n[1], H_phi_n[2], H_phi_n[3], eps)
        F_the = epsilon_diff(H_the_n[0], H_the_n[1], H_the_n[2], H_the_n[3], eps)

        H_phi_new = Hermite_S2(phi_grid, the_grid, F_phi[0], F_phi[1], F_phi[2], F_phi[3])
        H_the_new = Hermite_S2(phi_grid, the_grid, F_the[0], F_the[1], F_the[2], F_the[3])



    if deriv == False:
        return H_phi_new, H_the_new

    else:
        return H_phi_new, H_the_new, F_phi, F_the



    # if eps_c != False:
    #     eps1_c = OneStepSO(V, [y0[0] - eps_c, y0[1] - eps_c], t, dt)
    #     eps2_c = OneStepSO(V, [y0[0] - eps_c, y0[1] + eps_c], t, dt)
    #     eps3_c = OneStepSO(V, [y0[0] + eps_c, y0[1] - eps_c], t, dt)
    #     eps4_c = OneStepSO(V, [y0[0] + eps_c, y0[1] + eps_c], t, dt)
    #     yn = OneStepSO(V, y0, t, dt)
    #
    #     H_phi_n0_c = Hermite_proj_eps(H_phi, eps1_c, eps2_c, eps3_c, eps4_c, yn)
    #     H_the_n0_c = Hermite_proj_eps(H_the, eps1_c, eps2_c, eps3_c, eps4_c, yn)
    #
    #     # Displacement
    #     dX1_c = eps1_c - np.array([y0[0] - eps_c, y0[1] - eps_c])
    #     dX2_c = eps2_c - np.array([y0[0] - eps_c, y0[1] + eps_c])
    #     dX3_c = eps3_c - np.array([y0[0] + eps_c, y0[1] - eps_c])
    #     dX4_c = eps4_c - np.array([y0[0] + eps_c, y0[1] + eps_c])
    #
    #     H_phi_n_c = H_phi_n0_c + np.array([dX1_c[0], dX2_c[0], dX3_c[0], dX4_c[0]])
    #     H_the_n_c = H_the_n0_c + np.array([dX1_c[1], dX2_c[1], dX3_c[1], dX4_c[1]])
    #
    #
    #     F_phi = epsilon_diff([H_phi_n[0], H_phi_n_c[0]], [H_phi_n[1], H_phi_n_c[1]], \
    #                           [H_phi_n[2], H_phi_n_c[2]], [H_phi_n[3], H_phi_n_c[3]], eps, eps_c)
    #     F_the = epsilon_diff([H_the_n[0], H_phi_n_c[0]], [H_the_n[1], H_phi_n_c[1]], \
    #                           [H_the_n[2], H_phi_n_c[2]], [H_the_n[3], H_phi_n_c[3]], eps, eps_c)
    #
    #     H_phi_new = Hermite_S2(phi_grid, the_grid, F_phi[0], F_phi[1], F_phi[2], F_phi[3])
    #     H_the_new = Hermite_S2(phi_grid, the_grid, F_the[0], F_the[1], F_the[2], F_the[3])
    #
    # else:
#
# f_p = H3 - H1 + H4 - H2
# f_p_sign = np.sign(f_p)
# f_p_abs = np.absolute(f_p)/4
# f_p[np.where(f_p != 0.)] = f_p_sign[np.where(f_p != 0.)]*np.exp(np.log(f_p_abs[np.where(f_p != 0.)]) - np.log(eps))
#
# f_t = H2 - H1 + H4 - H3
# f_t_sign = np.sign(f_t)
# f_t_abs = np.absolute(f_t)/4
# f_t[np.where(f_t != 0.)] = f_t_sign[np.where(f_t != 0.)]*np.exp(np.log(f_t_abs[np.where(f_t != 0.)]) - np.log(eps))
#
# f_pt = H4 - H3 - H2 + H1
# f_pt_sign = np.sign(f_pt)
# f_pt_abs = np.absolute(f_pt)/4
# f_pt[np.where(f_pt != 0.)] = f_pt_sign[np.where(f_pt != 0.)]*np.exp(np.log(f_pt_abs[np.where(f_pt != 0.)]) - np.log(eps**2))




# # Transformation functions -------
# def cart2sphere(XYZ):
#     """
#     input cartesian coordinates, output spherical coordinates (phi, theta)
#     """
#     x,y,z = XYZ[0], XYZ[1], XYZ[2]
#     phi = np.arctan2(y,x)
#     theta = np.arctan2(np.sqrt(x**2 + y**2), z)
#
#     return [phi, theta]
#
# def sphere2cart(phi, theta):
#     x = sin(theta)*cos(phi)
#     y = sin(theta)*sin(phi)
#     z = cos(theta)
#
#     return np.array([x,y,z])
#
# s2c = MapS2(func = sphere2cart)
#
# def azim_cross(x_n, x_n1):
#     """
#     Input: two arrays Cartesian coordinates and determines whether or not the lines
#     phi = -pi or pi have been crossed during the trajectory connecting
#     x_n --> x_n1 on the surface of the sphere.
#
#     Output: two arrays of indices indicating which coordinates crossed
#     pi for the first array and -pi for the second array
#     """
#     y_n = np.sign(x_n[1])
#     y_n1 = np.sign(x_n1[1])
#
#     ys = y_n1 - y_n
#
#     return ys
#
# def Cart_Dynams(U_phi, U_the, XYZ):
#     """
#     Input time derivative of phi and theta,
#     output time derivative of x,y,z
#     """
#     # out = np.empty([len(XYZ[:,0]), XYZ[0,:],3])
#
#     inds = np.where(np.sqrt(XYZ[0]**2 + XYZ[1]**2) <= 1e-10)
#
#     x_dot = U_the*XYZ[0]*XYZ[2]/(np.sqrt(XYZ[0]**2 + XYZ[1]**2)) \
#             - U_phi*XYZ[1]/(np.sqrt(XYZ[0]**2 + XYZ[1]**2))
#     y_dot =  U_the*XYZ[1]*XYZ[2]/(np.sqrt(XYZ[0]**2 + XYZ[1]**2)) \
#          + U_phi*XYZ[0]/(np.sqrt(XYZ[0]**2 + XYZ[1]**2))
#     z_dot = -np.sqrt(XYZ[0]**2 + XYZ[1]**2)*U_the
#
#     x_dot[inds] = -U_phi[inds]*XYZ[1][inds]
#     y_dot[inds] = U_phi[inds]*XYZ[0][inds]
#
#     return np.array([x_dot, y_dot, z_dot])
#
# def Cart_Stencil(y0, eps = 1e-5):
#
#     yn = y0.copy()
#     neg_the = np.where(yn[1] <= 0.)
#
#     xyz = s2c.eval(yn[0], yn[1])
#
#     #Calculate Stencil Points:
#     r = np.sqrt(xyz[0]**2 + xyz[1]**2)
#
#     xyz_1 = np.array([xyz[0]*cos(eps)**2 + sin(eps)*cos(eps)*(xyz[1] - xyz[0]*xyz[2]/r) \
#                      - (xyz[1]*xyz[2]/r)*sin(eps)**2,
#             xyz[1]*cos(eps)**2 + sin(eps)*cos(eps)*(-xyz[0] - xyz[1]*xyz[2]/r) \
#                              + (xyz[0]*xyz[2]/r)*sin(eps)**2,
#             xyz[2]*cos(eps) + r*sin(eps)])
#
#     xyz_2 = np.array([xyz[0]*cos(eps)**2 + sin(eps)*cos(eps)*(xyz[1] + xyz[0]*xyz[2]/r) \
#                      - (xyz[1]*xyz[2]/r)*sin(eps)**2,
#             xyz[1]*cos(eps)**2 + sin(eps)*cos(eps)*(-xyz[0] + xyz[1]*xyz[2]/r) \
#                              - (xyz[0]*xyz[2]/r)*sin(eps)**2,
#             xyz[2]*cos(eps) - r*sin(eps)])
#
#     xyz_3 = np.array([xyz[0]*cos(eps)**2 + sin(eps)*cos(eps)*(-xyz[1] - xyz[0]*xyz[2]/r) \
#                      + (xyz[1]*xyz[2]/r)*sin(eps)**2,
#             xyz[1]*cos(eps)**2 + sin(eps)*cos(eps)*(xyz[0] - xyz[1]*xyz[2]/r) \
#                              - (xyz[0]*xyz[2]/r)*sin(eps)**2,
#             xyz[2]*cos(eps) + r*sin(eps)])
#
#     xyz_4 = np.array([xyz[0]*cos(eps)**2 + sin(eps)*cos(eps)*(xyz[1] - xyz[0]*xyz[2]/r) \
#                      - (xyz[1]*xyz[2]/r)*sin(eps)**2,
#             xyz[1]*cos(eps)**2 + sin(eps)*cos(eps)*(-xyz[0] - xyz[1]*xyz[2]/r) \
#                              + (xyz[0]*xyz[2]/r)*sin(eps)**2,
#             xyz[2]*cos(eps) + r*sin(eps)])
#
#     # inds = np.where(r <= 1e-10)
#     # xyz_1[inds] = xyz[0][inds]*cos(eps)**2  + sin(eps)*cos(eps)*xyz[1][inds]
#     # xyz_2[inds] = xyz[1][inds]*cos(eps)**2 + sin(eps)*cos(eps)*xyz[0][inds]
#     # xyz_3[inds] =
#
#     return [xyz_1, xyz_2, xyz_3, xyz_4]

#
# def OneStepSO(V, y0, t, dt, stencil = []):
#
#     yn = y0.copy()  # Don't destroy initial conditions
#     # Convert to Cartesian coordinates
#     neg_the = np.where(yn[1] <= 0.)
#
#     if len(stencil) == 0:
#         XYZ = s2c.eval(yn[0], yn[1])
#     else:
#         XYZ = stencil
#
#     #backwards RK3-TVD
#     tpdt = t + dt
#
#     # Step 1
#     V1 = V(tpdt, cart2sphere(XYZ))
#     V1_Cart = Cart_Dynams(V1[0], V1[1], XYZ)
#     step1 = XYZ - dt*V1_Cart
#     step1_ang = cart2sphere(step1)
#
#     #Step 2
#     V2 = V(t, step1_ang)
#     V2_Cart = Cart_Dynams(V2[0], V2[1], step1)
#
#     step2 = XYZ - dt*((1/4)*V1_Cart + (1/4)*V2_Cart)
#     step2_ang = cart2sphere(step2)
#
#     #Step3
#     V3 = V(t + 0.5*dt, step2_ang)
#     V3_Cart = Cart_Dynams(V3[0], V3[1], step2)
#
#     xyz_n1 = XYZ - dt*((1/6)*V1_Cart + (1/6)*V2_Cart + (2/3)*V3_Cart)
#
#     mods = azim_cross(XYZ, xyz_n1)
#     yn1 = cart2sphere(xyz_n1)
#
#     x_inds = np.where(xyz_n1[0] <= 0.)
#
#     # yn1[0][x_inds] = yn1[0][x_inds] - pi*mods[x_inds]
#
#     return yn1
