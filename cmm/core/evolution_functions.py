#/----
"""
This script provides all functionality related to time stepping and contains
most core components of the CMM for advection.
"""
#/----
# imports
import numpy as np
import pdb
from . import mesh_functions as meshes
from .interpolants.spherical_spline import sphere_diffeomorphism
from .interpolants.torus_interpolants import Hermite_Map, Hermite_T2
from .interpolants.channel_interpolant import Hermite_channel, channel_diffeomorphism
from . import utils

import threading

#------------------ numerical integration schemes on sphere --------------------
# Some useful integrators
# dt is included in the argument to be compatible with the temporal interpolated
# velocity fields used for the non-linear advection

def Euler_step_proj(t,dt,xn,V):
    #backwards Euler step
    v1 = V(t, dt, xn)
    #project back onto sphere
    return div_norm(xn-dt*v1)

def improved_Euler_proj(t,dt,xn,V):
    #backwards in time improved Euler scheme
    tpdt = t + dt
    # step 1:
    v1 = V(t,dt,xn)
    step1 = div_norm(xn-dt*v1)
    #step 2:
    v2 = V(t+dt, dt, step1)

    return div_norm(xn-(dt/2)*(v1 + v2))

def RK3_proj(t,dt,xn,V):
    #backwards RK3 combined with projection step
    tpdt = t + dt
    v1 = V(tpdt, dt, xn)

    step1 = div_norm(xn-dt*v1, xn, v1)
    v2 = V(t, dt, v1)

    step2 = div_norm(xn - (dt/4)*(v1+v2))
    v3 = V(t+0.5*dt, dt, step2)

    xn1 = xn - dt*((1/6)*v1 + (1/6)*v2 + (2/3)*v3)
    return div_norm(xn1)

def RK4_proj(t,dt,xn,V):
    #backwards RK4 combined with projection step
    tpdt = t + dt

    v1 = V(tpdt, dt, xn)
    step1 = div_norm(xn-dt*v1/2)

    v2 = V(t+dt/2, dt, step1)
    step2 = div_norm(xn - (dt/2)*v2)

    v3 = V(t+0.5*dt, dt, step2)

    step3 = div_norm(xn - dt*v3)
    v4 = V(t, dt, step3)

    xn1 = xn - (dt/6)*(v1 + 2*v2 + 2*v3 + v4)

    return div_norm(xn1)


def RK2_proj(t, dt, xn, V):
    #backwards RK2 combined with projection step
    tpdt = t + dt

    v1 = V(tpdt, dt, x_n)
    step1 = div_norm(xn-(dt/2)*v1)

    v2 = V(t+dt/2, dt, step1)
    xn1 = xn - dt*v2

    return div_norm(xn1)

def div_norm(x):
    # projection onto sphere for integrators, accepts more general arrays
    normalize = 1/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return normalize[None,:]*x

# non-spherical integrators -------------------------

def RK3_SSP(t,dt,xn,V):
    #backwards RK3 combined with projection step
    x_n = np.array(xn.copy())  # Don't destroy initial conditions
    tpdt = t + dt

    v1 = V(tpdt, dt, x_n)
    step1 =  xn - dt*v1
    v2 = V(t, dt, step1)

    step2 = xn - (dt/4)*(v1 + v2)
    v3 = V(t+0.5*dt, dt, step2)

    xn1 = xn - dt*((1/6)*v1 + (1/6)*v2 + (2/3)*v3)

    return xn1

def Euler_step(t,dt,xn,V):
    #backwards Euler step
    v1 = V(t, dt, xn)
    return xn-dt*v1

def improved_Euler(t,dt,xn,V):
    #backwards in time improved Euler scheme
    tpdt = t + dt
    # step 1:
    v1 = V(t,dt,xn)
    step1 = xn-dt*v1
    #step 2:
    v2 = V(t+dt, dt, step1)

    return xn-(dt/2)*(v1 + v2)

# ===== map related functions for evolution  ===================================

def epsilon_diff4(evals, interpolant, eps):
    # form data for the integrated foot points
    # average stencilling
    vals = (evals[:,0,:] + evals[:,1,:] + evals[:,2,:] + evals[:,3,:])/4

    #partial derivative in pre-computed orthonormal basis
    df_dx1 = (evals[:,1,:] - evals[:,0,:] + evals[:,3,:] - evals[:,2,:])/(4*eps)
    df_dx2 = (evals[:,2,:] - evals[:,0,:] + evals[:,3,:] - evals[:,1,:])/(4*eps)

    # re-express in Cartesian coordinates and arrange appropriately
    gammas = interpolant.mesh.tan_vects

    # TODO: get rid of this clunky rearrangement
    grad_vals = [(df_dx1[0][:,None]*gammas[0] + df_dx2[0][:,None]*gammas[1]).T,
                 (df_dx1[1][:,None]*gammas[0] + df_dx2[1][:,None]*gammas[1]).T,
                 (df_dx1[2][:,None]*gammas[0] + df_dx2[2][:,None]*gammas[1]).T]

    return vals, grad_vals

def spline_proj_sphere(interpolant, s_pts, eps= 1e-5):
    # Projection step for each map, updates values and gradient_values of interpolant
    vals, grad_vals = epsilon_diff4(s_pts, interpolant, eps)
    #normalize
    vals = utils.div_norm(vals.T)
    return sphere_diffeomorphism(mesh = interpolant.mesh, vals = vals, grad_vals = grad_vals)



def advect(interp, integrator, t, dt, V, identity = False):
        # define points to integrate
        spts = np.array(interp.mesh.s_pts)
        verts0 = interp.mesh.vertices.T

        # perform integration
        yn = integrator(t, dt, verts0, V)
        s = np.shape(spts)
        spts_n = integrator(t,dt, spts.reshape([3,s[1]*s[2]]),V)

        if identity == False:
            #perform evaluation of stencil points at previous map
            s_evals = interp.stencil_eval(q_pts = yn, st_pts = spts_n.reshape(s))
            return s_evals

        else:
            return spts_n.reshape(np.shape(spts))
            # # if identity map then simply return value of footpoint
            # outsx = [outs0[0,:], outs1[0,:], outs2[0,:], outs3[0,:]]
            # outsy = [outs0[1,:], outs1[1,:], outs2[1,:], outs3[1,:]]
            # outsz = [outs0[2,:], outs1[2,:], outs2[2,:], outs3[2,:]]
            #
            # return [outsx, outsy, outsz]

def hermite_stencil_eval(interpolant, eps1, eps2, eps3, eps4, yn):

    H1 = interpolant.stencil_eval(phi = yn[0], theta = yn[1],
                        phi_s = eps1[0], theta_s = eps1[1])
    H2 = interpolant.stencil_eval(phi = yn[0], theta = yn[1],
                        phi_s = eps2[0], theta_s = eps2[1])
    H3 = interpolant.stencil_eval(phi = yn[0], theta = yn[1],
                        phi_s = eps3[0], theta_s = eps3[1])
    H4 = interpolant.stencil_eval(phi = yn[0], theta = yn[1],
                        phi_s = eps4[0], theta_s = eps4[1])
    
    return [H1, H2, H3, H4]

def epsilon_diff_torus(H1, H2, H3, H4, eps = 1e-5):

    """
    H1-H4 can be lists of stencil points. If it is a list then the stencil
    pts correspond to the stencil for the gradient and mixed derivatives
    separately.
    """

    f = (1/4)*(H1 + H2 + H3 + H4)
    f_p = (1/(4*eps))*((H3 - H1) + (H4 - H2))
    f_t = (1/(4*eps))*((H2 - H1) + (H4 - H3))
    f_pt = (1/(4*(eps**2)))*(H4 - H3 - H2 + H1)

    return [f, f_p, f_t, f_pt]

def advect_project_torus(interp, integrator, t, dt, V, identity = False):
        # define points to integrate
        spts = np.array(interp.mesh.s_pts)
        y0 = interp.mesh.vertices

        #integrate for foot points, one step map
        eps1 = integrator(t, dt, spts[0], V)
        eps2 = integrator(t, dt, spts[1], V)
        eps3 = integrator(t, dt, spts[2], V)
        eps4 = integrator(t, dt, spts[3], V)
        yn = integrator(t, dt, y0, V)

        # only evaluated the displacement component
        Chi_x_n1 = hermite_stencil_eval(interp.Chi_x, eps1, eps2, eps3, eps4, yn)
        Chi_y_n1 = hermite_stencil_eval(interp.Chi_y, eps1, eps2, eps3, eps4, yn)

        # Displacement
        dX1 = eps1 - spts[0]; dX2 = eps2 - spts[1]; dX3 = eps3 - spts[2]; dX4 = eps4 - spts[3]
        # pdb.set_trace()
        # dX1 = -spts[0]; dX2 = -spts[1]; dX3 = -spts[2]; dX4 = -spts[3]
        H_phi_n = Chi_x_n1 + np.array([dX1[0], dX2[0], dX3[0], dX4[0]])
        H_the_n = Chi_y_n1 + np.array([dX1[1], dX2[1], dX3[1], dX4[1]])

        F_phi = epsilon_diff_torus(H_phi_n[0], H_phi_n[1], H_phi_n[2], H_phi_n[3], eps = 1e-5)
        F_the = epsilon_diff_torus(H_the_n[0], H_the_n[1], H_the_n[2], H_the_n[3], eps = 1e-5)

        H_phi_new = Hermite_T2(interp.Chi_x.mesh, F_phi[0], F_phi[1], F_phi[2], F_phi[3])
        H_the_new = Hermite_T2(interp.Chi_y.mesh, F_the[0], F_the[1], F_the[2], F_the[3])

        return Hermite_Map(interp.mesh, H_phi_new, H_the_new, identity = False)


def advect_channel(interp, integrator, t, dt, V, identity = False):
        #TODO: incorporate this functionaloty with the advect function
        """
        Specialized function
        perform one "Advect" step
        Inputs: interp (object) - spline_interp_vec class
                integrator (callable) desired integration routine
                t (float)
                dt (float)
        """

        spts = interp.mesh.s_pts
        # ----------------------------------------------------------------------
        # 4 - points implementation -- with extrapolation
        yn = integrator(t, dt, interp.mesh.pts_list, V)
        # pdb.set_trace()

        eps_int = [integrator(t,dt, spts[0],V),
                   integrator(t,dt, spts[1],V),
                   integrator(t,dt, spts[2],V),
                   integrator(t,dt, spts[3],V)]

        #perform evaluation for interior points
        s_evals = interp.stencil_eval(q_pts = yn, spts = eps_int)

        # # make modifications for the periodic boundary
        for i in range(4):
            s_evals[0][i] = eps_int[i][0] - spts[i][0] + s_evals[0][i]

        return s_evals


def epsilon_diff4_cubic(evals, eps):
    f = (evals[0] + evals[1] + evals[2] + evals[3])/4
    d_eps = 4*eps
    df_dx1 = (evals[2] - evals[0] + evals[3] - evals[1])/(d_eps)
    df_dx2 = (evals[1] - evals[0] + evals[3] - evals[2])/(d_eps)
    df_dx1dx2 = (1/(4*(eps**2)))*(evals[3] - evals[2] - evals[1] + evals[0])

    return [f, [df_dx1, df_dx2], df_dx1dx2]

def epsilon_diff_boundary(evals, eps):

    f = (evals[0] + evals[1])/2
    #directional derivative along boundary
    grad_f = (evals[0]-evals[1])/(2*eps)

    return [f, grad_f]

def spline_proj_channel(interpolant, s_pts, eps = 1e-5):
    #TODO: clean this code up
    
    # modify for periodic boundary
    x_n  = epsilon_diff4_cubic(s_pts[0], eps)
    y_n =  epsilon_diff4_cubic(s_pts[1], eps)

    xy_n = np.array([x_n[0], y_n[0]])

    vals_n = [interpolant.Chi_x.f.copy(), interpolant.Chi_y.f.copy()]

    grad_vals_xn = [interpolant.Chi_x.f_x.copy(), interpolant.Chi_x.f_y.copy()]
    grad_vals_yn = [interpolant.Chi_y.f_x.copy(), interpolant.Chi_y.f_y.copy()]

    cross_x_n = interpolant.Chi_x.f_xy.copy()
    cross_y_n = interpolant.Chi_y.f_xy.copy()

    # update values
    ss = np.shape(vals_n[0])
    vals_n[0] = xy_n[0,:].reshape(ss)
    vals_n[1] = xy_n[1,:].reshape(ss)

    #update gradient data
    grad_vals_xn[0] = x_n[1][0].reshape(ss)
    grad_vals_xn[1] = x_n[1][1].reshape(ss)

    grad_vals_yn[0] = y_n[1][0].reshape(ss)
    grad_vals_yn[1] = y_n[1][1].reshape(ss)

    #update the cross derivatives
    cross_x_n = x_n[2].reshape(ss)
    cross_y_n = y_n[2].reshape(ss)

    L_x = interpolant.Chi_x.L_x; L_y = interpolant.Chi_x.L_y;

    interp_x = Hermite_channel(interpolant.Chi_x.xs, interpolant.Chi_x.ys,
                                        vals_n[0], grad_vals_xn[0], grad_vals_xn[1],
                                        cross_x_n, L_x,  L_y)

    interp_y = Hermite_channel(interpolant.Chi_x.xs, interpolant.Chi_x.ys,
                                        vals_n[1], grad_vals_yn[0], grad_vals_yn[1],
                                        cross_y_n, L_x,  L_y)

    return channel_diffeomorphism(interp_x, interp_y, interpolant.mesh)


def compose_maps(remaps, evals, current = []):
     # reverse the order for evaluation
    rev_maps = (remaps + current)[::-1]
    for i in range(len(rev_maps)):
        evals = rev_maps[i](evals)
    return evals

def advect_project_sphere(interp, integrator, t, dt, V, identity = False):
    # combines advection and projection steps and redefines the interpolant
    # basic function for the time stepping in the CMM on sphere
    s_evals = advect(interp, integrator, t, dt, V, identity)
    interp = spline_proj_sphere(interp, s_evals)

    return interp

def advect_project_channel(interp, integrator, t, dt, V, identity = False):
    # combines advection and projection steps and redefines the interpolant
    # basic function for the time stepping in the CMM for a channel flow
    s_evals = advect_channel(interp, integrator, t, dt, V, identity)
    interp = spline_proj_channel(interp, s_evals)

    return interp

from itertools import repeat
def compose_maps_parallel(remaps, evals, pool, n_threads, current = []):

    #split evals by number of threads
    nn = np.shape(evals)[1]//n_threads
    evals_n = [evals[:,i*nn:(i+1)*nn] for i in range(n_threads-1)] + [evals[:,(n_threads-1)*nn:]]
    # pdb.set_trace()
    outs = pool.starmap(compose_maps, zip(repeat(remaps), evals_n, repeat(current)))
    return np.hstack(outs)



ys_out = [None]*6

def updater(integrator, t, dt, yn, i):
    global ys_out
    out = integrator(t,dt,yn)
    ys_out[i] = out

    return

def parallelizer(integrator, stencil_pts, t, dt):
    global ys_out
    ys_out = [None]*6

    thread_count = 6
    thread_list = []

    # create a thread
    #     # Create our thread list and start them by doing nothing

    for _ in range(thread_count):
        # create a thread
        th = threading.Thread()
        # add it to list
        thread_list.append(th)
        # get the thread started and ready for use in python
        th.start()

    for i in range(thread_count):
        thread_list[i] = threading.Thread(target=updater, args = (integrator, t, dt, stencil_pts[i],i) )
        thread_list[i].start()

    for i in range(thread_count):
        thread_list[i].join()

    return
