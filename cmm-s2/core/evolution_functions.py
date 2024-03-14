#/----
"""
This script provides all functionality related to time stepping and contains
most core components of the CMM algorithm for advection.
"""
#/----
# imports
import numpy as np
import pdb
import numba as nb
from . import mesh_functions as meshes
from ..core.spherical_spline import sphere_diffeomorphism
from . import utils
import threading

# Numerical integrators ==============================================================================
# schemes on sphere 

def Euler_step_proj(t,dt,xn,V):
    #backwards Euler step
    v1 = V(t, dt, xn)
    #project back onto sphere
    return div_norm(xn-dt*v1)

def improved_Euler_proj(t,dt,xn,V):
    #backwards in time improved Euler scheme
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
    v2 = V(t, dt, step1)

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

    v1 = V(tpdt, dt, xn)
    step1 = div_norm(xn-(dt/2)*v1)

    v2 = V(t+dt/2, dt, step1)
    xn1 = xn - dt*v2

    return div_norm(xn1)

def div_norm(x):
    # projection onto sphere for integrators, accepts more general arrays
    normalize = 1/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return normalize[None,:]*x

# non-spherical integrators

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

# evolution functions on sphere:

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

    #update the values of the interpolant:
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


def advect_project_sphere(interp, integrator, t, dt, V, identity = False):
    # combines advection and projection steps and redefines the interpolant
    # basic function for the time stepping in the CMM on sphere
    s_evals = advect(interp, integrator, t, dt, V, identity)
    interp = spline_proj_sphere(interp, s_evals)

    return interp

# --------------------------------------------------------
# map composition operations: 

def compose_maps(remaps, evals, current = []):
     # reverse the order for evaluation
    rev_maps = (remaps + current)[::-1]
    for i in range(len(rev_maps)):
        evals = rev_maps[i](evals)
    return evals


# for larger arrays this can speed things up
from itertools import repeat
def compose_maps_parallel(remaps, evals, pool, n_threads, current = []):

    #split evals by number of threads
    nn = np.shape(evals)[1]//n_threads
    evals_n = [evals[:,i*nn:(i+1)*nn] for i in range(n_threads-1)] + [evals[:,(n_threads-1)*nn:]]
    # pdb.set_trace()
    outs = pool.starmap(compose_maps, zip(repeat(remaps), evals_n, repeat(current)))
    return np.hstack(outs)