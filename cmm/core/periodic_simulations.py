#/---------
"""
Contains simulations on periodic domains
"""
#/---------

# import statements
import numpy as np
from cmm.core.interpolants import torus_interpolants as interps
from cmm.core import evolution_functions as evol
from . import mesh_functions as meshes
from . import dynamical_fields as vel
from . import utils
from . import spectral_tools as fspace

import pickle, pdb, scipy.io, time

from scipy.fft import fftshift, ifftshift, fft2, ifft2


def euler_simulation_T2(L, Nt, T, Nx, vorticity):

    # define sampling grid for the velocity:
    phi = np.linspace(0, 2*np.pi, L, endpoint = False)
    theta = np.linspace(0, 2*np.pi, L, endpoint = False)
    XX = np.meshgrid(phi, theta)

    # define mesh for the map
    mesh = meshes.torus_mesh(Nx, Nx)

    # initial displacement- CMM
    inverse_map = interps.Hermite_Map(mesh, None, None, identity = True)

    # time steps
    tspan = np.linspace(0, T, 2*Nt, endpoint = False)
    dt = abs(tspan[1]-tspan[0])

    # Initialize velocity field
    vort_init = vorticity(XX)
    u0 = fspace.Biot_Savart(fft2(vort_init), L)

    U = vel.velocity_interp(Vs = [u0], t0 = 0)
    inv_map1 = evol.advect_project_torus(inverse_map, evol.Euler_step, 0, dt, U, identity = True)

    u1 = fspace.Biot_Savart(fft2(vorticity(inv_map1(XX))),L)

    U.Vs.append(u1)

    inv_map2 = evol.advect_project_torus(inverse_map, evol.improved_Euler, 0, dt, U, identity = True)

    u2 = fspace.Biot_Savart(fft2(vorticity(inv_map2(XX))),L)
    U.Vs[1] = u2

    #then one more
    inverse_map = evol.advect_project_torus(inv_map2, evol.RK3_SSP, dt, dt, U)

    u3 = fspace.Biot_Savart(fft2(vorticity(inverse_map(XX))),L)

    U.Vs.append(u3)

    for t in tspan[2::]:

        inverse_map = evol.advect_project_torus(inverse_map, evol.RK3_SSP, t, dt, U)

        omg_n = vorticity(inverse_map(XX))
        u_n = fspace.Biot_Savart(fft2(omg_n), L)
        U = vel.velocity_interp(Vs = [U.Vs[1], U.Vs[2], u_n], t0 = U.t0 + dt)

    return inverse_map
