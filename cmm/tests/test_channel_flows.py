#/----------
"""
Advection test cases for the channel flow
"""
#/-----------

import numpy as np
import pdb
from ..core.interpolants.channel_interpolant import channel_diffeomorphism, Hermite_channel
from ..core.mesh_functions import channel_mesh
# from ..core import distmesh
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as velocity
import scipy.io


# test set-up----------------------------------------------
pi = np.pi
# # #test maps:
def Chi_x(X):
    return X[0]

def Chi_y(X):
    return X[1]

def DChi_x(X):
    return [X[0]*0 + 1, 0*X[1]]

def DChi_y(X):
    return [X[0]*0, 0*X[1] + 1]

def DChi_xy(X):
    return 0*X[0]


T = 1
#velocity field
def velocity_cylinder(t, xy):
    x,y = xy[0], xy[1]
    vy = np.cos(pi*t/T)*np.sin(2*pi*y)
    vx = 1 + 0*x
    return np.array([vx, vy])

def tracer(X):
    x = X[0]; y = X[1];
    return np.cos(2*pi*x) + np.sin(6*pi*y)


errors = []

Ns = np.array([16, 32, 64, 128, 256, 512, 1024])

# set-up the mesh
L_x = 1; L_y = 1;

def signed_dist_func(X):
    # TODO: get rid of these manipulations
    x1 = 0; x2 = L_x; y1 = 0; y2 = L_y;
    return -np.minimum(-y1+X[1],y2-X[1])




N_pts = 400
xs_fine = np.linspace(0, L_x, N_pts, endpoint = True)
ys_fine = np.linspace(0, L_y, N_pts, endpoint = True)
XX = np.meshgrid(xs_fine,ys_fine)
f_true = tracer(XX)
map_true = np.array([Chi_x(XX), Chi_y(XX)])
#
for N in Ns:
    # set up mesh
    xs = np.linspace(0, L_x, N, endpoint = False)
    ys = np.linspace(0, L_y, N, endpoint = True)
    Xs = np.meshgrid(xs,ys)

    # values
    vals = [Chi_x(Xs), Chi_y(Xs)]
    grad_vals = [DChi_x(Xs), DChi_y(Xs)]
    cross_vals = DChi_xy(Xs)

    # initialize map
    # compute on displacement map
    interp_x = Hermite_channel(xs, ys, 0*vals[0], 0*grad_vals[0][0], 0*grad_vals[0][1],
                              0*cross_vals, L_x,  L_y)

    interp_y = Hermite_channel(xs, ys, vals[1], grad_vals[1][0], grad_vals[1][1],
                              cross_vals, L_x,  L_y)

    mesh = channel_mesh(Xs, signed_dist_func)
    curr_map = channel_diffeomorphism(interp_x, interp_y, mesh)

    tspan = np.linspace(0, 1, 2*N, endpoint = False)
    dt = tspan[1]
    ii = 0
    XX_n = XX
    for t in tspan:

        curr_map = evol.advect_project_channel(curr_map, evol.RK3_SSP, t, dt, velocity_cylinder)
    #     if N == 64:
    #         f_num = tracer(Inv_Map(XX))
    #         scipy.io.savemat("./data/cylinder_test_N%s_iteration_%s.mat" %(N,ii), {"f_num": f_num, "f_true": f_true})
    #     ii +=1
    # #
    f_num = tracer(curr_map(XX))
    error = np.max(np.absolute(f_num - f_true))
    errors.append(error)
    #
    # scipy.io.savemat("./plotting/interp_test1.mat", {"f_num": f_num, "f_true": f_true} )
    print(errors)
