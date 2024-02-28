# ------------------------------------------------------------------------------
"""
Basic script to test the vorticity equation solver on the sphere
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle
import pyssht as pysh
from ..core.interpolants.spherical_spline import sphere_diffeomorphism
from ..core import evolution_functions as evol
from ..core import mesh_functions as meshes
from ..core import utils
from ..core import dynamical_fields as vel
from ..core.spherical_simulations import euler_simulation_rotating_sphere_remapping, euler_simulation_rotating_sphere

#--------------------------------- Setup --------------------------------------
name = "rossby_wave"
vorticity = vel.rossby_wave

def source(phi,theta):
    return vel.rossby_wave(phi,theta, t= 0)

def psi(phi,theta):
    x,y,z = utils.sphere2cart(phi,theta)

    return -z*((1-z**2)**2 -8*(1-z**2)*x**2 + 8*x**4)
    #return -np.cos(theta)*np.cos(4*phi)*np.sin(theta)**4

def velocity(X):
    x,y,z = X
    psi_x = -32*z*x**3 + 16*z*(1-z**2)*x
    psi_z = -(1-z**2)**2 + 4*z**2*(1-z**2) + 8*x**2*(1-3*z**2) - 8*x**4

    return np.array([y*psi_z - 2*np.pi*y, z*psi_x - x*psi_z + 2*np.pi*x, -y*psi_x])


# convergence test for the circulation functional: 
u_res = [16, 32, 64, 128, 256, 512, 1024]
resolutions = np.array([100*(2**i) for i in range(1,12)])
theta_c = np.pi/4


def line_integral(map, U, N):
    # compute line integral of the pullback of U by map:
    phis = np.linspace(0, 2*np.pi, 2*N, endpoint = False)
    dx = phis[1]-phis[0]
    sample_pts = np.array([np.sin(theta_c)*np.cos(phis), np.sin(theta_c)*np.sin(phis), np.cos(theta_c) + 0*phis])

    # evaluate the map values at the sample points:
    eval_pts = maps(sample_pts)
    grad_map = np.array(maps.eval_grad(sample_pts, eval_pts))
    #pullback of initial velocity:
    u_0 = velocity(eval_pts)
    u_pb = np.array([np.matmul(grad_map[:,:,i].T, u_0[:,i]) for i in range(len(phis))])
    integrand = np.sin(theta_c)*(u_pb[:,1]*np.cos(phis) - u_pb[:,0]*np.sin(phis))

    # approximate the integral via simpson's rule:
    integral = 0

    # Simpson's quadrature
    for i in [x for x in range(2*N-1) if x % 2 == 0]:
        i_end = (i+2) % len(phis) # to account for the periodicity
        integral += (dx/3)*(integrand[i] + 4*integrand[i+1] + integrand[i_end])
    
    return integral

for L in u_res:
    # load map data:
    file = open('./data/convergence_test_%s_%s.txt' %(name, L), "rb")
    maps = pickle.load(file)
    # compute one very fine integral
    integral_fine = line_integral(maps, velocity, N = int(1e6))

    print(L)
    for N in resolutions:
        # these are used to define the velocity field.
        integral = line_integral(maps, velocity, N)

        print(integral-integral_fine)
        

# phis = np.linspace(0, 2*np.pi,N, endpoint = False)
# sample_pts = np.array([np.sin(theta_c)*np.cos(phis), np.sin(theta_c)*np.sin(phis), np.cos(theta_c) + 0*phis])

# # evaluate the map values at the sample points:
# eval_pts = maps(sample_pts)
# grad_map = np.array(maps.eval_grad(sample_pts, eval_pts))
# #pullback of initial velocity:
# u_0 = velocity(eval_pts)
# u_pb = np.array([np.matmul(grad_map[:,:,i].T, u_0[:,i]) for i in range(N)])

# # compute length of the curve by summing at the phis:

# integrand = np.sin(theta_c)*(u_pb[:,1]*np.cos(phis) - u_pb[:,0]*np.sin(phis))

# integral = 2*np.pi*np.sum(integrand)/N