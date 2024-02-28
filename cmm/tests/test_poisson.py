from ..core import spectral_tools as spectral
from ..core import dynamical_fields as vel
import numpy as np
import pdb
import scipy

# test for spectral tools, mainly to evaluate convergence of the Poisson solvers

Ns = [16,32,64,128,256,512,1024,2048]

def omg(X):
    return -90**2*np.sin(90*X[1])

def psi(X):
    return np.sin(90*X[1])



for N in Ns:
    xs = np.linspace(0, 2*np.pi, N, endpoint = False)
    ys = np.linspace(0, 1, N, endpoint = True)
    XX = np.meshgrid(xs,ys)

    dy = ys[1]-ys[0]
    b_l = psi([0,0]); b_u = psi([0,1]);

    source = omg(XX)
    u_true = psi(XX)

    u_num = spectral.Poisson_solve_cylinder(source, dy, b_l, b_u)

    print(np.max(np.absolute(u_num- u_true)))

    # scipy.io.savemat('./data/poisson_solve.mat',{"u_num": u_num.real, "u_true": u_true})
