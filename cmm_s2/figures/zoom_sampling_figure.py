# ------------------------------------------------------------------------------
"""
Produce all the convergence test data
"""
# ------------------------------------------------------------------------------
import numpy as np
import pdb, stripy, time, pickle, sys
import pyssht as pysh
import scipy.io
from ..core import utils
from ..core import dynamical_fields as vel
from ..core import spherical_harmonic_tools as sph_tools
from ..core.spherical_spline import sphere_diffeomorphism
#--------------------------------- Setup --------------------------------------

path_to_data = ''

name = sys.argv[1]

if name == "sph_harm":
    vorticity = pickle.load(open(path_to_data + '/data/intial_vorticity_sph_harm_simulation.txt','rb'))
    rot_rate = 2*np.pi
    T = 4
    Nt = 1000
    n_maps = 20
    save_steps = 5
    file_name = '/data/spherical_harmonics_simulation'

if name == "multi_jet":
    file_name = '/data/multi_jet_simulation'

    vorticity = vel.multi_jet
    rot_rate = 2*np.pi
    T = 10
    Nt = 1000
    n_maps = 10
    save_steps = 5

if name == "condensate_rotating":
    vorticity = vel.perturbed_rossby_wave
    rot_rate = 2*np.pi
    T = 100
    Nt = 10000
    n_maps = 100
    save_steps = 50
    file_name = '/data/condensate_experiment_rotating'

if name == "condensate_static":
    vorticity = vel.perturbed_rossby_wave
    rot_rate = 0
    T = 100
    Nt = 10000
    n_maps = 100
    save_steps = 50
    file_name = '/data/condensate_experiment_nonrotating'


# load the submaps:
file = open(path_to_data + file_name + '_all_maps.txt', "rb")
submaps = pickle.load(file)

# focal point:
[x0, y0] = (3.22055,1.1963)

N_pts = 1500
# six zoom windows
omega_outs = np.zeros([6, N_pts, N_pts])

j = 0
for k in [-2,-4,-7, -10, -13]:
    print(k)
    xs = np.linspace(x0 - 2**k, x0 + 2**k, N_pts, endpoint = True)
    ys = np.linspace(x0 - 2**k, x0 + 2**k, N_pts, endpoint = True)
    XY = np.meshgrid(xs,ys)
    s_points = utils.sphere2cart(XY[0],XY[1]) 
    eval_pts = np.array([s_points[0].reshape([N_pts**2,]), s_points[1].reshape([N_pts**2,]),
                        s_points[2].reshape([N_pts**2,])]).T


    xyz = submaps(eval_pts).T 
    # evaluate the maps
    angles = utils.cart2sphere(xyz)
    angles = [angles[0].reshape([N_pts, N_pts]), angles[1].reshape([N_pts, N_pts])]

    # evaluate the vorticity
    omega_num = vorticity(angles[0], angles[1]) + 2*rot_rate*np.cos(angles[1])
    omega_outs[j] = omega_num

    scipy.io.savemat("./data/%s_zoom_figures.mat" %(name), {"omgs": omega_outs})
    j+=1
