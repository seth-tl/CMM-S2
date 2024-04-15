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
#--------------------------------- Setup --------------------------------------

# define grid to evaluate the error 
Lf = 1000
[thetas, phis] = pysh.sample_positions(Lf, Method = "MWSS", Grid = False)
XX = np.meshgrid(phis, thetas)
s_points = utils.sphere2cart(XX[0],XX[1]) 
eval_pts = np.array([s_points[0].reshape([2*Lf*(Lf+1),]), s_points[1].reshape([2*Lf*(Lf+1),]),
                     s_points[2].reshape([2*Lf*(Lf+1),])]).T

name = sys.argv[1]

if name == "zonal_jet":
    vorticity = vel.zonal_jet
    omega_true = vorticity(XX[0], XX[1])
    rot_rate = 2*np.pi
    T = 1

if name == "rossby_wave":
    vorticity = vel.rossby_wave
    omega_true = vorticity(XX[0], XX[1], t = 1)
    rot_rate = 2*np.pi
    T = 1

if name == "rossby_wave_static":
    vorticity = vel.rossby_wave
    omega_true = vorticity(XX[0], XX[1], t = 0)
    rot_rate = 0
    T = 1

if name == "gaussian_vortex":
    vorticity = vel.gaussian_vortex
    omega_true = vorticity(XX[0], XX[1])  # only to compare for conservation error
    rot_rate = 2*np.pi
    T = 1

remapping = False
if sys.argv[2] == "remapping":
    remapping = True

L_inf = []
Enst = []
Energy = []
edges = []

# compute the energy and enstrophy at t=0
omega_0 = vorticity(XX[0], XX[1]) + 2*rot_rate*np.cos(XX[1])
omg_T_lms = pysh.forward(omega_0, Lf, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
enst_true = np.absolute(np.sum(omg_T_lms*omg_T_lms.conjugate()))
n_U0 = np.absolute(0.5*np.sum(sph_tools.energy_spectrum(sph_tools.coeff_array(omg_T_lms,Lf),Lf)))

for k in range(8):
    # load maps
    if remapping:
        file = open('./data/convergence_test_%s_%s_remapping.txt' %(name, k), "rb")
        maps = pickle.load(file)
        xyz = maps(eval_pts).T  # TODO: get rid of this

    else:
        file = open('./data/convergence_test_%s_%s.txt' %(name, k), "rb")
        maps = pickle.load(file)
        xyz = maps(eval_pts)



    # evaluate the maps
    angles = utils.cart2sphere(xyz)
    angles = [angles[0].reshape([Lf+1, 2*Lf]), angles[1].reshape([Lf+1, 2*Lf])]

    # evaluate the vorticity
    omega_num = vorticity(angles[0], angles[1]) + 2*rot_rate*np.cos(angles[1])

    error = np.absolute(omega_num - 2*rot_rate*np.cos(XX[1]) - omega_true)
    L_inf.append(np.max(error)/np.max(np.absolute(omega_true)))

    #calculate enstrophy error
    omg_n_lms = pysh.forward(omega_num, Lf, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
    enst_num = np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate()))
    Enst.append(np.absolute(enst_num-enst_true)/enst_true)

    # calculate the energy error
    n_Us = 0.5*np.absolute(np.sum(sph_tools.energy_spectrum(sph_tools.coeff_array(omg_n_lms,Lf),Lf)))
    Energy.append(np.absolute(n_U0-n_Us)/n_U0)

    # get edge length:
    ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=k)
    edges.append(np.max(ico0.edge_lengths()))

    
    print("Energy:", Energy)
    print("Error:", L_inf)
    print("Enstrophy:", Enst)

edges = np.array(edges)
l_inf = np.array(L_inf)
energies = np.array(Energy)
enstrophies = np.array(Enst)
orders = np.log(l_inf[1::]/l_inf[0:-1])/np.log(edges[1::]/edges[0:-1])
orders1 = np.log(energies[1::]/energies[0:-1])/np.log(edges[1::]/edges[0:-1])
orders2 = np.log(enstrophies[1::]/enstrophies[0:-1])/np.log(edges[1::]/edges[0:-1])

# print the order of accuracy
print("orders:", orders, orders1, orders2)
print(Energy, Enst, L_inf)

# save the output to be plotts
if remapping:
    scipy.io.savemat("./data/errors_convergence_test_%s_remapping.mat" %(name), {"linf": L_inf, "energy": Energy, "enstrophy": Enst, "edges": edges})
else:
    scipy.io.savemat("./data/errors_convergence_test_%s.mat" %(name), {"linf": L_inf, "energy": Energy, "enstrophy": Enst, "edges": edges})


