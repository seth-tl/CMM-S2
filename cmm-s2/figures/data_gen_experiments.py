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

path_to_data = '/mnt/c/Users/setht/Research/GitHub_Repos/CMM-S2'

# define grid to evaluate the error 
Lf = 1000
[thetas, phis] = pysh.sample_positions(Lf, Method = "MWSS", Grid = False)
XX = np.meshgrid(phis, thetas)
s_points = utils.sphere2cart(XX[0],XX[1]) 
eval_pts = np.array([s_points[0].reshape([2*Lf*(Lf+1),]), s_points[1].reshape([2*Lf*(Lf+1),]),
                     s_points[2].reshape([2*Lf*(Lf+1),])]).T

name = sys.argv[1]

if name == "sph_harm":
    vorticity = pickle.load(open(path_to_data + '/data/intial_vorticity_sph_harm_simulation.txt','rb'))
    omega_true = vorticity(XX[0], XX[1])
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

omg_ls = np.zeros([Nt//n_maps, Lf, 2*Lf + 1], dtype = "complex128")
energy_spectra = np.zeros([Nt//n_maps + 1, Lf], dtype = "complex128")

error_enst = []
error_energy = []

# compute the energy and enstrophy at t=0
omega_0 = vorticity(XX[0], XX[1]) + 2*rot_rate*np.cos(XX[1])
omg_T_lms = pysh.forward(omega_0, Lf, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
enst_true = np.absolute(np.sum(omg_T_lms*omg_T_lms.conjugate()))
energy_spectra[0] = sph_tools.energy_spectrum(sph_tools.coeff_array(omg_T_lms,Lf),Lf)

n_U0 = np.absolute(0.5*np.sum(energy_spectra[0]))

# scipy.io.savemat("./data/initial_spectrum_experiment_%s.mat" %(name), {"spectra": energy_spectra[0], "omg_lms": omg_T_lms})


tspan = np.linspace(0, T, Nt, endpoint = False)

# load the submaps:
file = open(path_to_data + file_name + '_all_maps.txt', "rb")
submaps = pickle.load(file)

# # # dummy coefficient array:
coeffs0 = np.zeros([3, 19, np.shape(submaps.mesh.vertices[np.array(submaps.mesh.simplices)])[0]])

for i in range(len(submaps.ns)):
    print(i)
    xyz = submaps(eval_pts, i+1).T 
    # evaluate the maps
    angles = utils.cart2sphere(xyz)
    angles = [angles[0].reshape([Lf+1, 2*Lf]), angles[1].reshape([Lf+1, 2*Lf])]

    # evaluate the vorticity
    omega_num = vorticity(angles[0], angles[1]) + 2*rot_rate*np.cos(angles[1])

    #calculate enstrophy error
    omg_n_lms = pysh.forward(omega_num, Lf, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
    omg_ls[i] = sph_tools.coeff_array(omg_n_lms, Lf)
    
    enst_num = np.absolute(np.sum(omg_n_lms*omg_n_lms.conjugate()))
    error_enst.append((enst_num-enst_true)/enst_true)

    # calculate the energy error
    energy_k = sph_tools.energy_spectrum(sph_tools.coeff_array(omg_n_lms,Lf),Lf)
    energy_spectra[i] = energy_k

    n_Us = 0.5*np.absolute(np.sum(energy_k))
    error_energy.append((n_Us-n_U0)/n_U0)
    
    print("Energy:", error_enst[i])
    print("Enstrophy:", error_energy[i])


    scipy.io.savemat("./data/spectrum_experiment_%s.mat" %(name), {"spectra": energy_spectra, "omg_lms": omg_n_lms, "enst_error": error_enst, "energy_error": error_energy})

Ls = [256, 512, 1024, 2048, 4096]

for L in Ls:
    Lf = L
    [thetas, phis] = pysh.sample_positions(Lf, Method = "MWSS", Grid = False)
    XX = np.meshgrid(phis, thetas)
    s_points = utils.sphere2cart(XX[0],XX[1]) 
    eval_pts = np.array([s_points[0].reshape([2*Lf*(Lf+1),]), s_points[1].reshape([2*Lf*(Lf+1),]),
                        s_points[2].reshape([2*Lf*(Lf+1),])]).T

    xyz = submaps(eval_pts).T 

    angles = utils.cart2sphere(xyz)
    angles = [angles[0].reshape([Lf+1, 2*Lf]), angles[1].reshape([Lf+1, 2*Lf])]

    # evaluate the vorticity
    omega_num = vorticity(angles[0], angles[1]) + 2*rot_rate*np.cos(angles[1])

    #calculate enstrophy error
    omg_n_lms = pysh.forward(omega_num, Lf, Spin = 0, Method = "MWSS", Reality = False, backend = 'ducc', nthreads = 5)
    energy_k = sph_tools.energy_spectrum(sph_tools.coeff_array(omg_n_lms,Lf),Lf)

    scipy.io.savemat("./data/spectrum_experiment_%s_upsampling%s.mat" %(name, L), {"spectra": energy_k, "samples": omega_num})



# for figures:
# multi jet figure samples:
Lf = 1000
[thetas, phis] = pysh.sample_positions(Lf, Method = "MWSS", Grid = False)
XX = np.meshgrid(phis, thetas)
s_points = utils.sphere2cart(XX[0],XX[1]) 
eval_pts = np.array([s_points[0].reshape([2*Lf*(Lf+1),]), s_points[1].reshape([2*Lf*(Lf+1),]),
                     s_points[2].reshape([2*Lf*(Lf+1),])]).T


omega_outs = np.zeros([10,Lf+1, 2*Lf])
j = 0
for i in range(len(submaps.ns)):
    print(i)
    if i % 10 == 0:
        xyz = submaps(eval_pts, i+1).T 
        # evaluate the maps
        angles = utils.cart2sphere(xyz)
        angles = [angles[0].reshape([Lf+1, 2*Lf]), angles[1].reshape([Lf+1, 2*Lf])]

        # evaluate the vorticity
        omega_num = vorticity(angles[0], angles[1]) + 2*rot_rate*np.cos(angles[1])
        omega_outs[j] = omega_num

        scipy.io.savemat("./data/%s_experiment_voriticity_figures.mat" %(name), {"omgs": omega_outs})

        j+=1



