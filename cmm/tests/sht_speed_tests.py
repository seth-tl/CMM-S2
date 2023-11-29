#
#  Copyright (c) 2010-2018 Centre National de la Recherche Scientifique.
#  written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
#
#  nathanael.schaeffer@univ-grenoble-alpes.fr
#
#  This software is governed by the CeCILL license under French law and
#  abiding by the rules of distribution of free software. You can use,
#  modify and/or redistribute the software under the terms of the CeCILL
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL license and that you accept its terms.
#
 
 
 
import numpy        # numpy for arrays
import shtns        # shtns module compiled and installed using
import time
                    #   pip install shtns
 
lmax = 7            # maximum degree of spherical harmonic representation.
mmax = 3            # maximum order of spherical harmonic representation.
 
sh = shtns.sht(lmax, mmax)  # create sht object with given lmax and mmax (orthonormalized)
# sh = shtns.sht(lmax, mmax, mres=2, norm=shtns.sht_schmidt | shtns.SHT_NO_CS_PHASE)    # use schmidt semi-normalized harmonics
 
nlat, nphi = sh.set_grid()  # build default grid (gauss grid, phi-contiguous)
print(sh.nlat, sh.nphi)     # displays the latitudinal and longitudinal grid sizes.
 
cost = sh.cos_theta         # latitudinal coordinates of the grid as cos(theta)
el = sh.l                   # array of size sh.nlm giving the spherical harmonic degree l for any sh coefficient
l2 = el*(el+1)              # array l(l+1) that is useful for computing laplacian
 
 
 
ylm = sh.spec_array()       # a spherical harmonic spectral array, same as numpy.zeros(sh.nlm, dtype=complex)
vr = sh.spat_array()        # a spatial array, same as numpy.zeros(sh.spat_shape)
 
ylm[sh.idx(1, 0)] = 1.0     # set sh coefficient l=1, m=0 to value 1
 
print(ylm[sh.l == 1])       # print all l=1 coefficients
print(ylm[sh.m == 1])       # print all m=1 coefficients
 
ylm = ylm * l2              # multiply by l(l+1)

start, start_clock = time.perf_counter(), time.process_time()

y = sh.synth(ylm)           # transform sh description ylm into spatial representation y (scalar transform)

finish, finish_clock = time.perf_counter(), time.process_time()

print("wall time (s):", finish - start)
print("CPU time (s):", finish_clock - start_clock)



print(y)                    # display spatial field
 
i_theta = sh.nlat//2
i_phi = 1
print(y[i_theta, i_phi])    # spatial element of coordinate i_theta, i_phi
 
 
zlm = sh.analys(y)          # transform the spatial field back to spectral
print(zlm)
 
 
dy_dt, dy_dp = sh.synth_grad(ylm)   # compute gradients of ylm on a spatial grid.
 
 