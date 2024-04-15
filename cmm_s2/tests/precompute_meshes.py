"""
Run this script to precompute the icosahedral meshes. 
"""

import numpy as np
import pdb, stripy, pickle
from ..core import mesh_functions as meshes
from ..core import utils

path_to_repo = ''


for j in range(8):
  ico0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=j)
  mesh = meshes.spherical_triangulation(ico0.points)
  file = open(path_to_repo + '/data/icosahedral_mesh_ref_%s.txt' %j, "wb")
  pickle.dump(mesh, file)
