import stripy
import numpy as np
from scipy.spatial import Delaunay
import pdb



def bary_coords(v_1,v_2,v_3,v):
    """
    v1, v2, v3 define vertices of the containing triangle
    order counter-clockwise. v is the query point.
    """
    denom = det_vec([v_1, v_2, v_3])
    bcc_outs = np.array([det_vec([v, v_2, v_3])/denom,
                         det_vec([v_1,v,v_3])/denom,
                         det_vec([v_1, v_2, v])/denom])
    return bcc_outs.T

def det_vec(A):
    """
    should be input as A = [a,b,c] where a,b,c are considered as columns of A
    """
    v_1, v_2, v_3 = A[0], A[1], A[2]
    det = v_1[:,0]*v_2[:,1]*v_3[:,2] + v_2[:,0]*v_3[:,1]*v_1[:,2] + v_1[:,1]*v_2[:,2]*v_3[:,0] - \
          (v_3[:,0]*v_2[:,1]*v_1[:,2] + v_3[:,1]*v_2[:,2]*v_1[:,0] + v_2[:,0]*v_1[:,1]*v_3[:,2])
    return det

class spherical_triangulation(object):


    """
    generic spherical mesh from a set of points
    """

    def __init__(self, grid):

        vertices = grid.points
        self.points = vertices
        origin = np.array([0,0,0])
        ppoints = np.vstack((origin, grid.points))

        #enforcing the Delaunay condition might not be smart in the future
        mesh_d = Delaunay(ppoints)

        simps0 = mesh_d.simplices - 1
        #now need to remove all the zeros
        simps = np.array([T[T != -1] for T in simps0])

        v_pts2 = vertices[simps]
        dets2 = det_vec([v_pts2[:,0,:], v_pts2[:,1,:], v_pts2[:,2,:]])

        #correct for the orientation
        inds_1 = np.where(dets2 < 0)

        #flip the last two vectices to reverse orientation
        simps[inds_1,2:0:-1] = simps[inds_1,1::]
        v_pts1 = vertices[simps]
        dets1 = det_vec([v_pts1[:,0,:], v_pts1[:,1,:], v_pts1[:,2,:]])

        self.mesh_d = mesh_d
        self.simplices = simps
        self.points = vertices
        self.x, self.y, self.z = grid.x, grid.y, grid.z

        return
