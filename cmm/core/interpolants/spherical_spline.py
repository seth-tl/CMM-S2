#/----
"""
This scripts provides the base class and functions for all spherical interpolants
"""
#/----
# imports
import numpy as np
import igl as IGL
import pdb
from pathos.multiprocessing import ProcessingPool as Pool
from .. import mesh_functions as meshes
from .. import utils
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#===============================================================================

class spherical_spline(object):

    """
    Basic class for spherical spline interpolation for a scalar function
    on sphere.

    TODO: Implement further approximation spaces: CT-split, quintic, etc..

    inputs:
        mesh: spherical_triangulation object from mesh_functions.py
        vals: (N,3) np.array of values of each component at mesh.points()
        grad_vals: gradient of each component at grid.points()
    """

    def __init__(self, mesh, vals, grad_vals, ps_split = None):

        self.mesh = mesh
        self.vals = vals
        self.grad_vals = grad_vals
        # precompute all the coefficients defining the interpolant
        if ps_split == None:
            self.coefficients(mesh.ps_split_mesh())
        else:
            self.coefficients(ps_split)
        return

    def __call__(self, q_pts):
        # evaluate the interpolant at the points defined by q_pts

        # query the triangulation
        bcc, trangs, v_pts = self.mesh.query(q_pts)

        # compute the corresponding containing split triangle
        bb = bary_minmax(bcc)
        nCs = np.stack(Cs[bb[:,0], bb[:,1]], axis = 0)

        # define new vertex points based on split triangle
        v_pts_n = new_vpoints(v_pts, bb)

        #obtain coefficients
        cfs = np.array(self.coeffs)[:,trangs]

        #evaluation functions for PS-split
        out = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs)

        return out

    def coefficients(self, ps_split_mesh):
        inds = np.array(self.mesh.simplices)
        v_pts = self.mesh.vertices[inds]

        #obtain function values at vertices
        c123 = self.vals[inds]

        # arrange for gradient values at the vertices
        grad_f1 = np.array([self.grad_vals[0][inds[:,0]], self.grad_vals[1][inds[:,0]], self.grad_vals[2][inds[:,0]]])
        grad_f2 = np.array([self.grad_vals[0][inds[:,1]], self.grad_vals[1][inds[:,1]], self.grad_vals[2][inds[:,1]]])
        grad_f3 = np.array([self.grad_vals[0][inds[:,2]], self.grad_vals[1][inds[:,2]], self.grad_vals[2][inds[:,2]]])
        gradient_f = [grad_f1, grad_f2, grad_f3]


        self.coeffs = PS_split_coeffs(verts = v_pts, Hs = ps_split_mesh[0],
                                      Gs = ps_split_mesh[1], Es = ps_split_mesh[2],
                                      c123 =  c123, grad_f = gradient_f)
        return

    def gradient(self, q_pts):
        # evaluate the gradient of the interpolant

        # query the triangulation
        bcc, trangs, v_pts = self.mesh.query(q_pts)

        # compute the corresponding containing split triangle
        bb = bary_minmax(bcc)
        nCs = np.stack(Cs[bb[:,0], bb[:,1]], axis = 0)

        # define new vertex points based on split triangle
        v_pts_n = new_vpoints(v_pts, bb)

        #obtain coefficients
        coeffs = np.array(self.coeffs)[:,trangs]

        Cfs = [cffs[cc] for cffs,cc in zip(coeffs.T, nCs)]
        Cfs = np.array(Cfs)
        # compute values for the computation of the gradient of a HBB polynomial
        unos = np.ones([len(v_pts_n[:,0,0]),3])
        xs = (unos*np.array([1,0,0]))
        ys = (unos*np.array([0,1,0]))
        zs = (unos*np.array([0,0,1]))

        bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts.T)

        bcc_x = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], xs)
        bcc_y = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], ys)
        bcc_z = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], zs)
        Bs = [bcc_x, bcc_y, bcc_z]

        outs = grad_HBB(Cfs, bcc_n, Bs)

        return utils.d_Proj(q_pts, outs)


# numba-ize the evaluation function:


class sphere_diffeomorphism(object):

    """
    Basic class for spherical spline interpolation for a mapping of the sphere.
    Current implementation only uses the Powell-Sabin interpolation.

    # TODO: incorporate multi-scale funtionality, different interpolation spaces

    inputs:
        mesh: spherical_triangulation object from mesh_functions.py
        vals: (N,3) np.array of values of each component at mesh.points()
        grad_vals: gradient of each component at grid.points()
    """

    def __init__(self, mesh, vals, grad_vals, ps_split = None):

        self.mesh = mesh
        self.vals = vals
        self.grad_vals = grad_vals
        # precompute all the coefficients defining the interpolant
        if ps_split == None:
            self.coeffs_x, self.coeffs_y, self.coeffs_z = self.assemble_coefficients(mesh.ps_split())
        else:
            self.coeffs_x, self.coeffs_y, self.coeffs_z = self.assemble_coefficients(ps_split)

        return


    def __call__(self, q_pts):

        bcc, trangs, v_pts = self.mesh.query(q_pts)

        bb = bary_minmax(bcc)
        nCs = np.stack(Cs[bb[:,0], bb[:,1]], axis = 0)

        v_pts_n = new_vpoints(v_pts, bb)
        cfs_x, cfs_y, cfs_z = np.array(self.coeffs_x)[:,trangs], np.array(self.coeffs_y)[:,trangs], np.array(self.coeffs_z)[:,trangs]
        
        out_x = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_x)
        out_y = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_y)
        out_z = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_z)

        norm = np.sqrt(out_x**2 + out_y**2 + out_z**2)
        return np.array([out_x/norm, out_y/norm, out_z/norm])


    def eval_grad(self, q_pts, eval_pts):
        # Computes the differential of a map D\varphi_x : T_x M ---> T_{\varphi(x)}M
        # x = q_pts, varphi(x) = eval_pts
        
        bcc, trangs, v_pts = self.mesh.query(q_pts)
        bmm = bary_minmax(bcc)

        v_pts_n = new_vpoints(v_pts, bmm)
        cfs_x = np.array(self.coeffs_x)[:,trangs]
        cfs_y = np.array(self.coeffs_y)[:,trangs]
        cfs_z = np.array(self.coeffs_z)[:,trangs]

        nCs = np.stack(Cs[bmm[:,0], bmm[:,1]], axis = 0)

        Cfs_x = [cffs[cc] for cffs,cc in zip(cfs_x.T, nCs)]
        Cfs_y = [cffs[cc] for cffs,cc in zip(cfs_y.T, nCs)]
        Cfs_z = [cffs[cc] for cffs,cc in zip(cfs_z.T, nCs)]

        Cfs_x = np.array(Cfs_x)
        Cfs_y = np.array(Cfs_y)
        Cfs_z = np.array(Cfs_z)

        unos = np.ones([len(v_pts_n[:,0,0]),3])
        xs = (unos*np.array([1,0,0]))
        ys = (unos*np.array([0,1,0]))
        zs = (unos*np.array([0,0,1]))

        bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts.T)

        bcc_x = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], xs)
        bcc_y = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], ys)
        bcc_z = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], zs)
        Bs = [bcc_x, bcc_y, bcc_z]

        outs_x = grad_HBB(Cfs_x, bcc_n, Bs)
        outs_y = grad_HBB(Cfs_y, bcc_n, Bs)
        outs_z = grad_HBB(Cfs_z, bcc_n, Bs)

        grad_map = [outs_x, outs_y, outs_z]
        norm_X = np.sqrt(eval_pts[0]**2 + eval_pts[1]**2 + eval_pts[2]**2)

        # TODO: write this using matrix multplication
        # X dot \nabla X:
        x_dot_gX = [eval_pts[0]*grad_map[0][0] + eval_pts[1]*grad_map[1][0] + eval_pts[2]*grad_map[2][0],
                    eval_pts[0]*grad_map[0][1] + eval_pts[1]*grad_map[1][1] + eval_pts[2]*grad_map[2][1],
                    eval_pts[0]*grad_map[0][2] + eval_pts[1]*grad_map[1][2] + eval_pts[2]*grad_map[2][2]]

        grad_map_n = [[grad_map[0][0]/norm_X - eval_pts[0]*x_dot_gX[0]/(norm_X**3),
                       grad_map[0][1]/norm_X - eval_pts[0]*x_dot_gX[1]/(norm_X**3),
                       grad_map[0][2]/norm_X - eval_pts[0]*x_dot_gX[2]/(norm_X**3)],
                      [grad_map[1][0]/norm_X - eval_pts[1]*x_dot_gX[0]/(norm_X**3),
                       grad_map[1][1]/norm_X - eval_pts[1]*x_dot_gX[1]/(norm_X**3),
                       grad_map[1][2]/norm_X - eval_pts[1]*x_dot_gX[2]/(norm_X**3)],
                      [grad_map[2][0]/norm_X - eval_pts[2]*x_dot_gX[0]/(norm_X**3),
                       grad_map[2][1]/norm_X - eval_pts[2]*x_dot_gX[1]/(norm_X**3),
                       grad_map[2][2]/norm_X - eval_pts[2]*x_dot_gX[2]/(norm_X**3)]]

        return grad_map_n

    def det_jac(self, q_pts, map_pts):
        # TODO: fix this, incorporate analytic metric...
        #first compute the gradient:
        # eval_pts = self.eval(q_pts, map = False)
        grad_map_n = self.eval_grad(q_pts, map_pts)

        # then form a matrix representation of the grad_map
        # w.r.t. orthonormal bases of each tangent space.

        #first for T_p S^2
        #separate everything away from the poles
        inds0 = np.where(np.absolute(dot(q_pts, [0,1,0])) >= (1-1e-5))[0]
        a1 = np.array(cross(q_pts,[0,1,0]))
        a2 = np.array(cross(q_pts, a1))
        #replace first and last rows
        a1[:,inds0] = cross(q_pts[:,inds0], [0,0,1])
        a2[:,inds0] = cross(q_pts[:,inds0], a1[:,inds0])

        #normalize the direction vectors
        a1_n, a2_n = div_norm(a1.T), div_norm(a2.T)

        # then for T_{\alpha} S^2
        inds0_a = np.where(np.absolute(dot(map_pts, [0,1,0])) >= (1-1e-5))[0]
        a1_a = np.array(cross(map_pts,[0,1,0]))
        a2_a = np.array(cross(map_pts, a1_a))
        #replace first and last rows
        a1_a[:,inds0_a] = cross(map_pts[:,inds0_a], [0,0,1])
        a2_a[:,inds0_a] = cross(map_pts[:,inds0_a], a1_a[:,inds0_a])

        #normalize the direction vectors
        a1_a_n, a2_a_n = div_norm(a1_a.T), div_norm(a2_a.T)


        #change bases and compute determinant
        out1_temp = mat_mul(grad_map_n, a1_n)
        out2_temp = mat_mul(grad_map_n, a2_n)

        out11 = dot(a1_a_n, out1_temp)
        out12 = dot(a2_a_n, out1_temp)
        out21 = dot(a1_a_n, out2_temp)
        out22 = dot(a2_a_n, out2_temp)

        return np.absolute(out11*out22 - out12*out21)


    def stencil_eval(self, q_pts, st_pts):
        # queries at q_pts and evaluates at st_pts.
        # reduces the computational cost of querying at the expense of
        # performing a small extrapolation for the stencil points

        bcc, trangs, v_pts = self.mesh.query(q_pts)
        b_maxmin = bary_minmax(bcc)

        v_pts_n = new_vpoints(v_pts, b_maxmin)
        cfs_x = np.array(self.coeffs_x)[:,trangs]
        cfs_y = np.array(self.coeffs_y)[:,trangs]
        cfs_z = np.array(self.coeffs_z)[:,trangs]
        # in (x,y,z)
        # TODO: vectorize this operation?
        s_x = eval_stencils(v_pts_n, bmm = b_maxmin, st_pts = st_pts, coeffs = cfs_x)
        s_y = eval_stencils(v_pts_n, bmm = b_maxmin, st_pts = st_pts, coeffs = cfs_y)
        s_z = eval_stencils(v_pts_n, bmm = b_maxmin, st_pts = st_pts, coeffs = cfs_z)

        return np.array([s_x, s_y, s_z])


    def assemble_coefficients(self, ps_split):
        """
        Void function to assemble all the coefficients to perform the PS split
        interpolation.

        TODO: vectorize this operation
        """
        inds = np.array(self.mesh.simplices)
        v_pts = self.mesh.vertices[inds]

        grad_f1 = np.array([self.grad_vals[0][0,:][inds[:,0]], self.grad_vals[0][1,:][inds[:,0]], self.grad_vals[0][2,:][inds[:,0]]])
        grad_f2 = np.array([self.grad_vals[0][0,:][inds[:,1]], self.grad_vals[0][1,:][inds[:,1]], self.grad_vals[0][2,:][inds[:,1]]])
        grad_f3 = np.array([self.grad_vals[0][0,:][inds[:,2]], self.grad_vals[0][1,:][inds[:,2]], self.grad_vals[0][2,:][inds[:,2]]])
        grad_fx = [grad_f1, grad_f2, grad_f3]
        c123x = self.vals[0][inds]
        coeffs_x = PS_split_coeffs(verts = v_pts, Hs = ps_split[0], Gs = ps_split[1],
                                    Es = ps_split[2], c123 =  c123x, grad_f = grad_fx)

        grad_f1y = np.array([self.grad_vals[1][0,:][inds[:,0]], self.grad_vals[1][1,:][inds[:,0]], self.grad_vals[1][2,:][inds[:,0]]])
        grad_f2y = np.array([self.grad_vals[1][0,:][inds[:,1]], self.grad_vals[1][1,:][inds[:,1]], self.grad_vals[1][2,:][inds[:,1]]])
        grad_f3y = np.array([self.grad_vals[1][0,:][inds[:,2]], self.grad_vals[1][1,:][inds[:,2]], self.grad_vals[1][2,:][inds[:,2]]])
        grad_fy = [grad_f1y, grad_f2y, grad_f3y]
        c123y = self.vals[1][inds]

        coeffs_y = PS_split_coeffs(verts = v_pts, Hs = ps_split[0], Gs = ps_split[1],
                                    Es = ps_split[2], c123 =  c123y, grad_f = grad_fy)

        grad_f1z = np.array([self.grad_vals[2][0,:][inds[:,0]], self.grad_vals[2][1,:][inds[:,0]], self.grad_vals[2][2,:][inds[:,0]]])
        grad_f2z = np.array([self.grad_vals[2][0,:][inds[:,1]], self.grad_vals[2][1,:][inds[:,1]], self.grad_vals[2][2,:][inds[:,1]]])
        grad_f3z = np.array([self.grad_vals[2][0,:][inds[:,2]], self.grad_vals[2][1,:][inds[:,2]], self.grad_vals[2][2,:][inds[:,2]]])
        grad_fz = [grad_f1z, grad_f2z, grad_f3z]
        c123z = self.vals[2][inds]
        coeffs_z = PS_split_coeffs(verts = v_pts, Hs = ps_split[0], Gs = ps_split[1],
                                    Es = ps_split[2], c123 =  c123z, grad_f = grad_fz)
        
        return coeffs_x, coeffs_y, coeffs_z

class sphere_diffeomorphism_linear(object):

    """
    Basic class for spherical linear spline interpolation for a
    mapping of the sphere

    inputs:
        mesh: spherical_triangulation object from mesh_functions.py
        vals: (N,3) np.array of values of each component at mesh.points()
    """

    def __init__(self, mesh, vals):

        self.mesh = mesh
        self.vals = vals
        #precompute the coefficients:
        inds = np.array(self.mesh.simplices)
        self.coeffs = np.array(self.vals)[:, inds.T]
        return


    def __call__(self, q_pts):

        bcc, trangs, v_pts = self.mesh.query(q_pts)
        cfs = self.coeffs[:,:,trangs]
        out_x = bcc[:,0]*cfs[0,0,:] + bcc[:,1]*cfs[0,1,:] + bcc[:,2]*cfs[0,2,:]
        out_y = bcc[:,0]*cfs[1,0,:] + bcc[:,1]*cfs[1,1,:] + bcc[:,2]*cfs[1,2,:]
        out_z = bcc[:,0]*cfs[2,0,:] + bcc[:,1]*cfs[2,1,:] + bcc[:,2]*cfs[2,2,:]
        norm = np.sqrt(out_x**2 + out_y**2 + out_z**2)
        return np.array([out_x/norm, out_y/norm, out_z/norm])


def PS_split_coeffs(verts, Hs, Gs, Es, c123, grad_f):
    """
    Function which returns the 19 coefficients defining the S^1_2(T_{PS})
    spherical spline interpolant
    """
    v1, v2, v3, v4 = verts[0], verts[1], verts[2], verts[3]
    h12, h21, h23, h32, h13, h31, h41, h42, h43 = Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5], Hs[6], Hs[7], Hs[8]
    g12, g21, g23, g32, g13, g31, g14, g24, g34, A = Gs[0], Gs[1], Gs[2], Gs[3], Gs[4], Gs[5], Gs[6], Gs[7], Gs[8], Gs[9]
    g_1, g_2, g_3 = Es[0], Es[1], Es[2]

    #obtain coefficients and combine appropriately
    c1, c2, c3 = c123[:,0], c123[:,1], c123[:,2]
    c4 = (utils.dot(h12, grad_f[0])/2 - g12[:,0]*c1)/g12[:,1]
    c5 = (utils.dot(h41, grad_f[0])/2 - g14[:,0]*c1)/g14[:,1]
    c6 = (utils.dot(h13, grad_f[0])/2 - g13[:,0]*c1)/g13[:,2]

    c7 = (utils.dot(h23, grad_f[1])/2 - g23[:,0]*c2)/g23[:,1]
    c8 = (utils.dot(h42, grad_f[1])/2 - g24[:,0]*c2)/g24[:,1]
    c9 = (utils.dot(h21, grad_f[1])/2 - g21[:,0]*c2)/g21[:,2]

    c10 = (utils.dot(h31, grad_f[2])/2 - g31[:,0]*c3)/g31[:,1]
    c11 = (utils.dot(h43, grad_f[2])/2 - g34[:,0]*c3)/g34[:,1]
    c12 = (utils.dot(h32, grad_f[2])/2 - g32[:,0]*c3)/g32[:,2]


    c13 = g_1[:,0]*c4 + g_1[:,1]*c9
    c14 = g_2[:,1]*c7 + g_2[:,2]*c12
    c15 = g_3[:,2]*c10 + g_3[:,0]*c6
    c16 = g_1[:,0]*c5 + g_1[:,1]*c8
    c17 = g_2[:,1]*c8 + g_2[:,2]*c11
    c18 = g_3[:,0]*c5 + g_3[:,2]*c11

    #barycentre coords of middle points:
    c19 = A[:,0]*c5 + A[:,1]*c8 + A[:,2]*c11

    return [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19]

#Coefficient list, this is needed for some silly indexing.
Cs = np.array([[0, [18,1,13,7,6,16], [13,2,18,11,10,16]],
      [[18,14,0,17,5,4],0,[14,18,2,17,10,9]],
       [[0,12,18,3,15,4],[1,18,12,7,15,8],0]], dtype = object)
edges_ps = np.array([np.array([np.array([0,0,0]),np.array([3,1,5]), np.array([5,2,3])]),
         np.array([np.array([3,6,0]),np.array([0,0,0]),np.array([6,3,2])]),
         np.array([np.array([0,4,3]),np.array([1,3,4]),np.array([0,0,0])])])

def new_vpoints(v_pts, bmm):

    """
    This defines the new query points and coefficients to evaluate
    a quadratic spherical spline.
    """

    v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
    e1, e2, e3 = utils.div_norm(v1/2+v2/2).T, utils.div_norm(v2/2+v3/2).T, utils.div_norm(v3/2 + v1/2).T
    v4 = utils.div_norm((v1 + v2 + v3)/3).T
    
    #counterclockwise-----------------------
    #first associate every possible edge with a number. The array will look
    # like  [0,[v4,v2,e2],[e2,v3,v4]], [[v4,e3,v1],0,[e3,v4,v3]], [[v1,e1,v4],[v2,v4,e1],0]]

    EE = np.array([v1, v2, v3, v4, e1, e2, e3])
    v_pts_n = np.empty(np.shape(v_pts))
    Js = np.vstack(edges_ps[bmm[:,0], bmm[:,1]])
    N = np.shape(Js)[0]
    v_pts_n1, v_pts_n2, v_pts_n3 = EE[Js[:,0],range(N),:], EE[Js[:,1],range(N),:], EE[Js[:,2],range(N),:]
    v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:] = v_pts_n1, v_pts_n2, v_pts_n3

    return v_pts_n

#clockwise ordering ----------------------------------------
# edges = [[0,[v4,e2,v2],[e2,v4,v3]], [[v4,v1,e3],0,[e3,v3,v4]], [[v1,v4,e1],[v2,e1,v4],0]]
#
# Cs = [[0, [18,13,1,16,6,7], [13,18,2,16,10,11]],
#       [[18,0,14,4,5,17],0,[14,2,18,9,10,17]],
#        [[0,18,12,4,15,3],[1,12,18,8,15,7],0]]

def bary_minmax(X):
    # input: X should be a (N,3) array
    # output: (N,) array of index (0,1,2) for min and max barycentric coord
    # TODO: optimize this operation further
    return np.array([np.argmin(X, axis = 1), np.argmax(X, axis = 1)]).T


def det_vec(A):
    """
    should be input as A = [a,b,c] where a,b,c are considered as columns of A
    """
    v_1, v_2, v_3 = A[0], A[1], A[2]
    det = v_1[:,0]*v_2[:,1]*v_3[:,2] + v_2[:,0]*v_3[:,1]*v_1[:,2] + v_1[:,1]*v_2[:,2]*v_3[:,0] - \
          (v_3[:,0]*v_2[:,1]*v_1[:,2] + v_3[:,1]*v_2[:,2]*v_1[:,0] + v_2[:,0]*v_1[:,1]*v_3[:,2])
    return det

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

def ps_split_eval(v_pts_n, nCs, q_pts, coeffs):
    # evaluation of the Powell-Sabin split

    inds = range(len(nCs))
    #TODO: get rid of this operation or find a better one
    Cfs = [coeffs[nCs[:,0], inds], coeffs[nCs[:,1], inds], coeffs[nCs[:,2], inds],
            coeffs[nCs[:,3], inds], coeffs[nCs[:,4], inds], coeffs[nCs[:,5], inds]]

    bcc_n = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts)

    Cfs = np.array(Cfs).T

    # formula for the evaluation of a quadratic Berstein-Bezier polynomial
    evals = Cfs[:,0]*(bcc_n[:,0]**2) + Cfs[:,1]*(bcc_n[:,1]**2) \
            + Cfs[:,2]*(bcc_n[:,2]**2) \
            + 2*Cfs[:,3]*bcc_n[:,0]*bcc_n[:,1] \
            + 2*Cfs[:,4]*bcc_n[:,1]*bcc_n[:,2] \
            + 2*Cfs[:,5]*bcc_n[:,0]*bcc_n[:,2]

    return evals

def grad_HBB(Cfs, bcc_n, Bs):
    bcc_x, bcc_y, bcc_z = Bs[0], Bs[1], Bs[2]
    L_q = [[2,0,0],[0,2,0], [0,0,2], [1,1,0], [0,1,1], [1,0,1]]
    c_facts = [1,1,1,2,2,2]
    # pdb.set_trace()

    dp_b1 = 2*Cfs[:,0]*bcc_n[:,0] + 2*Cfs[:,3]*bcc_n[:,1] + 2*Cfs[:,5]*bcc_n[:,2]

    dp_b2 = 2*Cfs[:,1]*bcc_n[:,1] + 2*Cfs[:,3]*bcc_n[:,0] + 2*Cfs[:,4]*bcc_n[:,2]

    dp_b3 = 2*Cfs[:,2]*bcc_n[:,2] + 2*Cfs[:,4]*bcc_n[:,1] + 2*Cfs[:,5]*bcc_n[:,0]

    outx = bcc_x[:,0]*dp_b1 +  bcc_x[:,1]*dp_b2 + bcc_x[:,2]*dp_b3
    outy = bcc_y[:,0]*dp_b1 +  bcc_y[:,1]*dp_b2 + bcc_y[:,2]*dp_b3
    outz = bcc_z[:,0]*dp_b1 +  bcc_z[:,1]*dp_b2 + bcc_z[:,2]*dp_b3

    return [outx, outy, outz]

def eval_stencils(v_pts_n, bmm, st_pts, coeffs):
    # 4- point implementation
    # stencils length is exactly four times that of the other arrays
    nCs = np.stack(Cs[bmm[:,0], bmm[:,1]], axis = 0)
    s1 =  ps_split_eval(v_pts_n, nCs, q_pts = st_pts[:,0,:].T, coeffs = coeffs)
    s2 =  ps_split_eval(v_pts_n, nCs, q_pts = st_pts[:,1,:].T, coeffs = coeffs)
    s3 =  ps_split_eval(v_pts_n, nCs, q_pts = st_pts[:,2,:].T, coeffs = coeffs)
    s4 =  ps_split_eval(v_pts_n, nCs, q_pts = st_pts[:,3,:].T, coeffs = coeffs)

    return np.array([s1, s2, s3, s4])




#spline interpolant for the velocity field.
class spline_interp_vec(object):


    def __init__(self, grid, phi, theta, simplices, msimplices, vals, grad_vals):

        self.grid = grid
        # self.simplices = self.grid.simplices
        self.simplices = simplices
        self.msimplices = msimplices
        self.points = grid.points
        self.vals = vals
        self.grad_vals = grad_vals
        self.phi = phi.copy()
        self.theta = theta.copy()
        self.coeffs_x, self.coeffs_y, self.coeffs_z = self.assemble_coefficients(inds = np.array(self.simplices),
                                                                         points = self.points,
                                                                         vals = self.vals,
                                                                         grad_vals = self.grad_vals)
        return


    def eval(self, q_pts, st_pts = None, order = 2):
        bcc, trangs, v_pts = self.query(q_pts)
        bb = bary_minmax(bcc)
        nCs = np.stack(Cs[bb[:,0], bb[:,1]], axis = 0)

        v_pts_n = new_vpoints(v_pts, bb)
        cfs_x = np.array(self.coeffs_x)[:,trangs]
        cfs_y = np.array(self.coeffs_y)[:,trangs]
        cfs_z = np.array(self.coeffs_z)[:,trangs]

        out_x = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_x)
        out_y = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_y)
        out_z = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs_z)

        return np.array([out_x, out_y, out_z])

    def query(self, q_pts):
        [phi,theta] = utils.cart2sphere(q_pts)
        ijs = self.inds(phi,theta)
        return self.containing_simplex_and_bcc_structured(ijs, q_pts.T)

    def assemble_coefficients(self, inds, points, vals, grad_vals):
        """
        Void function to assemble all the coefficients to perform the PS split
        interpolation
        """
        #inds = np.array(self.simplices)
        # All values that don't need to be recomputed:
        v_pts = points[inds]
        v1r, v2r, v3r = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
        v4r = utils.div_norm((v1r + v2r + v3r)/3).T
        e1r, e2r, e3r = utils.div_norm(v1r/2+v2r/2).T, utils.div_norm(v2r/2+v3r/2).T, utils.div_norm(v3r/2 + v1r/2).T
        # Calculate barycentric coords of the edges
        # h12r, h21r, h23r, h32r, h13r, h31r = h(e1r,v1r).T, h(e1r,v2r).T, h(e2r,v2r).T, h(e2r,v3r).T, h(e3r,v1r).T, h(e3r,v3r).T
        #tangential projection
        h12r, h21r, h23r = meshes.sphere_tan_proj(v2r-v1r,v1r).T, meshes.sphere_tan_proj(v1r-v2r,v2r).T, meshes.sphere_tan_proj(v3r-v2r,v2r).T
        h32r, h13r, h31r = meshes.sphere_tan_proj(v2r-v3r,v3r).T, meshes.sphere_tan_proj(v3r-v1r,v1r).T, meshes.sphere_tan_proj(v1r-v3r,v3r).T
        h41, h42, h43 = meshes.sphere_tan_proj(v4r-v1r,v1r).T, meshes.sphere_tan_proj(v4r-v2r,v2r).T, meshes.sphere_tan_proj(v4r-v3r,v3r).T

        g12r, g21r = utils.bary_coords(v1r,e1r,v4r,h12r), utils.bary_coords(v2r,v4r,e1r,h21r)
        g23r, g32r = utils.bary_coords(v2r,e2r,v4r,h23r), utils.bary_coords(v3r,v4r,e2r,h32r)
        g13r, g31r = utils.bary_coords(v1r,v4r,e3r,h13r), utils.bary_coords(v3r,e3r,v4r,h31r)

        g14r = utils.bary_coords(v1r,v4r,e3r,meshes.sphere_tan_proj(v4r,v1r).T)
        g24r = utils.bary_coords(v2r,v4r,e1r,meshes.sphere_tan_proj(v4r,v2r).T)
        g34r = utils.bary_coords(v3r,v4r,e2r,meshes.sphere_tan_proj(v4r,v3r).T)

        Ar = utils.bary_coords(v1r,v2r,v3r,v4r)

        #assemble into nice lists
        verts = [v1r,v2r,v3r,v4r]
        Hs = [h12r.T, h21r.T, h23r.T, h32r.T, h13r.T, h31r.T, h41.T, h42.T, h43.T]
        Gs = [g12r, g21r, g23r, g32r, g13r, g31r, g14r, g24r, g34r, Ar]
        g_1r, g_2r, g_3r = utils.bary_coords(v1r,v2r,v3r,e1r), utils.bary_coords(v1r,v2r,v3r,e2r), utils.bary_coords(v1r,v2r,v3r,e3r)
        gs = [g_1r, g_2r, g_3r]

        # now the non-recyclable quantities
        #in x--------

        grad_f1 = np.array([grad_vals[0][0,:][inds[:,0]], grad_vals[0][1,:][inds[:,0]], grad_vals[0][2,:][inds[:,0]]])
        grad_f2 = np.array([grad_vals[0][0,:][inds[:,1]], grad_vals[0][1,:][inds[:,1]], grad_vals[0][2,:][inds[:,1]]])
        grad_f3 = np.array([grad_vals[0][0,:][inds[:,2]], grad_vals[0][1,:][inds[:,2]], grad_vals[0][2,:][inds[:,2]]])
        grad_fx = [grad_f1, grad_f2, grad_f3]
        c123x = vals[0][inds]
        coeffs_x = PS_split_coeffs(verts = verts, Hs = Hs, Gs = Gs, Es = gs,
                                        c123 =  c123x, grad_f = grad_fx)

        #in y------------
        grad_f1y = np.array([grad_vals[1][0,:][inds[:,0]], grad_vals[1][1,:][inds[:,0]], grad_vals[1][2,:][inds[:,0]]])
        grad_f2y = np.array([grad_vals[1][0,:][inds[:,1]], grad_vals[1][1,:][inds[:,1]], grad_vals[1][2,:][inds[:,1]]])
        grad_f3y = np.array([grad_vals[1][0,:][inds[:,2]], grad_vals[1][1,:][inds[:,2]], grad_vals[1][2,:][inds[:,2]]])
        grad_fy = [grad_f1y, grad_f2y, grad_f3y]
        c123y = vals[1][inds]

        coeffs_y = PS_split_coeffs(verts = verts, Hs = Hs, Gs = Gs, Es = gs,
                                        c123 =  c123y, grad_f = grad_fy)

        # in y------------
        grad_f1z = np.array([grad_vals[2][0,:][inds[:,0]], grad_vals[2][1,:][inds[:,0]], grad_vals[2][2,:][inds[:,0]]])
        grad_f2z = np.array([grad_vals[2][0,:][inds[:,1]], grad_vals[2][1,:][inds[:,1]], grad_vals[2][2,:][inds[:,1]]])
        grad_f3z = np.array([grad_vals[2][0,:][inds[:,2]], grad_vals[2][1,:][inds[:,2]], grad_vals[2][2,:][inds[:,2]]])
        grad_fz = [grad_f1z, grad_f2z, grad_f3z]
        c123z = vals[2][inds]
        coeffs_z = PS_split_coeffs(verts = verts, Hs = Hs, Gs = Gs, Es = gs,
                                        c123 =  c123z, grad_f = grad_fz)

        return coeffs_x, coeffs_y, coeffs_z


    def containing_simplex_and_bcc_structured(self, inds, pts):
        #obtain indices of which half of quadralateral
        phi0, theta0 = self.phi[inds[0]], self.theta[inds[1]]
        v_0 = utils.sphere2cart(phi0, theta0)

        # For the endpoint:
        ijsp1 = [inds[0].copy() + 1, inds[1].copy() + 1]
        ijsp1[0] = ijsp1[0] % (len(self.phi))

        phi_p1, theta_p1 = self.phi[ijsp1[0]], self.theta[ijsp1[1]]

        v_1 = np.array(utils.sphere2cart(phi_p1, theta_p1))

        n_vs = utils.cross(v_0, v_1)
        s_inds = np.heaviside(-utils.dot(n_vs, pts.T), 0).astype(int)
        # s_inds = np.heaviside(-det([v_0, v_1, pts.T]),1).astype(int)
    #       the indices swap since the lambda moves across columns,
    #       whereas theta increases along the rows
        tri_out_temp = self.msimplices[inds[1], inds[0], :, :].reshape([len(pts),2, 4])
        # now assemble triangle list
        l = np.shape(tri_out_temp)[0]
        tri_out = np.array(list(tri_out_temp[np.arange(0,l),s_inds]))
        trangs = tri_out[:,3]
        verts = self.points[tri_out[:,0:3]]

        bcc = utils.bary_coords(verts[:,0,:], verts[:,1,:], verts[:,2,:], pts)

        return bcc, trangs, verts

    def inds(self, phiq, thetaq):
        """
        input a value (phi,theta) to interpolate
        phi0, theta0 should be meshgrids with the same amount of points and
        square.

        phi_s0, theta_s0: Numpy array of 4 meshgrids of the stencils points
        each grid represents one of the four corners.

        output: list of indices, base points of position in the cell also
        position of the stencil point relative to the cell
        """

        dphi = abs(self.phi[1]-self.phi[0])
        dthe = abs(self.theta[1]-self.theta[0])
        #Properly account for values outside of the interval
        phiq_n = phiq % (2*np.pi)
        phi_l = ((phiq_n-self.phi[0])//dphi).astype(int) % (len(self.phi))
        #don't mod this direction, just correct for boundary
        theta_l = ((thetaq-self.theta[0])//dthe).astype(int)

        theta_l[theta_l == (len(self.theta)-1)] = len(self.theta)-2
        # #also compute position within the cell
        phi_c = (phiq_n-self.phi[phi_l])/dphi
        theta_c = (thetaq-self.theta[theta_l])/dthe

        #need added check if query lands on theta_c = 0 or 1 lines.
        #this is a tunable range and only pertains to the interior points
        # TODO: make this less ad hoc
        inds0 = np.where((theta_c <= 0.03) & (theta_l != 0))
        inds1 = np.where((theta_c > 0.85) & (theta_l !=  len(self.theta)-2))

        phi_p1 = (phi_l + 1) % (len(self.phi))
        the_p1 = theta_l + 1

        #in or outside triangle for theta_c = 0.
        v_0 = np.array(utils.sphere2cart(self.phi[phi_l[inds0]], self.theta[theta_l[inds0]]))
        v_1 = np.array(utils.sphere2cart(self.phi[phi_p1[inds0]], self.theta[theta_l[inds0]]))

        n_vs = utils.cross(v_0, v_1)
        q_pts0 = np.array(utils.sphere2cart(phiq_n[inds0], thetaq[inds0]))
        s_inds0 = np.heaviside(utils.dot(n_vs, q_pts0), 0).astype(int)
        # pdb.set_trace()
        theta_l[inds0] = theta_l[inds0]-s_inds0
        #in or outside triangle for theta_c = 0.
        v_01 = np.array(utils.sphere2cart(self.phi[phi_l[inds1]], self.theta[the_p1[inds1]]))
        v_11 = np.array(utils.sphere2cart(self.phi[phi_p1[inds1]], self.theta[the_p1[inds1]]))

        n_vs2 = utils.cross(v_01, v_11)
        q_pts1 = np.array(utils.sphere2cart(phiq_n[inds1], thetaq[inds1]))

        s_inds1 = np.heaviside(-utils.dot(n_vs2, q_pts1), 0).astype(int)

        theta_l[inds1] = theta_l[inds1] + s_inds1

        return [phi_l, theta_l] #, [phi_c, theta_c]


class spline_interp_structured(object):


    def __init__(self, grid, simplices, msimplices, phi, theta, vals, grad_vals):

        self.grid = grid
        # self.simplices = self.grid.simplices
        self.simplices = simplices
        self.msimplices = msimplices
        self.points = grid.points
        self.vals = vals
        self.grad_vals = grad_vals
        self.phi = phi.copy()
        self.theta = theta.copy()
        
        self.coeffs = self.assemble_coefficients(inds = np.array(self.simplices),
                                                points = self.points,
                                                vals = self.vals,
                                                grad_vals = self.grad_vals)
        return

    def __call__(self, q_pts):
        return self.eval(q_pts)
    
    def eval(self, q_pts):
        bcc, trangs, v_pts = self.query(q_pts)
        bb = bary_minmax(bcc)
        nCs = np.stack(Cs[bb[:,0], bb[:,1]], axis = 0)

        v_pts_n = new_vpoints(v_pts, bb)
        cfs = np.array(self.coeffs)[:,trangs]

        outs = ps_split_eval(v_pts_n, nCs = nCs, q_pts = q_pts.T, coeffs = cfs)

        return np.array([outs])

    def query(self, q_pts):
        [phi,theta] = utils.cart2sphere(q_pts)
        ijs = self.inds(phi,theta)
        return self.containing_simplex_and_bcc_structured(ijs, q_pts.T)

    def assemble_coefficients(self, inds, points, vals, grad_vals):
        """
        Void function to assemble all the coefficients to perform the PS split
        interpolation
        """
        #inds = np.array(self.simplices)
        # All values that don't need to be recomputed:
        v_pts = points[inds]
        v1r, v2r, v3r = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
        v4r = utils.div_norm((v1r + v2r + v3r)/3).T
        e1r, e2r, e3r = utils.div_norm(v1r/2+v2r/2).T, utils.div_norm(v2r/2+v3r/2).T, utils.div_norm(v3r/2 + v1r/2).T
        # Calculate barycentric coords of the edges
        # h12r, h21r, h23r, h32r, h13r, h31r = h(e1r,v1r).T, h(e1r,v2r).T, h(e2r,v2r).T, h(e2r,v3r).T, h(e3r,v1r).T, h(e3r,v3r).T
        #tangential projection
        h12r, h21r, h23r = meshes.sphere_tan_proj(v2r-v1r,v1r).T, meshes.sphere_tan_proj(v1r-v2r,v2r).T, meshes.sphere_tan_proj(v3r-v2r,v2r).T
        h32r, h13r, h31r = meshes.sphere_tan_proj(v2r-v3r,v3r).T, meshes.sphere_tan_proj(v3r-v1r,v1r).T, meshes.sphere_tan_proj(v1r-v3r,v3r).T
        h41, h42, h43 = meshes.sphere_tan_proj(v4r-v1r,v1r).T, meshes.sphere_tan_proj(v4r-v2r,v2r).T, meshes.sphere_tan_proj(v4r-v3r,v3r).T

        g12r, g21r = utils.bary_coords(v1r,e1r,v4r,h12r), utils.bary_coords(v2r,v4r,e1r,h21r)
        g23r, g32r = utils.bary_coords(v2r,e2r,v4r,h23r), utils.bary_coords(v3r,v4r,e2r,h32r)
        g13r, g31r = utils.bary_coords(v1r,v4r,e3r,h13r), utils.bary_coords(v3r,e3r,v4r,h31r)

        g14r = utils.bary_coords(v1r,v4r,e3r,meshes.sphere_tan_proj(v4r,v1r).T)
        g24r = utils.bary_coords(v2r,v4r,e1r,meshes.sphere_tan_proj(v4r,v2r).T)
        g34r = utils.bary_coords(v3r,v4r,e2r,meshes.sphere_tan_proj(v4r,v3r).T)

        Ar = utils.bary_coords(v1r,v2r,v3r,v4r)

        #assemble into nice lists
        verts = [v1r,v2r,v3r,v4r]
        Hs = [h12r.T, h21r.T, h23r.T, h32r.T, h13r.T, h31r.T, h41.T, h42.T, h43.T]
        Gs = [g12r, g21r, g23r, g32r, g13r, g31r, g14r, g24r, g34r, Ar]
        g_1r, g_2r, g_3r = utils.bary_coords(v1r,v2r,v3r,e1r), utils.bary_coords(v1r,v2r,v3r,e2r), utils.bary_coords(v1r,v2r,v3r,e3r)
        gs = [g_1r, g_2r, g_3r]

        # now the non-recyclable quantities
        #in x--------
        grad_f1 = np.array([grad_vals[0][inds[:,0]], grad_vals[1][inds[:,0]], grad_vals[2][inds[:,0]]])
        grad_f2 = np.array([grad_vals[0][inds[:,1]], grad_vals[1][inds[:,1]], grad_vals[2][inds[:,1]]])
        grad_f3 = np.array([grad_vals[0][inds[:,2]], grad_vals[1][inds[:,2]], grad_vals[2][inds[:,2]]])
        grad_f = [grad_f1, grad_f2, grad_f3]
        c123 = vals[inds]
        coeffs = PS_split_coeffs(verts = verts, Hs = Hs, Gs = Gs, Es = gs,
                                        c123 =  c123, grad_f = grad_f)


        return coeffs


    def containing_simplex_and_bcc_structured(self, inds, pts):
        #obtain indices of which half of quadralateral
        phi0, theta0 = self.phi[inds[0]], self.theta[inds[1]]
        v_0 = utils.sphere2cart(phi0, theta0)

        # For the endpoint:
        ijsp1 = [inds[0].copy() + 1, inds[1].copy() + 1]
        ijsp1[0] = ijsp1[0] % (len(self.phi))

        phi_p1, theta_p1 = self.phi[ijsp1[0]], self.theta[ijsp1[1]]

        v_1 = np.array(utils.sphere2cart(phi_p1, theta_p1))

        n_vs = utils.cross(v_0, v_1)
        s_inds = np.heaviside(-utils.dot(n_vs, pts.T), 0).astype(int)
        # s_inds = np.heaviside(-det([v_0, v_1, pts.T]),1).astype(int)
    #       the indices swap since the lambda moves across columns,
    #       whereas theta increases along the rows
        tri_out_temp = self.msimplices[inds[1], inds[0], :].reshape([len(pts),2])
        # now assemble triangle list
        l = np.shape(tri_out_temp)[0]
        tri_out = np.array(list(tri_out_temp[np.arange(0,l),s_inds]))
        trangs = tri_out[:,3]
        verts = self.points[tri_out[:,0:3]]

        bcc = utils.bary_coords(verts[:,0,:], verts[:,1,:], verts[:,2,:], pts)

        return bcc, trangs, verts

    def inds(self, phiq, thetaq):
        """
        input a value (phi,theta) to interpolate
        phi0, theta0 should be meshgrids with the same amount of points and
        square.

        phi_s0, theta_s0: Numpy array of 4 meshgrids of the stencils points
        each grid represents one of the four corners.

        output: list of indices, base points of position in the cell also
        position of the stencil point relative to the cell
        """

        dphi = abs(self.phi[1]-self.phi[0])
        dthe = abs(self.theta[1]-self.theta[0])
        #Properly account for values outside of the interval
        phiq_n = phiq % (2*np.pi)
        phi_l = ((phiq_n-self.phi[0])//dphi).astype(int) % (len(self.phi))
        #don't mod this direction, just correct for boundary
        theta_l = ((thetaq-self.theta[0])//dthe).astype(int)

        theta_l[theta_l == (len(self.theta)-1)] = len(self.theta)-2
        # #also compute position within the cell
        phi_c = (phiq_n-self.phi[phi_l])/dphi
        theta_c = (thetaq-self.theta[theta_l])/dthe

        #need added check if query lands on theta_c = 0 or 1 lines.
        #this is a tunable range and only pertains to the interior points
        # TODO: make this less ad hoc
        inds0 = np.where((theta_c <= 0.03) & (theta_l != 0))
        inds1 = np.where((theta_c > 0.85) & (theta_l !=  len(self.theta)-2))

        phi_p1 = (phi_l + 1) % (len(self.phi))
        the_p1 = theta_l + 1

        #in or outside triangle for theta_c = 0.
        v_0 = np.array(utils.sphere2cart(self.phi[phi_l[inds0]], self.theta[theta_l[inds0]]))
        v_1 = np.array(utils.sphere2cart(self.phi[phi_p1[inds0]], self.theta[theta_l[inds0]]))

        n_vs = utils.cross(v_0, v_1)
        q_pts0 = np.array(utils.sphere2cart(phiq_n[inds0], thetaq[inds0]))
        s_inds0 = np.heaviside(utils.dot(n_vs, q_pts0), 0).astype(int)
        # pdb.set_trace()
        theta_l[inds0] = theta_l[inds0]-s_inds0
        #in or outside triangle for theta_c = 0.
        v_01 = np.array(utils.sphere2cart(self.phi[phi_l[inds1]], self.theta[the_p1[inds1]]))
        v_11 = np.array(utils.sphere2cart(self.phi[phi_p1[inds1]], self.theta[the_p1[inds1]]))

        n_vs2 = utils.cross(v_01, v_11)
        q_pts1 = np.array(utils.sphere2cart(phiq_n[inds1], thetaq[inds1]))

        s_inds1 = np.heaviside(-utils.dot(n_vs2, q_pts1), 0).astype(int)

        theta_l[inds1] = theta_l[inds1] + s_inds1

        return [phi_l, theta_l] #, [phi_c, theta_c]


#===============================================================================
#
# class spline_interp_structured(object):
#
#     """
#     Class to perform interpolation spherical spline interpolation on arbitrary
#     spherical triangulations.
#
#     Args:
#         mesh: options
#             1) stripy.sTriangulation object: querying will performed using the
#             stripy package's fortan wrapper. (This performs fairly slowly...)
#             2) 2 arrays defining a grid on [-pi,pi) \times [0,\pi]. Grid cells containing poles will be treated differently.
#             Should allow for much faster querying.
#         rho: callable
#             scalar function rho : R^3 \to R defining the sphere-like surface
#         normal: callable
#             defines normal vector to surface
#         grid: stripy.sTriangulation object defining the underlying triangulation
#         if option 2) is chosen then this is only used for certain attributes
#         and methods, not querying
#
#         vals: (N,) np.array. Scalar field
#             defining the values at points determined by grid.points()
#             in cartesian coordinates.
#         grad_vals:
#               (N,3) np.arrays in a list.
#     """
#
#     def __init__(self, mesh, simplices, msimplices, phi, theta, vals, grad_vals, structured = True):
#
#         self.grid = mesh
#
#
#         #self.points = mesh.points
#         #self.simplices = mesh.simplices
#         #append an index to the end of each simplex element
#         self.vals = vals
#         self.grad_vals = grad_vals
#
#         self.phi = phi.copy()
#         self.theta = theta.copy()
#         # create vertex array
#         # self.points = mesh.points #grid_assemble(phi, theta)
#         if structured == True:
#             self.simplices = simplices
#             self.points = mesh.points.copy()
#
#         if structured == False:
#             self.simplices = mesh.simplices
#             self.points = mesh.points #rho(mesh.points.T)[:,None]*mesh.points
#
#
#         self.msimplices = msimplices
#         self.coeffs = None
#
#         # for Powell-Sabin Split
#         self.assemble_coefficients()
#
#         return
#
#     def __call__(self, phi, theta):
#         N,M = np.shape(phi)[0], np.shape(phi)[1]
#         temp = self.eval(np.array(sphere2cart(phi.reshape([N*M,]),theta.reshape([N*M,]))))
#         return temp.reshape([N,M])
#
#     def eval(self, q_pts, order = 0, angs = False):
#         """
#         Inputs:
#             q_pts  3 (N,) numpy arrays of the Cartesian coordinates of query
#
#         Output: interpolant evaluated at q_pts (N,) numpy array
#         """
#         #project q_pts onto sphere in order to perform a query
#         [phi,theta] = cart2sphere(q_pts)
#         # N, M = len(phi), len(theta)
#         # ijs, coords = self.inds(phi,theta)
#         ijs = self.inds(phi,theta)
#
#         # phi = pth[0].copy().reshape([N*M,])
#         # theta =  pth[1].copy().reshape([N*M,])
#
#         # # q_pts_pt = np.array(sphere2cart(phi,theta)).T
#         #
#         # For the endpoint:
#         ijsp1 = [ijs[0] + 1, ijs[1] + 1]
#         ijs[0][np.where(ijs[0] == len(self.phi))] = 0 # ijsp1[0] % (len(self.phi))
#         ijs[1][np.where(ijs[1] == len(self.phi))] = 0 #ijsp1[1] % (len(self.theta))
#
#         bcc, trangs, v_pts = self.containing_simplex_and_bcc_structured(ijs, q_pts.T)
#         # bcc, trangs, v_pts = self.containing_simplex_and_bcc(q_pts.T)
#         # bcc, trangs, v_pts = bcc_simp_trang(self.points, self.simplices, q_pts)
#
#
#         bb = bary_minmax(bcc)
#         v_pts_n = new_vpoints(v_pts, bb)
#
#         coeffs = np.array(self.coeffs)[:,trangs]
#         nCs = np.stack(Cs[bb[:,0], bb[:,1]], axis = 0)
#
#         outs = C1_PS_Split(v_pts_n, nCs, q_pts = q_pts.T, coeffs = coeffs)
#
#         return outs
#
#     def eval_grad(self, q_pts, order = 0):
#         """
#         Inputs:
#             q_pts  3 (N,) numpy arrays of the Cartesian coordinates of query
#
#         Output: interpolant evaluated at q_pts (N,) numpy array
#         """
#
#         [phi,theta] = cart2sphere(q_pts)
#         N, M = len(phi), len(theta)
#
#         ijs, coords = self.inds(phi,theta)
#
#         #phi = pth[0].copy().reshape([N*M,])
#         #theta =  pth[1].copy().reshape([N*M,])
#
#         #q_pts = np.array(sphere2cart(phi.reshape([N*M,]),theta.reshape([N*M,])))
#
#         # # For the endpoint:
#         # ijsp1 = [ijs[0] + 1, ijs[1] + 1]
#         # ijsp1[0] = ijsp1[0] % (len(self.phi))  #[where(ijsp1[0] == len(self.phi))] = 0
#         # ijsp1[1] = ijsp1[1] % (len(self.theta)) #[where(ijsp1[1] == len(self.phi))] = 0
#
#         bcc, trangs, v_pts = self.containing_simplex_and_bcc_structured(ijs, coords, q_pts.T)
#
#
#         #bcc1, trangs1, v_pts1 = self.containing_simplex_and_bcc(q_pts.T)
#
#         #aj = det_vec([v_pts[:,0,:], v_pts[:,1,:], v_pts[:,2,:]])
#
#         b_maxmin = bary_minmax(bcc)
#         v_pts_n = self.new_vpoints(v_pts, b_maxmin)
#
#         #pdb.set_trace()
#         coeffs = np.array(self.coeffs)[:,trangs]
#         Cs = np.array([[0, [18,1,13,7,6,16], [13,2,18,11,10,16]],
#               [[18,14,0,17,5,4],0,[14,18,2,17,10,9]],
#                [[0,12,18,3,15,4],[1,18,12,7,15,8],0]], dtype = object)
#
#         bb = np.array(b_maxmin)
#         nCs = np.stack(Cs[bb[:,0], bb[:,1]], axis = 0)
#
#         Cfs = np.array([cffs[cc] for cffs,cc in zip(coeffs.T, nCs)])
#         unos = np.ones([len(v_pts_n[:,0,0]),3])
#         xs = (unos*np.array([1,0,0]))
#         ys = (unos*np.array([0,1,0]))
#         zs = (unos*np.array([0,0,1]))
#
#         bcc_n = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts.T)
#
#         bcc_x = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], xs)
#         bcc_y = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], ys)
#         bcc_z = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], zs)
#         Bs = [bcc_x, bcc_y, bcc_z]
#         out_g = grad_HBB(Cfs, bcc_n, Bs)
#         # outs = C1_PS_Split(v_pts_n, bmm = b_maxmin, q_pts = q_pts.T, coeffs = coeffs)
#
#         return d_Proj(q_pts, out_g)
#
#
#
#     def assemble_coefficients(self):
#         """
#         Void function to assemble all the coefficients to perform the PS split
#         interpolation
#         """
#         inds = np.array(self.simplices)
#         v_pts = self.points[inds]
#
#         v1r, v2r, v3r = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
#         v4r = div_norm((v1r + v2r + v3r)/3).T
#         e1r, e2r, e3r = div_norm(v1r/2+v2r/2).T, div_norm(v2r/2+v3r/2).T, div_norm(v3r/2 + v1r/2).T
#         # Calculate barycentric coords of the edges
#         # h12r, h21r, h23r, h32r, h13r, h31r = h(e1r,v1r).T, h(e1r,v2r).T, h(e2r,v2r).T, h(e2r,v3r).T, h(e3r,v1r).T, h(e3r,v3r).T
#         #tangential projection
#         h12r, h21r, h23r = h(v2r-v1r,v1r).T, h(v1r-v2r,v2r).T, h(v3r-v2r,v2r).T
#         h32r, h13r, h31r = h(v2r-v3r,v3r).T, h(v3r-v1r,v1r).T, h(v1r-v3r,v3r).T
#         h41, h42, h43 = h(v4r-v1r,v1r).T, h(v4r-v2r,v2r).T, h(v4r-v3r,v3r).T
#
#         g12r, g21r = bary_coords(v1r,e1r,v4r,h12r), bary_coords(v2r,v4r,e1r,h21r)
#         g23r, g32r = bary_coords(v2r,e2r,v4r,h23r), bary_coords(v3r,v4r,e2r,h32r)
#         g13r, g31r = bary_coords(v1r,v4r,e3r,h13r), bary_coords(v3r,e3r,v4r,h31r)
#
#         g14r = bary_coords(v1r,v4r,e3r,h(v4r,v1r).T)
#         g24r = bary_coords(v2r,v4r,e1r,h(v4r,v2r).T)
#         g34r = bary_coords(v3r,v4r,e2r,h(v4r,v3r).T)
#
#         Ar = bary_coords(v1r,v2r,v3r,v4r)
#
#         #assemble into nice lists
#         verts = [v1r,v2r,v3r,v4r]
#         Hs = [h12r, h21r, h23r, h32r, h13r, h31r]
#         Gs = [g12r, g21r, g23r, g32r, g13r, g31r, g14r, g24r, g34r, Ar]
#         g_1r, g_2r, g_3r = bary_coords(v1r,v2r,v3r,e1r), bary_coords(v1r,v2r,v3r,e2r), bary_coords(v1r,v2r,v3r,e3r)
#         gs = [g_1r, g_2r, g_3r]
#
#         # now the non-recyclable quantities
#         #in x--------
#
#         grad_f1 = np.array([self.grad_vals[0][inds[:,0]], self.grad_vals[1][inds[:,0]], self.grad_vals[2][inds[:,0]]])
#         grad_f2 = np.array([self.grad_vals[0][inds[:,1]], self.grad_vals[1][inds[:,1]], self.grad_vals[2][inds[:,1]]])
#         grad_f3 = np.array([self.grad_vals[0][inds[:,2]], self.grad_vals[1][inds[:,2]], self.grad_vals[2][inds[:,2]]])
#         grad_fx = [grad_f1, grad_f2, grad_f3]
#         c123 = self.vals[inds]
#         self.coeffs = PS_split_coeffs(verts = verts, Hs = Hs, Gs = Gs, gs = gs,
#                                         c123 =  c123, grad_f = grad_fx)
#
#         return
#
#
#
#     def CT_split_coeffs(self, verts, n_s, Hs, Gs, a_ns, a_es, c123, grad_f, grad_f_e):
#         """
#         Function which returns the coefficients needed to evaluate a HBB
#         polynomial
#         """
#         #initialize empty array:
#         # vertices based on max barycentric coordinates
#         v1, v2, v3, v4 = verts[0], verts[1], verts[2], verts[3]
#         h12, h21, h23, h32, h13, h31, h41, h42, h43 = Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5], Hs[6], Hs[7], Hs[8]
#         g12, g21, g23, g32, g13, g31, g14, g24, g34, A = Gs[0], Gs[1], Gs[2], Gs[3], Gs[4], Gs[5], Gs[6], Gs[7], Gs[8], Gs[9]
#         n1, n2, n3 = n_s[0], n_s[1], n_s[2]
#         #simply find all coefficients then combine in appropriate polynomials
#         c1, c2, c3 = c123[:,0], c123[:,1], c123[:,2]
#
#         #pdb.set_trace()
#         c4 = (dot(h12.T, grad_f[0])/3 - g12[:,0]*c1)/g12[:,1]
#         c6 = (dot(h13.T, grad_f[0])/3 - g13[:,2]*c1)/g13[:,1]
#
#         c5 = (dot(h41.T, grad_f[0])/3 - g14[:,0]*c1)/g14[:,1]
#
#         c7 = (dot(h23.T, grad_f[1])/3 - g23[:,0]*c2)/g23[:,1]
#         c8 = (dot(h42.T,grad_f[1])/3 - g24[:,0]*c2)/g24[:,1]
#         c9 = (dot(h21.T, grad_f[1])/3 - g21[:,0]*c2)/g21[:,2]
#
#         c10 = (dot(h31.T, grad_f[2])/3 - g31[:,0]*c3)/g31[:,1]
#         c11 = (dot(h43.T,grad_f[2])/3 - g34[:,0]*c3)/g34[:,1]
#         c12 = (dot(h32.T, grad_f[2])/3 - g32[:,0]*c3)/g32[:,2]
#
#
#         #edge 1
#         a_1 = a_ns[0]
#         a_t1 = a_es[0]
#
#         # c13 = (dot(n1.T,grad_f_e[0])/3 - (at1[:,0]**2)*(a_1[:,0]*c1 + a_1[:,1]*c4 \
#         #         + a_1[:,2]*c5) -2*at1[:,0]*at1[:,1]*(a_1[:,0]*c4 + a_1[:,1]*c9) \
#         #        -(at1[:,1]**2)*(a_1[:,0]*c9 + a_1[:,1]*c2 + a_1[:,2]*c8))/(2*at1[:,0]*at1[:,1]*a_1[:,2])
#
#
#         c13 = (dot(n1.T, grad_f_e[0]) - a_1[:,0]*(3*c1*a_t1[:,0]**2 + 6*c4*a_t1[:,0]*a_t1[:,1] + 3*c9*a_t1[:,1]**2) \
#                -a_1[:,1]*(3*c2*a_t1[:,1]**2 + 3*c4*a_t1[:,0]**2 + 6*c9*a_t1[:,0]*a_t1[:,1]) \
#                - a_1[:,2]*(3*c5*a_t1[:,0]**2 + 3*c8*a_t1[:,1]**2))/(6*a_t1[:,0]*a_t1[:,1]*a_1[:,2])
#
#         #edge 2
#         a_2 = a_ns[1]
#         a_t2 = a_es[1]
#         #
#         # c14 = (dot(n2.T,grad_f_e[1])/3 - (at2[:,0]**2)*(a_2[:,0]*c2 + a_2[:,1]*c7 \
#         #         + a_2[:,2]*c8) -2*at2[:,0]*at2[:,1]*(a_2[:,0]*c7 + a_2[:,1]*c12) \
#         #        -(at2[:,1]**2)*(a_2[:,0]*c12 + a_2[:,1]*c3 + a_2[:,2]*c11))/(2*at2[:,0]*at2[:,1]*a_2[:,2])
#
#         c14 = (dot(n2.T, grad_f_e[1]) - a_2[:,0]*(3*c2*a_t2[:,0]**2 + 6*c7*a_t2[:,0]*a_t2[:,1] + 3*c12*a_t2[:,1]**2) \
#                -a_2[:,1]*(3*c3*a_t2[:,1]**2 + 3*c7*a_t2[:,0]**2 + 6*c12*a_t2[:,0]*a_t2[:,1]) \
#                - a_2[:,2]*(3*c8*a_t2[:,0]**2 + 3*c11*a_t2[:,1]**2))/(6*a_t2[:,0]*a_t2[:,1]*a_2[:,2])
#
#
#         #edge 3
#         a_3 = a_ns[2]
#         a_t3 = a_es[2]
#
#         # c15 = (dot(n3.T,grad_f_e[2])/3 - (at3[:,0]**2)*(a_3[:,0]*c3 + a_3[:,1]*c10 \
#         #         + a_3[:,2]*c11)-2*at3[:,0]*at3[:,1]*(a_3[:,0]*c10 + a_3[:,1]*c6) \
#         #        -(at3[:,1]**2)*(a_3[:,0]*c6 + a_3[:,1]*c1 + a_3[:,2]*c5))/(2*at3[:,0]*at3[:,1]*a_3[:,2])
#
#         c15 = (dot(n3.T, grad_f_e[2]) - a_3[:,0]*(3*c3*a_t3[:,0]**2 + 6*c10*a_t3[:,0]*a_t3[:,1] + 3*c6*a_t3[:,1]**2) \
#                -a_3[:,1]*(3*c1*a_t3[:,1]**2 + 3*c10*a_t3[:,0]**2 + 6*c6*a_t3[:,0]*a_t3[:,1]) \
#                - a_3[:,2]*(3*c11*a_t3[:,0]**2 + 3*c5*a_t3[:,1]**2))/(6*a_t3[:,0]*a_t3[:,1]*a_3[:,2])
#
#
#         c16 = (A[:,0]*c5 + A[:,1]*c13 + A[:,2]*c15)/3
#         c17 = (A[:,1]*c13 + A[:,1]*c8 + A[:,2]*c14)/3
#         c18 = (A[:,0]*c15 + A[:,1]*c14 + A[:,2]*c11)/3
#
#         #barycentre coords of middle points:
#         c19 = (A[:,0]*c16 + A[:,1]*c17 + A[:,2]*c18)/3
#
#
#         return [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19]
#
#
#     def PS_split_coeffs(self, verts, Hs, Gs, gs, c123, grad_f):
#         """
#         Function which returns the coefficients needed to evaluate a HBB
#         polynomial on the Powell-Sabin Split
#         """
#         #initialize empty array:
#         # vertices based on max barycentric coordinates
#         v1, v2, v3, v4 = verts[0], verts[1], verts[2], verts[3]
#         h12, h21, h23, h32, h13, h31, h41, h42, h43 = Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5], Hs[6], Hs[7], Hs[8]
#         g12, g21, g23, g32, g13, g31, g14, g24, g34, A = Gs[0], Gs[1], Gs[2], Gs[3], Gs[4], Gs[5], Gs[6], Gs[7], Gs[8], Gs[9]
#         g_1, g_2, g_3 = gs[0], gs[1], gs[2]
#
#         #simply find all coefficients then combine in appropriate polynomials
#         c1, c2, c3 = c123[:,0], c123[:,1], c123[:,2]
#         #see Spherical_Splines_Computational_Methods
#         c4 = (dot(h12.T, grad_f[0])/2 - g12[:,0]*c1)/g12[:,1]
#         c5 = (dot(h41.T,grad_f[0])/2 - g14[:,0]*c1)/g14[:,1]  #A[:,0]*c1 + A[:,1]*c4 + A[:,2]*c6
#         c6 = (dot(h13.T, grad_f[0])/2 - g13[:,0]*c1)/g13[:,2]
#
#         c7 = (dot(h23.T, grad_f[1])/2 - g23[:,0]*c2)/g23[:,1]
#         c8 = (dot(h42.T,grad_f[1])/2 - g24[:,0]*c2)/g24[:,1]  #A[:,0]*c9 + A[:,1]*c2 + A[:,2]*c7
#         c9 = (dot(h21.T, grad_f[1])/2 - g21[:,0]*c2)/g21[:,2]
#
#         c10 = (dot(h31.T, grad_f[2])/2 - g31[:,0]*c3)/g31[:,1]
#         c11 = (dot(h43.T,grad_f[2])/2 - g34[:,0]*c3)/g34[:,1]
#         c12 = (dot(h32.T, grad_f[2])/2 - g32[:,0]*c3)/g32[:,2]
#
#         c13 = g_1[:,0]*c4 + g_1[:,1]*c9
#         c14 = g_2[:,1]*c7 + g_2[:,2]*c12
#         c15 = g_3[:,2]*c10 + g_3[:,0]*c6
#         c16 = g_1[:,0]*c5 + g_1[:,1]*c8
#         c17 = g_2[:,1]*c8 + g_2[:,2]*c11
#         c18 = g_3[:,0]*c5 + g_3[:,2]*c11
#
#         #barycentre coords of middle points:
#         c19 = A[:,0]*c5 + A[:,1]*c8 + A[:,2]*c11
#         #c19 = (1/3)*(c5 + c8 + c11)
#         #.......
#
#         return [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19]
#
#     def PS_split(self, v_pts, bcc):
#         #initialize empty array:
#         # vertices based on max barycentric coordinates
#         b_maxmin = bary_minmax(bcc)
#         v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
#         e1, e2, e3 = div_norm(v1/2+v2/2).T, div_norm(v2/2+v3/2).T, div_norm(v3/2 + v1/2).T
#
#         #edges = [[0,[2,1],[1,2]], [[1,0],0,[0,2]], [[1,0],[0,1],0]]
#         v4 = div_norm((v1 + v2 + v3)/3).T
#         #counterclockwise
#         edges = [[0,[v4,v2,e2],[e2,v3,v4]], [[v4,e3,v1],0,[e3,v4,v3]], [[v1,e1,v4],[v2,v4,e1],0]]
#         #clockwise
#         #edges = [[0,[v4,e2,v2],[e2,v4,v3]], [[v4,v1,e3],0,[e3,v3,v4]], [[v1,v4,e1],[v2,e1,v4],0]]
#
#
#         v_pts_n = np.empty(np.shape(v_pts))
#
#         #Then input new v_pts into CT
#         for i in range(len(b_maxmin)):  #figure out how to turn this into a slicing
#             e = b_maxmin[i]
#             js = edges[e[0]][e[1]]
#             v_pts_n[i,0,:] = js[0][i,:]
#             v_pts_n[i,1,:] = js[1][i,:]
#             v_pts_n[i,2,:] = js[2][i,:]
#
#         return v_pts_n
#
#     def C1_PS_split(self, v_pts_n, bmm, q_pts, coeffs):
#
#         #Coefficient list
#         #counterclockwise
#         Cs = np.array([[0, [18,1,13,7,6,16], [13,2,18,11,10,16]],
#               [[18,14,0,17,5,4],0,[14,18,2,17,10,9]],
#                [[0,12,18,3,15,4],[1,18,12,7,15,8],0]], dtype = object)
#         # # clockwise
#         # Cs = np.array([[0, [18,13,1,16,6,7], [13,18,2,16,10,11]],
#         #       [[18,0,14,4,5,17],0,[14,2,18,9,10,17]],
#         #        [[0,18,12,4,15,3],[1,12,18,8,15,7],0]], dtype = object)
#
#
#         L_q = [[2,0,0],[0,2,0], [0,0,2], [1,1,0], [0,1,1], [1,0,1]]
#         c_facts = [1,1,1,2,2,2]
#
#         bb = np.array(bmm)
#         nCs = np.stack(Cs[bb[:,0], bb[:,1]], axis = 0)
#
#         Cfs = []
#         for cffs, cc in zip(coeffs.T, nCs):
#             Cfs.append(cffs[cc])
#         #pdb.set_trace()
#
#         bcc_n = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts)
#         Cfs = np.array(Cfs)
#
#         evals = c_facts[0]*Cfs[:,0]*(bcc_n[:,0]**L_q[0][0])*(bcc_n[:,1]**L_q[0][1])*(bcc_n[:,2]**L_q[0][2]) \
#                 + c_facts[1]*Cfs[:,1]*(bcc_n[:,0]**L_q[1][0])*(bcc_n[:,1]**L_q[1][1])*(bcc_n[:,2]**L_q[1][2]) \
#                 + c_facts[2]*Cfs[:,2]*(bcc_n[:,0]**L_q[2][0])*(bcc_n[:,1]**L_q[2][1])*(bcc_n[:,2]**L_q[2][2]) \
#                 + c_facts[3]*Cfs[:,3]*(bcc_n[:,0]**L_q[3][0])*(bcc_n[:,1]**L_q[3][1])*(bcc_n[:,2]**L_q[3][2]) \
#                 + c_facts[4]*Cfs[:,4]*(bcc_n[:,0]**L_q[4][0])*(bcc_n[:,1]**L_q[4][1])*(bcc_n[:,2]**L_q[4][2]) \
#                 + c_facts[5]*Cfs[:,5]*(bcc_n[:,0]**L_q[5][0])*(bcc_n[:,1]**L_q[5][1])*(bcc_n[:,2]**L_q[5][2])
#
#
#         return np.array(evals)
#
#     def C1_CT_split(self, inds, bcc, q_pts, coeffs):
#         #initialize empty array:
#         # vertices based on max barycentric coordinates
#         v_pts = self.grid.points[inds]
#         b_maxmin = self.bary_min(bcc)
#         v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
#         e1, e2, e3 = div_norm(v1/2+v2/2).T, div_norm(v2/2+v3/2).T, div_norm(v3/2 + v1/2).T
#
#         v4 = div_norm((v1 + v2 + v3)/3).T
#
#         #counterclockwise-----------------------
#         edges = [[v3,v4,v2],[v4,v3,v1],[v1,v2,v4]]
#
#         #Coefficient list
#         Cs = [[2,18,1,10,11,16,17,6,7,13],
#               [18,2,0,17,15,9,10,4,5,14],
#               [0,1,18,3,4,7,8,15,16,12]]
#
#         L_c = [[3,0,0], [0,3,0], [0,0,3], [2,1,0], [2,0,1], [0,2,1], [1,2,0], [1,0,2],
#                [0,1,2],[1,1,1]]
#
#         v_pts_n = np.empty(np.shape(v_pts))
#         evals = []
#         #Then input new v_pts into CT
#         for i in range(len(b_maxmin)):  #figure out how to turn this into a slicing
#             e = b_maxmin[i]
#             js = edges[e]
#             v_pts_n[i,0,:] = js[0][i,:]
#             v_pts_n[i,1,:] = js[1][i,:]
#             v_pts_n[i,2,:] = js[2][i,:]
#             cc = Cs[e]
#
#             bcc_n = bary_coords(js[0][i,:], js[1][i,:], js[2][i,:],
#                                 q_pts[i,:], scl = True)
#             j = 0
#             temp_val = 0
#             for T in L_c:
#                 c_fact = factorial(sum(T))/(factorial(T[0])*factorial(T[1])*factorial(T[2]))
#                 temp_val += c_fact*coeffs[cc[j]][i]*(bcc_n[0]**T[0])*(bcc_n[1]**T[1])*(bcc_n[2]**T[2])
#                 j+=1
#             evals.append(temp_val)
#
#         return np.array(evals)
#
#     def CT_split(self, v_pts, bcc):
#         #initialize empty array:
#         # vertices based on max barycentric coordinates
#
#         #************
#         #!!! this might need to be max
#         b_min = self.bary_min(bcc)
#
#         # possible bug ^^^^
#         #*************************
#
#         v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
#
#         V = []
#         for i in range(len(b_min)):
#             indices = [0,1,2]
#             indices.remove(b_min[i])
#             V.append(v_pts[i, indices, :])
#         V = np.array(V)
#         v4 = div_norm((v1 + v2 + v3)/3)
#
#         #Then input new v_pts into CT
#
#         # watch out this might mess ordering
#         v_pts_n = np.empty(np.shape(v_pts))
#         # pdb.set_trace()
#         v_pts_n[:,0,:] = v4.T.copy()
#         v_pts_n[:,1,:] = V[:,0,:].copy()
#         v_pts_n[:,2,:] = V[:,1,:].copy()
#
#         return v_pts_n
#
#     def control_points(self, v_pts, b_min = None, which = "cubic"):
#
#         if which == "cubic":
#             #initialize empty array:
#             s = np.shape(v_pts)
#             out = np.zeros([s[0], 3, 10])
#             # vertices
#             # pdb.set_trace()
#             v1, v2, v3 = v_pts[:,0,:], v_pts[:,1,:], v_pts[:,2,:]
#             out[:,:,0], out[:,:,1], out[:,:,2] = v1.copy(), v2.copy(), v3.copy()
#             # then edges in order:
#             bary = (v1 + v2 + v3)/3
#
#             out[:,:,3] = div_norm((2*v1 + v2)/3).T
#             out[:,:,4] = div_norm((2*v1 + v3)/3).T
#             out[:,:,5] = div_norm((2*v2 + v3)/3).T
#             out[:,:,6] = div_norm((2*v2 + v1)/3).T
#
#
#             out[:,:,7] = div_norm((2*v3 + v1)/3).T
#             out[:,:,8] = div_norm((2*v3 + v2)/3).T
#
#             #then interior points in lexicographical ordering
#             out[:,:,9] = div_norm(bary).T
#
#         if which == "quadratic":
#             #initialize empty array:
#             s = np.shape(v_pts)
#             out = np.zeros([s[0], 3, 6])
#             # vertices
#             # pdb.set_trace()
#             v1, v2, v3 = v_pts[:,0,:], v_pts[:,1,:], v_pts[:,2,:]
#             out[:,:,0], out[:,:,1], out[:,:,2] = v1.copy(), v2.copy(), v3.copy()
#             # then edges in order:
#
#             out[:,:,3] = div_norm((v1 + v2)/2).T
#             out[:,:,4] = div_norm((v2 + v3)/2).T
#             out[:,:,5] = div_norm((v3 + v1)/2).T
#
#
#         return out
#
#     def bary_minmax(self,X):
#         """
#         x should be a (N,3) array
#
#         output:
#         coefficient of maximum barycentric coordinates
#
#         TODO: consider using the built-in sorted function
#         """
#         #return [[np.where(x == min(x))[0][0], np.where(x == max(x))[0][0]] for x in X]
#         return [[list(x).index(min(x)), list(x).index(max(x))] for x in X]
#
#     def bary_max(self, X):
#         return [np.where(x == np.max(x))[0][0] for x in X]
#     def bary_min(self, X):
#         return [np.where(x == np.min(x))[0][0] for x in X]
#
#
#     def inds(self, phiq, thetaq):
#         """
#         input a value (phi,theta) to interpolate
#         phi0, theta0 should be meshgrids with the same amount of points and
#         square.
#
#         phi_s0, theta_s0: Numpy array of 4 meshgrids of the stencils points
#         each grid represents one of the four corners.
#
#         output: list of indices, base points of position in the cell also
#         position of the stencil point relative to the cell
#         """
#
#         dphi = abs(self.phi[1]-self.phi[0])
#         dthe = abs(self.theta[1]-self.theta[0])
#         #Properly account for values outside of the interval
#         # #parse list a bit. --------------------------
#         # p_u, p_l = phiq[np.where(phiq >= pi)], phiq[np.where(phiq < -pi)]
#         # # #t_u, t_l = thetaq[np.where(thetaq >= pi)], thetaq[np.where(thetaq < -pi)]
#         # #
#         # p_u = (p_u + pi) % (2*pi) - pi
#         # # #t_u = (t_u + pi) % (2*pi) - pi
#         # p_l = (p_l + pi) % (2*pi) - pi
#         # # #t_l = (t_l + pi) % (2*pi) - pi
#         # #
#         # # # Make transformation
#         # phiq[np.where(phiq >= pi)] = p_u
#         # phiq[np.where(phiq < -pi)] = p_l
#         #thetaq[np.where(thetaq >= pi)], thetaq[np.where(thetaq < -pi)] = t_u, t_l
#         # #----------------------------------------------
#
#         phiq_n = phiq % (2*pi)
#
#         phi_l = ((phiq_n-self.phi[0])//dphi).astype(int) % (len(self.phi))
#         #don't mod this direction, just correct for boundary
#         theta_l = ((thetaq-self.theta[0])//dthe).astype(int)
#
#         theta_l[theta_l == (len(self.theta)-1)] = len(self.theta)-2
#         # #also compute position within the cell
#         phi_c = (phiq_n-self.phi[phi_l])/dphi
#         theta_c = (thetaq-self.theta[theta_l])/dthe
#
#         #need added check if query lands on theta_c = 0 or 1 lines.
#         #this is a tunable range
#         #only pertain to the interior points
#         inds0 = np.where((theta_c <= 0.03) & (theta_l != 0))
#         inds1 = np.where((theta_c > 0.85) & (theta_l !=  len(self.theta)-2))
#
#         phi_p1 = (phi_l + 1) % (len(self.phi))
#         the_p1 = theta_l + 1
#
#         #in or outside triangle for theta_c = 0.
#         v_0 = np.array(sphere2cart(self.phi[phi_l[inds0]], self.theta[theta_l[inds0]]))
#         v_1 = np.array(sphere2cart(self.phi[phi_p1[inds0]], self.theta[theta_l[inds0]]))
#
#         n_vs = cross(v_0, v_1)
#         q_pts0 = np.array(sphere2cart(phiq_n[inds0], thetaq[inds0]))
#         s_inds0 = np.heaviside(dot(n_vs, q_pts0), 0).astype(int)
#         # pdb.set_trace()
#         theta_l[inds0] = theta_l[inds0]-s_inds0
#         #in or outside triangle for theta_c = 0.
#         v_01 = np.array(sphere2cart(self.phi[phi_l[inds1]], self.theta[the_p1[inds1]]))
#         v_11 = np.array(sphere2cart(self.phi[phi_p1[inds1]], self.theta[the_p1[inds1]]))
#
#         n_vs2 = cross(v_01, v_11)
#         q_pts1 = np.array(sphere2cart(phiq_n[inds1], thetaq[inds1]))
#
#         s_inds1 = np.heaviside(-dot(n_vs2, q_pts1), 0).astype(int)
#
#         theta_l[inds1] = theta_l[inds1] + s_inds1
#
#
#         return [phi_l, theta_l] #, [phi_c, theta_c]
#
#
#     def containing_simplex_and_bcc_structured(self, inds, pts):
#         """
#         Returns the simplices containing pts
#         and the local barycentric, normalised coordinates.
#
#         Input:
#             inds: indices over meshgrid
#             coords: normalized coordinates in the meshgrid
#             pts: query points
#
#         Returns:
#             bcc : normalised barycentric coordinates
#             tri : simplices containing pts
#
#         Notes:
#
#         """
#
#         #obtain indices of which half of quadralateral
#         # s_inds = (np.heaviside(coords[1]-coords[0],0)).astype(int).reshape([len(pts),])
#         #s_inds = s_inds.reshape([len(pts),])
#         phi0, theta0 = self.phi[inds[0]], self.theta[inds[1]]
#         v_0 = sphere2cart(phi0, theta0)
#
#         # For the endpoint:
#         ijsp1 = [inds[0].copy() + 1, inds[1].copy() + 1]
#         ijsp1[0] = ijsp1[0] % (len(self.phi))
#
#         phi_p1, theta_p1 = self.phi[ijsp1[0]], self.theta[ijsp1[1]]
#
#         v_1 = np.array(sphere2cart(phi_p1, theta_p1))
#
#         n_vs = cross(v_0, v_1)
#         s_inds = np.heaviside(-dot(n_vs, pts.T), 0).astype(int)
#         # s_inds = np.heaviside(-det([v_0, v_1, pts.T]),1).astype(int)
#
# #       the indices swap since the lambda moves across columns,
# #       whereas theta increases along the rows
#         tri_out_temp = self.msimplices[inds[1], inds[0], :].reshape([len(pts),2])
#         # now assemble triangle list
#         l = np.shape(tri_out_temp)[0]
#         tri_out = np.array(list(tri_out_temp[np.arange(0,l),s_inds]))
#         trangs = tri_out[:,3]
#         verts = self.points[tri_out[:,0:3]]
#
#         bcc = bary_coords(verts[:,0,:], verts[:,1,:], verts[:,2,:], pts)
#
#         return bcc, trangs, verts
#
# #===============================================================================

#
#
#     def stencil_pts(self, eps = 1e-5):
#         """
#
#         """
#         verts = self.points
#         # form direction vectors from every vertex
#         a1 = np.array(cross(verts.T,[0,0,1])).T
#         a2 = np.array(cross(verts.T, a1.T)).T
#
#         #replace first and last rows
#         a1[0,:] = cross(verts[0,:], [1/np.sqrt(2),1/np.sqrt(2),0])
#         a1[-1,:] = cross(verts[-1,:], [1/np.sqrt(2),1/np.sqrt(2),0])
#
#         a2[0,:] = cross(verts[0,:], a1[0,:])
#         a2[-1,:] = cross(verts[-1,:], a1[-1,:])
#
#         #normalize the direction vectors
#         a1_n, a2_n = div_norm(a1).T, div_norm(a2).T
#
#         # pdb.set_trace()
#         #define the stencil points
#         # 4 - point averaged
#         eps_1p = verts - eps*a1_n - eps*a2_n
#         eps_1m = verts + a1_n*eps - eps*a2_n
#         eps_2p = verts - eps*a1_n + eps*a2_n
#         eps_2m = verts + a1_n*eps + a2_n*eps
#
#         #along coordinate direction
#         # eps_1p = verts + eps*a1_n
#         # eps_1m = verts - a1_n*eps
#         # eps_2p = verts + eps*a2_n
#         # eps_2m = verts - a2_n*eps
#
#         # # #normalize once again and output
#         # dirs = [div_norm(eps_1p), div_norm(eps_1m), div_norm(eps_2p),
#         #         div_norm(eps_2m)]
#
#         dirs = [g_proj(eps_1p, verts), g_proj(eps_1m, verts), g_proj(eps_2p, verts),
#                 g_proj(eps_2m, verts)]
#
#         #save directions for later use
#         self.tan_vects = [a1_n, a2_n]
#         # # 6-point stencils implementation
#         # v1 = np.array([verts[:,0] + eps, verts[:,1], verts[:,2]]).T
#         # v2 = np.array([verts[:,0] - eps, verts[:,1], verts[:,2]]).T
#         #
#         # v3 = np.array([verts[:,0], verts[:,1] + eps, verts[:,2]]).T
#         # v4 = np.array([verts[:,0], verts[:,1] - eps, verts[:,2]]).T
#         #
#         # v5 = np.array([verts[:,0], verts[:,1], verts[:,2] + eps]).T
#         # v6 = np.array([verts[:,0], verts[:,1], verts[:,2] - eps]).T
#         #
#         #
#         # dirs = [div_norm(v1), div_norm(v2), div_norm(v3), div_norm(v4),
#         #           div_norm(v5), div_norm(v6)]
#         # #dirs = [v1.T, v2.T, v3.T, v4.T, v5.T, v6.T]
#
#         return dirs
#
#
# class spline_interp_vec_divfree(object):
#
#     """
#     Class to perform interpolation on a spherical geodesic grid for a vector
#     valued function combined. Recylces computations from the spline_interp_structured
#     eval() method.
#
#     Args:
#         grid: icosohedral grid defined by stripy.sTriangulation object
#
#             (N,3) np.array if map
#                 defines values at points determined by grid.points()
#                 in cartesian coordinates.
#     """
#
#     def __init__(self, grid, phi, theta, simplices, msimplices, vals, grad_vals, st_pts = None, tan_vects = None):
#
#         self.grid = grid
#         # self.simplices = self.grid.simplices
#         self.simplices = simplices
#         self.msimplices = msimplices
#         self.points = grid.points
#         self.vals = vals
#         self.grad_vals = grad_vals
#
#         if st_pts == None:
#             self.st_pts = self.stencil_pts()
#         else:
#             self.st_pts = st_pts
#             self.tan_vects = tan_vects
#
#         self.phi = phi.copy()
#         self.theta = theta.copy()
#
#
#         self.coeffs = self.assemble_coefficients(inds = np.array(self.simplices),
#                                              points = self.points,
#                                              vals = self.vals,
#                                              grad_vals = self.grad_vals)
#
#         return
#
#
#     def eval(self, q_pts, st_pts = None, order = 2, map = False):
#         #performs x \times \nabla \psi
#
#         [phi,theta] = cart2sphere(q_pts)
#         N, M = len(phi), len(theta)
#         ijs = self.inds(phi,theta)
#         bcc, trangs, v_pts = self.containing_simplex_and_bcc_structured(ijs, q_pts.T)
#         # bcc, trangs, v_pts = self.containing_simplex_and_bcc(q_pts.T)
#         b_maxmin = bary_minmax(bcc)
#
#         v_pts_n = new_vpoints(v_pts, b_maxmin)
#         # bb = np.array(b_maxmin)
#         nCs = np.stack(Cs[b_maxmin[:,0], b_maxmin[:,1]], axis = 0)
#
#
#         v_pts_n = new_vpoints(v_pts, b_maxmin)
#         coeffs = np.array(self.coeffs)[:,trangs]
#
#         Cfs = [coeffs[nCs[:,0], range(len(nCs))], coeffs[nCs[:,1], range(len(nCs))], coeffs[nCs[:,2], range(len(nCs))],
#                 coeffs[nCs[:,3], range(len(nCs))], coeffs[nCs[:,4], range(len(nCs))], coeffs[nCs[:,5], range(len(nCs))]]
#
#         unos = np.ones([len(v_pts_n[:,0,0]),3])
#         xs = (unos*np.array([1,0,0]))
#         ys = (unos*np.array([0,1,0]))
#         zs = (unos*np.array([0,0,1]))
#
#         bcc_n = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts.T)
#
#         bcc_x = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], xs)
#         bcc_y = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], ys)
#         bcc_z = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], zs)
#         Bs = [bcc_x, bcc_y, bcc_z]
#
#         outs = grad_HBB(np.array(Cfs).T, bcc_n, Bs)
#
#
#         return cross(q_pts, outs)
#
#     def eval_grad(self, q_pts):
#
#         [phi,theta] = cart2sphere(q_pts)
#         N, M = len(phi), len(theta)
#         ijs = self.inds(phi,theta)
#         bcc, trangs, v_pts = self.containing_simplex_and_bcc_structured(ijs, q_pts.T)
#         # bcc, trangs, v_pts = self.containing_simplex_and_bcc(q_pts.T)
#         b_maxmin = bary_minmax(bcc)
#
#         v_pts_n = new_vpoints(v_pts, b_maxmin)
#         cfs_x = np.array(self.coeffs_x)[:,trangs]
#         cfs_y = np.array(self.coeffs_y)[:,trangs]
#         cfs_z = np.array(self.coeffs_z)[:,trangs]
#
#         return C1_PS_Split_grad_map(v_pts_n, b_maxmin, q_pts.T, [cfs_x, cfs_y, cfs_z])
#
#
#     def det_jacobian(self, q_pts, map = True):
#
#         [phi,theta] = cart2sphere(q_pts)
#         N, M = len(phi), len(theta)
#
#         ijs, coords = self.inds(phi,theta)
#
#         #phi = pth[0].copy().reshape([N*M,])
#         #theta =  pth[1].copy().reshape([N*M,])
#
#         #q_pts = np.array(sphere2cart(phi.reshape([N*M,]),theta.reshape([N*M,])))
#
#         # # For the endpoint:
#         # ijsp1 = [ijs[0] + 1, ijs[1] + 1]
#         # ijsp1[0] = ijsp1[0] % (len(self.phi))  #[where(ijsp1[0] == len(self.phi))] = 0
#         # ijsp1[1] = ijsp1[1] % (len(self.theta)) #[where(ijsp1[1] == len(self.phi))] = 0
#
#         bcc, trangs, v_pts = self.containing_simplex_and_bcc_structured(ijs, coords, q_pts.T)
#
#         b_maxmin = bary_minmax(bcc)
#
#         v_pts_n = new_vpoints(v_pts, b_maxmin)
#         cfs_x = np.array(self.coeffs_x)[:,trangs]
#         cfs_y = np.array(self.coeffs_y)[:,trangs]
#         cfs_z = np.array(self.coeffs_z)[:,trangs]
#
#         out = det_jac(v_pts_n, b_maxmin, q_pts, coeffs = [cfs_x, cfs_y, cfs_z])
#
#         return out
#
#     def stencil_eval(self, q_pts, st_pts = None, order = 2, map = True):
#         [phi,theta] = cart2sphere(q_pts)
#         N, M = len(phi), len(theta)
#
#         ijs = self.inds(phi,theta)
#
#         bcc, trangs, v_pts = self.containing_simplex_and_bcc_structured(ijs, q_pts.T)
#         # bcc, trangs, v_pts = self.containing_simplex_and_bcc(q_pts.T)
#
#         b_maxmin = bary_minmax(bcc)
#
#         # neg_inds = np.where(bcc < -0.0000001)
#         # if np.size(neg_inds) != 0:
#         #     # n_vs = v_pts[neg_inds[0]]
#         #
#         # #     pdb.set_trace()
#         #     # print(v_pts[neg_inds[0]])
#         #     print('no working', len(neg_inds[0]))
#
#         v_pts_n = new_vpoints(v_pts, b_maxmin)
#         cfs_x = np.array(self.coeffs_x)[:,trangs]
#         cfs_y = np.array(self.coeffs_y)[:,trangs]
#         cfs_z = np.array(self.coeffs_z)[:,trangs]
#
#         # in (x,y,z)
#         s_x = eval_stencils(v_pts_n, bmm = b_maxmin, st_pts = st_pts, coeffs = cfs_x)
#         s_y = eval_stencils(v_pts_n, bmm = b_maxmin, st_pts = st_pts, coeffs = cfs_y)
#         s_z = eval_stencils(v_pts_n, bmm = b_maxmin, st_pts = st_pts, coeffs = cfs_z)
#
#         return [s_x, s_y, s_z]
#
#     def Query(self, pts):
#         [phi,theta] = cart2sphere(q_pts)
#         N, M = len(phi), len(theta)
#         ijs = self.inds(phi,theta)
#         return self.containing_simplex_and_bcc_structured(ijs, q_pts.T)
#
#     def PS_split(self, v_pts, bcc):
#         #initialize empty array:
#         # vertices based on max barycentric coordinates
#         b_maxmin = bary_minmax(bcc)
#         v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
#         e1, e2, e3 = div_norm(v1/2+v2/2).T, div_norm(v2/2+v3/2).T, div_norm(v3/2 + v1/2).T
#
#         #edges = [[0,[2,1],[1,2]], [[1,0],0,[0,2]], [[1,0],[0,1],0]]
#         v4 = div_norm((v1 + v2 + v3)/3).T
#         #counterclockwise
#         edges = [[0,[v4,v2,e2],[e2,v3,v4]], [[v4,e3,v1],0,[e3,v4,v3]], [[v1,e1,v4],[v2,v4,e1],0]]
#         #clockwise
#         #edges = [[0,[v4,e2,v2],[e2,v4,v3]], [[v4,v1,e3],0,[e3,v3,v4]], [[v1,v4,e1],[v2,e1,v4],0]]
#
#
#         v_pts_n = np.empty(np.shape(v_pts))
#
#         #Then input new v_pts into CT
#         for i in range(len(b_maxmin)):  #figure out how to turn this into a slicing
#             e = b_maxmin[i]
#             js = edges[e[0]][e[1]]
#             v_pts_n[i,0,:] = js[0][i,:]
#             v_pts_n[i,1,:] = js[1][i,:]
#             v_pts_n[i,2,:] = js[2][i,:]
#
#         return v_pts_n
#
#
#     def C1_PS_split_grad(self, inds, bcc, q_pts, coeffs):
#         #initialize empty array:
#         # vertices based on max barycentric coordinates
#         v_pts = self.grid.points[inds]
#         b_maxmin = bary_minmax(bcc)
#         v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
#         e1, e2, e3 = div_norm(v1/2+v2/2).T, div_norm(v2/2+v3/2).T, div_norm(v3/2 + v1/2).T
#
#         #edges = [[0,[2,1],[1,2]], [[1,0],0,[0,2]], [[1,0],[0,1],0]]
#         v4 = div_norm((v1 + v2 + v3)/3).T
#
#         #counterclockwise-----------------------
#         edges = [[0,[v4,v2,e2],[e2,v3,v4]], [[v4,e3,v1],0,[e3,v4,v3]], [[v1,e1,v4],[v2,v4,e1],0]]
#
#         #Coefficient list
#         Cs = [[0, [18,1,13,7,6,16], [13,2,18,11,10,16]],
#               [[18,14,0,17,5,4],0,[14,18,2,17,10,9]],
#                [[0,12,18,3,15,4],[1,18,12,7,15,8],0]]
#         #clockwise------
#         # edges = [[0,[v4,e2,v2],[e2,v4,v3]], [[v4,v1,e3],0,[e3,v3,v4]], [[v1,v4,e1],[v2,e1,v4],0]]
#         #
#         # Cs = [[0, [18,13,1,16,6,7], [13,18,2,16,10,11]],
#         #       [[18,0,14,4,5,17],0,[14,2,18,9,10,17]],
#         #        [[0,18,12,4,15,3],[1,12,18,8,15,7],0]]
#         #-----------------
#         L_q = [[2,0,0],[0,2,0], [0,0,2], [1,1,0], [0,1,1], [1,0,1]]
#
#
#         v_pts_n = np.empty(np.shape(v_pts))
#         evals = []
#         #Then input new v_pts into CT
#         for i in range(len(b_maxmin)):  #figure out how to turn this into a slicing
#             e = b_maxmin[i]
#             js = edges[e[0]][e[1]]
#             v_pts_n[i,0,:] = js[0][i,:]
#             v_pts_n[i,1,:] = js[1][i,:]
#             v_pts_n[i,2,:] = js[2][i,:]
#             cc = Cs[e[0]][e[1]]
#
#             bcc_n = bary_coords(js[0][i,:], js[1][i,:],js[2][i,:],
#                                 q_pts[i,:], scl = True)
#             bcc_x = bary_coords(js[0][i,:], js[1][i,:],js[2][i,:],
#                                 np.array([1,0,0]), scl = True)
#             bcc_y = bary_coords(js[0][i,:], js[1][i,:],js[2][i,:],
#                                 np.array([0,1,0]), scl = True)
#             bcc_z = bary_coords(js[0][i,:], js[1][i,:],js[2][i,:],
#                                 np.array([0,0,1]), scl = True)
#
#             j = 0
#             temp_val = 0
#             for T in L_q:
#                 c_fact = factorial(sum(T))/(factorial(T[0])*factorial(T[1])*factorial(T[2]))
#                 temp_val += c_fact*coeffs[cc[j]][i]*(bcc_n[0]**T[0])*(bcc_n[1]**T[1])*(bcc_n[2]**T[2])
#                 j+=1
#             evals.append(temp_val)
#
#         return np.array(evals)
#
#     def CT_split(self, v_pts, bcc):
#         #initialize empty array:
#         # vertices based on max barycentric coordinates
#         b_min = bary_minmax(bcc)
#         v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
#
#         V = []
#         for i in range(len(b_min)):
#             indices = [0,1,2]
#             indices.remove(b_min[i])
#             V.append(v_pts[i, indices, :])
#         V = np.array(V)
#         v4 = div_norm((v1 + v2 + v3)/3)
#
#         #Then input new v_pts into CT
#
#         # watch out this might mess ordering
#         v_pts_n = np.empty(np.shape(v_pts))
#         # pdb.set_trace()
#         v_pts_n[:,0,:] = v4.T.copy()
#         v_pts_n[:,1,:] = V[:,0,:].copy()
#         v_pts_n[:,2,:] = V[:,1,:].copy()
#
#         return v_pts_n
#
#     def control_points(self, v_pts, b_min = None, which = "cubic"):
#
#         if which == "cubic":
#             #initialize empty array:
#             s = np.shape(v_pts)
#             out = np.zeros([s[0], 3, 10])
#             # vertices
#             # pdb.set_trace()
#             v1, v2, v3 = v_pts[:,0,:], v_pts[:,1,:], v_pts[:,2,:]
#             out[:,:,0], out[:,:,1], out[:,:,2] = v1.copy(), v2.copy(), v3.copy()
#             # then edges in order:
#             bary = (v1 + v2 + v3)/3
#
#             out[:,:,3] = div_norm((2*v1 + v2)/3).T
#             out[:,:,4] = div_norm((2*v1 + v3)/3).T
#             out[:,:,5] = div_norm((2*v2 + v3)/3).T
#             out[:,:,6] = div_norm((2*v2 + v1)/3).T
#
#
#             out[:,:,7] = div_norm((2*v3 + v1)/3).T
#             out[:,:,8] = div_norm((2*v3 + v2)/3).T
#
#             #then interior points in lexicographical ordering
#             out[:,:,9] = div_norm(bary).T
#
#         if which == "quadratic":
#             #initialize empty array:
#             s = np.shape(v_pts)
#             out = np.zeros([s[0], 3, 6])
#             # vertices
#             # pdb.set_trace()
#             v1, v2, v3 = v_pts[:,0,:], v_pts[:,1,:], v_pts[:,2,:]
#             out[:,:,0], out[:,:,1], out[:,:,2] = v1.copy(), v2.copy(), v3.copy()
#             # then edges in order:
#
#             out[:,:,3] = div_norm((v1 + v2)/2).T
#             out[:,:,4] = div_norm((v2 + v3)/2).T
#             out[:,:,5] = div_norm((v3 + v1)/2).T
#
#
#         return out
#
#     def bary_max(self, X):
#         return [np.where(x == max(x))[0][0] for x in X]
#     def bary_min(self, X):
#         return [np.where(x == min(x))[0][0] for x in X]
#
#     # def containing_simplex_and_bcc(self, pts):
#     #     """
#     #     Returns the simplices containing pts
#     #     and the local barycentric, normalised coordinates.
#     #
#     #     Args:
#     #         lons : float / array of floats, shape(l,)
#     #             longitudinal coordinates in radians
#     #         lats : float / array of floats, shape(l,)
#     #             latitudinal coordinates in radians
#     #
#     #     Returns:
#     #         bcc : normalised barycentric coordinates
#     #         tri : simplices containing pts
#     #
#     #     Notes:
#     #         That the ordering of the vertices may differ from
#     #         that stored in the self.simplices array but will
#     #         still be a loop around the simplex.
#     #     """
#     #     p = self.grid._permutation
#     #     sorted_simplices = np.sort(self.grid._simplices, axis=1)
#     #
#     #     # pdb.set_trace()
#     #     triangles = []
#     #     for pt in pts:
#     #         t = _stripack.trfind(3, pt, self.grid._x, self.grid._y, self.grid._z, self.grid.lst, self.grid.lptr, self.grid.lend )
#     #         tri = np.sort(t[3:6]) - 1
#     #
#     #         triangles.append(np.where(np.all(p[sorted_simplices]==p[tri], axis=1))[0])
#     #
#     #     tri_out = self.grid.simplices[np.array(triangles).reshape(-1)]
#     #     vs = self.grid.points[tri_out]
#     #
#     #     bcc = bary_coords(vs[:,0,:],vs[:,1,:], vs[:,2,:], pts)
#     #
#     #     #normalize the coordinates
#     #     #bcc /= bcc.sum(axis=1).reshape(-1,1)
#     #
#     #     return bcc, np.array(triangles).reshape(-1), vs
#
#     def containing_simplex_and_bcc_structured(self, inds, pts):
#         """
#         Returns the simplices containing pts
#         and the local barycentric, normalised coordinates.
#
#         Input:
#             inds: indices over meshgrid
#             coords: normalized coordinates in the meshgrid
#             pts: query points
#
#         Returns:
#             bcc : normalised barycentric coordinates
#             tri : simplices containing pts
#
#         Notes:
#
#         """
#
#         #obtain indices of which half of quadralateral
#         # s_inds = (np.heaviside(coords[1]-coords[0],0)).astype(int).reshape([len(pts),])
#         #s_inds = s_inds.reshape([len(pts),])
#         phi0, theta0 = self.phi[inds[0]], self.theta[inds[1]]
#         v_0 = sphere2cart(phi0, theta0)
#
#         # For the endpoint:
#         ijsp1 = [inds[0].copy() + 1, inds[1].copy() + 1]
#         ijsp1[0] = ijsp1[0] % (len(self.phi))
#
#         phi_p1, theta_p1 = self.phi[ijsp1[0]], self.theta[ijsp1[1]]
#
#         v_1 = np.array(sphere2cart(phi_p1, theta_p1))
#
#         n_vs = cross(v_0, v_1)
#         s_inds = np.heaviside(-dot(n_vs, pts.T), 0).astype(int)
#         # s_inds = np.heaviside(-det([v_0, v_1, pts.T]),1).astype(int)
#
# #       the indices swap since the lambda moves across columns,
# #       whereas theta increases along the rows
#         tri_out_temp = self.msimplices[inds[1], inds[0], :].reshape([len(pts),2])
#         # now assemble triangle list
#         l = np.shape(tri_out_temp)[0]
#         tri_out = np.array(list(tri_out_temp[np.arange(0,l),s_inds]))
#         trangs = tri_out[:,3]
#         verts = self.points[tri_out[:,0:3]]
#
#         bcc = bary_coords(verts[:,0,:], verts[:,1,:], verts[:,2,:], pts)
#
#         return bcc, trangs, verts
#
#
#
#     def inds(self, phiq, thetaq):
#         """
#         input a value (phi,theta) to interpolate
#         phi0, theta0 should be meshgrids with the same amount of points and
#         square.
#
#         phi_s0, theta_s0: Numpy array of 4 meshgrids of the stencils points
#         each grid represents one of the four corners.
#
#         output: list of indices, base points of position in the cell also
#         position of the stencil point relative to the cell
#         """
#
#         dphi = abs(self.phi[1]-self.phi[0])
#         dthe = abs(self.theta[1]-self.theta[0])
#         #Properly account for values outside of the interval
#         # #parse list a bit. --------------------------
#         # p_u, p_l = phiq[np.where(phiq >= pi)], phiq[np.where(phiq < -pi)]
#         # # #t_u, t_l = thetaq[np.where(thetaq >= pi)], thetaq[np.where(thetaq < -pi)]
#         # #
#         # p_u = (p_u + pi) % (2*pi) - pi
#         # # #t_u = (t_u + pi) % (2*pi) - pi
#         # p_l = (p_l + pi) % (2*pi) - pi
#         # # #t_l = (t_l + pi) % (2*pi) - pi
#         # #
#         # # # Make transformation
#         # phiq[np.where(phiq >= pi)] = p_u
#         # phiq[np.where(phiq < -pi)] = p_l
#         #thetaq[np.where(thetaq >= pi)], thetaq[np.where(thetaq < -pi)] = t_u, t_l
#         # #----------------------------------------------
#
#         phiq_n = phiq % (2*pi)
#
#         phi_l = ((phiq_n-self.phi[0])//dphi).astype(int) % (len(self.phi))
#         #don't mod this direction, just correct for boundary
#         theta_l = ((thetaq-self.theta[0])//dthe).astype(int)
#
#         theta_l[theta_l == (len(self.theta)-1)] = len(self.theta)-2
#         # #also compute position within the cell
#         phi_c = (phiq_n-self.phi[phi_l])/dphi
#         theta_c = (thetaq-self.theta[theta_l])/dthe
#
#         #need added check if query lands on theta_c = 0 or 1 lines.
#         #this is a tunable range
#         #only pertain to the interior points
#         inds0 = np.where((theta_c <= 0.03) & (theta_l != 0))
#         inds1 = np.where((theta_c > 0.85) & (theta_l !=  len(self.theta)-2))
#
#         phi_p1 = (phi_l + 1) % (len(self.phi))
#         the_p1 = theta_l + 1
#
#         #in or outside triangle for theta_c = 0.
#         v_0 = np.array(sphere2cart(self.phi[phi_l[inds0]], self.theta[theta_l[inds0]]))
#         v_1 = np.array(sphere2cart(self.phi[phi_p1[inds0]], self.theta[theta_l[inds0]]))
#
#         n_vs = cross(v_0, v_1)
#         q_pts0 = np.array(sphere2cart(phiq_n[inds0], thetaq[inds0]))
#         s_inds0 = np.heaviside(dot(n_vs, q_pts0), 0).astype(int)
#         # pdb.set_trace()
#         theta_l[inds0] = theta_l[inds0]-s_inds0
#         #in or outside triangle for theta_c = 0.
#         v_01 = np.array(sphere2cart(self.phi[phi_l[inds1]], self.theta[the_p1[inds1]]))
#         v_11 = np.array(sphere2cart(self.phi[phi_p1[inds1]], self.theta[the_p1[inds1]]))
#
#         n_vs2 = cross(v_01, v_11)
#         q_pts1 = np.array(sphere2cart(phiq_n[inds1], thetaq[inds1]))
#
#         s_inds1 = np.heaviside(-dot(n_vs2, q_pts1), 0).astype(int)
#
#         theta_l[inds1] = theta_l[inds1] + s_inds1
#
#
#         return [phi_l, theta_l] #, [phi_c, theta_c]
#
#
#
#     def stencil_pts(self, eps = 1e-5):
#         """
#
#         """
#         verts = self.points
#         # form direction vectors from every vertex
#         a1 = np.array(cross(verts.T,[0,0,1])).T
#         a2 = np.array(cross(verts.T, a1.T)).T
#
#         #replace first and last rows
#         a1[0,:] = cross(verts[0,:], [1/np.sqrt(2),1/np.sqrt(2),0])
#         a1[-1,:] = cross(verts[-1,:], [1/np.sqrt(2),1/np.sqrt(2),0])
#
#         a2[0,:] = cross(verts[0,:], a1[0,:])
#         a2[-1,:] = cross(verts[-1,:], a1[-1,:])
#
#         #normalize the direction vectors
#         a1_n, a2_n = div_norm(a1).T, div_norm(a2).T
#
#         # pdb.set_trace()
#         #define the stencil points
#         # 4 - point averaged
#         eps_1p = verts - eps*a1_n - eps*a2_n
#         eps_1m = verts + a1_n*eps - eps*a2_n
#         eps_2p = verts - eps*a1_n + eps*a2_n
#         eps_2m = verts + a1_n*eps + a2_n*eps
#
#         #along coordinate direction
#         # eps_1p = verts + eps*a1_n
#         # eps_1m = verts - a1_n*eps
#         # eps_2p = verts + eps*a2_n
#         # eps_2m = verts - a2_n*eps
#
#         # # #normalize once again and output
#         # dirs = [div_norm(eps_1p), div_norm(eps_1m), div_norm(eps_2p),
#         #         div_norm(eps_2m)]
#
#         dirs = [g_proj(eps_1p, verts), g_proj(eps_1m, verts), g_proj(eps_2p, verts),
#                 g_proj(eps_2m, verts)]
#
#         #save directions for later use
#         self.tan_vects = [a1_n, a2_n]
#         # # 6-point stencils implementation
#         # v1 = np.array([verts[:,0] + eps, verts[:,1], verts[:,2]]).T
#         # v2 = np.array([verts[:,0] - eps, verts[:,1], verts[:,2]]).T
#         #
#         # v3 = np.array([verts[:,0], verts[:,1] + eps, verts[:,2]]).T
#         # v4 = np.array([verts[:,0], verts[:,1] - eps, verts[:,2]]).T
#         #
#         # v5 = np.array([verts[:,0], verts[:,1], verts[:,2] + eps]).T
#         # v6 = np.array([verts[:,0], verts[:,1], verts[:,2] - eps]).T
#         #
#         #
#         # dirs = [div_norm(v1), div_norm(v2), div_norm(v3), div_norm(v4),
#         #           div_norm(v5), div_norm(v6)]
#         # #dirs = [v1.T, v2.T, v3.T, v4.T, v5.T, v6.T]
#
#         return dirs
#
#     def assemble_coefficients(self, inds, points, vals, grad_vals):
#         """
#         Void function to assemble all the coefficients to perform the PS split
#         interpolation
#         """
#         #inds = np.array(self.simplices)
#         # All values that don't need to be recomputed:
#         v_pts = points[inds]
#         v1r, v2r, v3r = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
#         v4r = div_norm((v1r + v2r + v3r)/3).T
#         e1r, e2r, e3r = div_norm(v1r/2+v2r/2).T, div_norm(v2r/2+v3r/2).T, div_norm(v3r/2 + v1r/2).T
#         # Calculate barycentric coords of the edges
#         # h12r, h21r, h23r, h32r, h13r, h31r = h(e1r,v1r).T, h(e1r,v2r).T, h(e2r,v2r).T, h(e2r,v3r).T, h(e3r,v1r).T, h(e3r,v3r).T
#         #tangential projection
#         h12r, h21r, h23r = h(v2r-v1r,v1r).T, h(v1r-v2r,v2r).T, h(v3r-v2r,v2r).T
#         h32r, h13r, h31r = h(v2r-v3r,v3r).T, h(v3r-v1r,v1r).T, h(v1r-v3r,v3r).T
#         h41, h42, h43 = h(v4r-v1r,v1r).T, h(v4r-v2r,v2r).T, h(v4r-v3r,v3r).T
#
#         g12r, g21r = bary_coords(v1r,e1r,v4r,h12r), bary_coords(v2r,v4r,e1r,h21r)
#         g23r, g32r = bary_coords(v2r,e2r,v4r,h23r), bary_coords(v3r,v4r,e2r,h32r)
#         g13r, g31r = bary_coords(v1r,v4r,e3r,h13r), bary_coords(v3r,e3r,v4r,h31r)
#
#         g14r = bary_coords(v1r,v4r,e3r,h(v4r,v1r).T)
#         g24r = bary_coords(v2r,v4r,e1r,h(v4r,v2r).T)
#         g34r = bary_coords(v3r,v4r,e2r,h(v4r,v3r).T)
#
#         Ar = bary_coords(v1r,v2r,v3r,v4r)
#
#         #assemble into nice lists
#         verts = [v1r,v2r,v3r,v4r]
#         Hs = [h12r, h21r, h23r, h32r, h13r, h31r]
#         Gs = [g12r, g21r, g23r, g32r, g13r, g31r, g14r, g24r, g34r, Ar]
#         g_1r, g_2r, g_3r = bary_coords(v1r,v2r,v3r,e1r), bary_coords(v1r,v2r,v3r,e2r), bary_coords(v1r,v2r,v3r,e3r)
#         gs = [g_1r, g_2r, g_3r]
#
#         # now the non-recyclable quantities
#         #in x--------
#         grad_f1 = np.array([grad_vals[0][inds[:,0]], grad_vals[1][inds[:,0]], grad_vals[2][inds[:,0]]])
#         grad_f2 = np.array([grad_vals[0][inds[:,1]], grad_vals[1][inds[:,1]], grad_vals[2][inds[:,1]]])
#         grad_f3 = np.array([grad_vals[0][inds[:,2]], grad_vals[1][inds[:,2]], grad_vals[2][inds[:,2]]])
#         grad_fx = [grad_f1, grad_f2, grad_f3]
#         c123x = vals[inds]
#         coeffs_x = PS_split_coeffs(verts = verts, Hs = Hs, Gs = Gs, gs = gs,
#                                         c123 =  c123x, grad_f = grad_fx)
#
#         return coeffs_x

#===============================================================================

# This is not working properly yet -----

# interpolation on the Clough-Tocher split

# def C1_CT_Split(v_pts_n, bmm, q_pts, coeffs):
#     #counterclockwise-----------------------
#     # for reference edges = [[v3,v4,v2],[v4,v3,v1],[v1,v2,v4]]
#
#     # #Coefficient list
#     Cs = np.array([[3,19,2,11,18,17,8,7,12,14],
#                    [19,3,1,18,11,10,6,5,16,15],
#                    [1,2,19,4,9,8,17,16,5,13]]) -1
#
#     bb = np.array(bmm)
#     nCs = np.stack(Cs[bb], axis = 0)
#
#
#     L_c = [[3,0,0], [0,3,0], [0,0,3], [2,1,0], [1,2,0], [0,2,1], [0,1,2], [1,0,2],
#            [2,0,1],[1,1,1]]
#
#     #cfact = factorial(sum(T))/(factorial(T[0])*factorial(T[1])*factorial(T[2]))
#     c_facts = [1, 1, 1, 3, 3, 3, 3, 3, 3, 6]
#
#     bcc_n = bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts)
#
#     Cfs = [coeffs[nCs[:,0], range(len(nCs))], coeffs[nCs[:,1], range(len(nCs))], coeffs[nCs[:,2], range(len(nCs))],
#             coeffs[nCs[:,3], range(len(nCs))], coeffs[nCs[:,4], range(len(nCs))], coeffs[nCs[:,5], range(len(nCs))],
#             coeffs[nCs[:,6], range(len(nCs))], coeffs[nCs[:,7], range(len(nCs))], coeffs[nCs[:,8], range(len(nCs))],
#             coeffs[nCs[:,9], range(len(nCs))]]
#
#     evals = c_facts[0]*coeffs[0,:]*(bcc_n[:,0]**L_c[0][0])*(bcc_n[:,1]**L_c[0][1])*(bcc_n[:,2]**L_c[0][2])
#     j = 1
#     for T in L_c[1:]:
#         evals += c_facts[j]*coeffs[j,:]*(bcc_n[:,0]**T[0])*(bcc_n[:,1]**T[1])*(bcc_n[:,2]**T[2])
#         j+=1
#
#     return np.array(evals)
