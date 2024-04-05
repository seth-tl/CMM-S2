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
from . import mesh_functions as meshes
from . import utils
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#===============================================================================
# ==================================================================
# Spherical diffeomorphism interpolation classes:

class spherical_spline(object):

    """
    Basic class for spherical spline interpolation of a function on the sphere using 
    the approximation space S^1_2(T_{PS})

    inputs:
        mesh: spherical_triangulation object from mesh_functions.py
        vals: (4,N) array defining the jet (f, Df) at the grid points
        coeffs: (19,N) array defining the pre-allocated array of coefficients 
    """

    def __init__(self, mesh, vals, coeffs):

        self.mesh = mesh
        self.vals = vals

        # precompute all the coefficients defining the interpolant
        self.coeffs = self.assemble_coefficients(coeffs)
        self.vort = False
        return


    def __call__(self, phi,the):
       
       # to be incorporatedn with the barotropic vorticity solver: 
        ss = np.shape(phi)
        q_pts = np.array(utils.sphere2cart(phi.reshape([ss[0]*ss[1]]),the.reshape(ss[0]*ss[1]))).T
        # compute barycentric coordinates and containing triangle
        bcc, trangs, v_pts = self.mesh.query(q_pts)
        # # additional operation to find split triangle
        bb = bary_minmax(bcc)
        nCs = Cs[bb[0], bb[1]]

        # vertices of split triangle
        v_pts_n = new_vpoints(v_pts, bb) # TODO: get rid of this operation
        # get appropriate coefficients and recompute barycentric coordinates
        cfs =  self.coeffs[:,trangs]

        bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts)
        # evaluate the quadratic Berstein-Bezier polynomial in each split triangle
        inds = range(len(nCs))

        outs = (bcc_n[:,0]**2)*cfs[nCs[:,0], inds] + \
               (bcc_n[:,1]**2)*cfs[nCs[:,1], inds] + \
               (bcc_n[:,2]**2)*cfs[nCs[:,2], inds] + \
             2*(bcc_n[:,1]*bcc_n[:,0])*cfs[nCs[:,3], inds] +\
             2*(bcc_n[:,2]*bcc_n[:,1])*cfs[nCs[:,4], inds] +\
             2*(bcc_n[:,2]*bcc_n[:,0])*cfs[nCs[:,5], inds]
        

        return outs.reshape([ss[0], ss[1]])


    def gradient(self, q_pts):
        # Computes the differential of a map D\varphi_x : T_x M ---> T_{\varphi(x)}M
        # x = q_pts, varphi(x) = eval_pts
        
        bcc, trangs, v_pts = self.mesh.query(q_pts)

        # # additional operation to find split triangle
        bb = bary_minmax(bcc)
        nCs = Cs[bb[0], bb[1]]

        # vertices of split triangle
        v_pts_n = new_vpoints(v_pts, bb) # TODO: get rid of this operation

        # get appropriate coefficients and recompute barycentric coordinates
        cfs =  self.coeffs[:,trangs]

        unos = np.ones([len(v_pts_n[:,0,0]),3])
        xs = (unos*np.array([1,0,0]))
        ys = (unos*np.array([0,1,0]))
        zs = (unos*np.array([0,0,1]))

        bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts)
        bcc_x = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], xs)
        bcc_y = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], ys)
        bcc_z = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], zs)

        inds = range(len(nCs))

        dp_b1 = 2*cfs[nCs[:,0], inds]*bcc_n[:,0] + 2*cfs[nCs[:,3], inds]*bcc_n[:,1] + 2*cfs[nCs[:,5], inds]*bcc_n[:,2]

        dp_b2 = 2*cfs[nCs[:,1], inds]*bcc_n[:,1] + 2*cfs[nCs[:,3], inds]*bcc_n[:,0] + 2*cfs[nCs[:,4], inds]*bcc_n[:,2]

        dp_b3 = 2*cfs[nCs[:,2], inds]*bcc_n[:,2] + 2*cfs[nCs[:,4], inds]*bcc_n[:,1] + 2*cfs[nCs[:,5], inds]*bcc_n[:,0]

        outx = bcc_x[:,0]*dp_b1 +  bcc_x[:,1]*dp_b2 + bcc_x[:,2]*dp_b3
        outy = bcc_y[:,0]*dp_b1 +  bcc_y[:,1]*dp_b2 + bcc_y[:,2]*dp_b3
        outz = bcc_z[:,0]*dp_b1 +  bcc_z[:,1]*dp_b2 + bcc_z[:,2]*dp_b3

        return d_Proj(q_pts, [outx, outy, outz])



    def assemble_coefficients(self, coeffs):
        """
        assemble all the coefficients to perform the PS split interpolation.
        """
        # void function replaces the coefficients of the array:

        tri_vals = self.vals[:,np.array(self.mesh.simplices)]
        ps_split = self.mesh.ps_split


        coeffs[0,:] =  tri_vals[0,:,0] 
        coeffs[1,:] =  tri_vals[0,:,1]
        coeffs[2,:] =  tri_vals[0,:,2]

        coeffs[3,:] = (1/ps_split[1][0][:,1])*(utils.dot(ps_split[0][0], tri_vals[1::,:,0])/2 - (ps_split[1][0][:,0])*tri_vals[0,:,0])
        coeffs[4,:] = (1/ps_split[1][6][:,1])*(utils.dot(ps_split[0][6], tri_vals[1::,:,0])/2 - (ps_split[1][6][:,0])*tri_vals[0,:,0])
        coeffs[5,:] = (1/ps_split[1][4][:,2])*(utils.dot(ps_split[0][4], tri_vals[1::,:,0])/2 - (ps_split[1][4][:,0])*tri_vals[0,:,0])

        coeffs[6,:] = (1/ps_split[1][2][:,1])*(utils.dot(ps_split[0][2], tri_vals[1::,:,1])/2 - (ps_split[1][2][:,0])*tri_vals[0,:,1])
        coeffs[7,:] = (1/ps_split[1][7][:,1])*(utils.dot(ps_split[0][7], tri_vals[1::,:,1])/2 - (ps_split[1][7][:,0])*tri_vals[0,:,1])
        coeffs[8,:] = (1/ps_split[1][1][:,2])*(utils.dot(ps_split[0][1], tri_vals[1::,:,1])/2 - (ps_split[1][1][:,0])*tri_vals[0,:,1])

        coeffs[9,:] = (1/ps_split[1][5][:,1])*(utils.dot(ps_split[0][5], tri_vals[1::,:,2])/2 - (ps_split[1][5][:,0])*tri_vals[0,:,2])
        coeffs[10,:] = (1/ps_split[1][8][:,1])*(utils.dot(ps_split[0][8], tri_vals[1::,:,2])/2 - (ps_split[1][8][:,0])*tri_vals[0,:,2])
        coeffs[11,:] = (1/ps_split[1][3][:,2])*(utils.dot(ps_split[0][3], tri_vals[1::,:,2])/2 - (ps_split[1][3][:,0])*tri_vals[0,:,2])


        coeffs[12,:] = (ps_split[2][0][:,0])*coeffs[3,:] + (ps_split[2][0][:,1])*coeffs[8,:]
        coeffs[13,:] = (ps_split[2][1][:,1])*coeffs[6,:] + (ps_split[2][1][:,2])*coeffs[11,:]
        coeffs[14,:] = (ps_split[2][2][:,2])*coeffs[9,:] + (ps_split[2][2][:,0])*coeffs[5,:]
        coeffs[15,:] = (ps_split[2][0][:,0])*coeffs[4,:] + (ps_split[2][0][:,1])*coeffs[7,:]
        coeffs[16,:] = (ps_split[2][1][:,1])*coeffs[7,:] + (ps_split[2][1][:,2])*coeffs[10,:]
        coeffs[17,:] = (ps_split[2][2][:,0])*coeffs[4,:] + (ps_split[2][2][:,2])*coeffs[10,:]

        #barycentre coords of middle points:
        coeffs[18,:] = (ps_split[1][9][:,0])*coeffs[4,:] + (ps_split[1][9][:,1])*coeffs[7,:] + (ps_split[1][9][:,2])*coeffs[10,:]

        return coeffs


class sphere_diffeomorphism(object):

    """
    Basic class for spherical spline interpolation defining a diffeomorphism of the sphere.
    Interpolation is defined by the Powell-Sabin split.

    inputs:
        mesh: spherical_triangulation object from mesh_functions.py
        vals: (3,4,N) array defining the jet (\phi, D\varphi) at the grid points
        coeffs: (3,19,N) array defining the pre-allocated array of coefficients 
    """

    def __init__(self, mesh, vals, coeffs):

        self.mesh = mesh
        self.vals = vals

        # precompute all the coefficients defining the interpolant
        self.coeffs = self.assemble_coefficients(coeffs)

        return


    def __call__(self, q_pts):
        
        # compute barycentric coordinates and containing triangle
        bcc, trangs, v_pts = self.mesh.query(q_pts)
 
        # # additional operation to find split triangle
        bb = bary_minmax(bcc)
        nCs = Cs[bb[0], bb[1]]

        # vertices of split triangle
        v_pts_n = new_vpoints(v_pts, bb) # TODO: get rid of this operation

        # get appropriate coefficients and recompute barycentric coordinates
        cfs =  self.coeffs[:,:,trangs]
        bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts)
        # evaluate the quadratic Berstein-Bezier polynomial in each split triangle
        inds = range(len(nCs))

        outs = (bcc_n[:,0]**2)[None,:]*cfs[:,nCs[:,0], inds] + \
               (bcc_n[:,1]**2)[None,:]*cfs[:,nCs[:,1], inds] + \
               (bcc_n[:,2]**2)[None,:]*cfs[:,nCs[:,2], inds] + \
             2*(bcc_n[:,1]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,3], inds] +\
             2*(bcc_n[:,2]*bcc_n[:,1])[None,:]*cfs[:,nCs[:,4], inds] +\
             2*(bcc_n[:,2]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,5], inds]
        
        norm = np.sqrt(outs[0,:]**2 + outs[1,:]**2 + outs[2,:]**2)

        return (1/norm)[None,:]*outs


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


    def stencil_eval(self, q_pts, st_pts):
        # queries at q_pts and evaluates at st_pts.
        # reduces the computational cost of querying at the expense of
        # performing a small extrapolation for the stencil points

        bcc, trangs, v_pts = self.mesh.query(q_pts.T)
        bmm = bary_minmax(bcc)

        v_pts_n = new_vpoints(v_pts, bmm) # TODO: get rid of this operation
        cfs = self.coeffs[:,:,trangs]

        nCs = Cs[bmm[0], bmm[1]]

        inds = range(len(nCs))
        st_pts_n = st_pts[:]

        for i in range(4):

            bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], st_pts[:,i,:].T)

            # evaluate the quadratic Berstein-Bezier polynomial in each split trianglea
            outs = (bcc_n[:,0]**2)[None,:]*cfs[:,nCs[:,0], inds] + \
                (bcc_n[:,1]**2)[None,:]*cfs[:,nCs[:,1], inds] + \
                (bcc_n[:,2]**2)[None,:]*cfs[:,nCs[:,2], inds] + \
                2*(bcc_n[:,1]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,3], inds] +\
                2*(bcc_n[:,2]*bcc_n[:,1])[None,:]*cfs[:,nCs[:,4], inds] +\
                2*(bcc_n[:,2]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,5], inds]

            st_pts_n[:,i,:] = outs

        return st_pts_n

    def assemble_coefficients(self, coeffs):
        """
        assemble all the coefficients to perform the PS split interpolation.
        """
        # void function replaces the coefficients of the array:

        tri_vals = self.vals[:,:,np.array(self.mesh.simplices)]
        ps_split = self.mesh.ps_split


        coeffs[:,0,:] =  tri_vals[:,0,:,0] 
        coeffs[:,1,:] =  tri_vals[:,0,:,1]
        coeffs[:,2,:] =  tri_vals[:,0,:,2]

        coeffs[:,3,:] = (1/ps_split[1][0][:,1])[None,:]*(utils.dot(ps_split[0][0], tri_vals[:,1::,:,0])/2 - (ps_split[1][0][:,0])[None,:]*tri_vals[:,0,:,0])
        coeffs[:,4,:] = (1/ps_split[1][6][:,1])[None,:]*(utils.dot(ps_split[0][6], tri_vals[:,1::,:,0])/2 - (ps_split[1][6][:,0])[None,:]*tri_vals[:,0,:,0])
        coeffs[:,5,:] = (1/ps_split[1][4][:,2])[None,:]*(utils.dot(ps_split[0][4], tri_vals[:,1::,:,0])/2 - (ps_split[1][4][:,0])[None,:]*tri_vals[:,0,:,0])

        coeffs[:,6,:] = (1/ps_split[1][2][:,1])[None,:]*(utils.dot(ps_split[0][2], tri_vals[:,1::,:,1])/2 - (ps_split[1][2][:,0])[None,:]*tri_vals[:,0,:,1])
        coeffs[:,7,:] = (1/ps_split[1][7][:,1])[None,:]*(utils.dot(ps_split[0][7], tri_vals[:,1::,:,1])/2 - (ps_split[1][7][:,0])[None,:]*tri_vals[:,0,:,1])
        coeffs[:,8,:] = (1/ps_split[1][1][:,2])[None,:]*(utils.dot(ps_split[0][1], tri_vals[:,1::,:,1])/2 - (ps_split[1][1][:,0])[None,:]*tri_vals[:,0,:,1])

        coeffs[:,9,:] = (1/ps_split[1][5][:,1])[None,:]*(utils.dot(ps_split[0][5], tri_vals[:,1::,:,2])/2 - (ps_split[1][5][:,0])[None,:]*tri_vals[:,0,:,2])
        coeffs[:,10,:] = (1/ps_split[1][8][:,1])[None,:]*(utils.dot(ps_split[0][8], tri_vals[:,1::,:,2])/2 - (ps_split[1][8][:,0])[None,:]*tri_vals[:,0,:,2])
        coeffs[:,11,:] = (1/ps_split[1][3][:,2])[None,:]*(utils.dot(ps_split[0][3], tri_vals[:,1::,:,2])/2 - (ps_split[1][3][:,0])[None,:]*tri_vals[:,0,:,2])


        coeffs[:,12,:] = (ps_split[2][0][:,0])[None,:]*coeffs[:,3,:] + (ps_split[2][0][:,1])[None,:]*coeffs[:,8,:]
        coeffs[:,13,:] = (ps_split[2][1][:,1])[None,:]*coeffs[:,6,:] + (ps_split[2][1][:,2])[None,:]*coeffs[:,11,:]
        coeffs[:,14,:] = (ps_split[2][2][:,2])[None,:]*coeffs[:,9,:] + (ps_split[2][2][:,0])[None,:]*coeffs[:,5,:]
        coeffs[:,15,:] = (ps_split[2][0][:,0])[None,:]*coeffs[:,4,:] + (ps_split[2][0][:,1])[None,:]*coeffs[:,7,:]
        coeffs[:,16,:] = (ps_split[2][1][:,1])[None,:]*coeffs[:,7,:] + (ps_split[2][1][:,2])[None,:]*coeffs[:,10,:]
        coeffs[:,17,:] = (ps_split[2][2][:,0])[None,:]*coeffs[:,4,:] + (ps_split[2][2][:,2])[None,:]*coeffs[:,10,:]

        #barycentre coords of middle points:
        coeffs[:,18,:] = (ps_split[1][9][:,0])[None,:]*coeffs[:,4,:] + (ps_split[1][9][:,1])[None,:]*coeffs[:,7,:] + (ps_split[1][9][:,2])[None,:]*coeffs[:,10,:]

        return coeffs


class composite_sphere_diffeomorphism(object):

    """
    Basic class for composite splines interpolation

    same parameters as the sphere_diffeomorphism class
    with additional specification for the number of maps in the composition

    inputs:
        mesh: spherical_triangulation object from mesh_functions.py
        vals: (n_maps,3,4,N) array defining the jet (\phi, D\varphi) at the grid points
        coeffs: (n_maps,3,19,N) array defining the pre-allocated array of coefficients 
        ns: (n_maps,) array of time points when remapping is performed
    """

    def __init__(self, mesh, vals, coeffs, ns):

        self.mesh = mesh
        self.vals = vals
        self.ns = ns
        self.coeffs = coeffs # don't intialize any coefficients though
        self.Nc = 0 # counter for the number of maps that have been filled

        return

    def __call__(self, q_pts, N_c = None):
        
        q_pts_n = q_pts.copy()
        if N_c == None:
            N_c = self.Nc
        
        for i in reversed(range(N_c)):
            # compute barycentric coordinates and containing triangle
            bcc, trangs, v_pts = self.mesh.query(q_pts_n)
    
            # # additional operation to find split triangle
            bb = bary_minmax(bcc)
            nCs = Cs[bb[0], bb[1]]

            # vertices of split triangle
            v_pts_n = new_vpoints(v_pts, bb) # TODO: get rid of this operation
            # get appropriate coefficients and recompute barycentric coordinates
            cfs =  self.coeffs[i][:,:,trangs]
            bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts_n)

            # evaluate the quadratic Berstein-Bezier polynomial in each split triangle
            inds = range(len(nCs))

            outs = (bcc_n[:,0]**2)[None,:]*cfs[:,nCs[:,0], inds] + \
                   (bcc_n[:,1]**2)[None,:]*cfs[:,nCs[:,1], inds] + \
                   (bcc_n[:,2]**2)[None,:]*cfs[:,nCs[:,2], inds] + \
                 2*(bcc_n[:,1]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,3], inds] +\
                 2*(bcc_n[:,2]*bcc_n[:,1])[None,:]*cfs[:,nCs[:,4], inds] +\
                 2*(bcc_n[:,2]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,5], inds]
            
            norm = np.sqrt(outs[0,:]**2 + outs[1,:]**2 + outs[2,:]**2)
            q_pts_n = (((1/norm)[None,:]*outs)).T

        return q_pts_n

    def eval_all(self, q_pts):
        # same as call except it outputs all the intermediate steps:
                
        q_pts_n = q_pts.copy()
        eval_outs = [q_pts.copy() for _ in range(self.Nc+1)]
        j = 1
        for i in reversed(range(self.Nc)):
            # compute barycentric coordinates and containing triangle
            bcc, trangs, v_pts = self.mesh.query(q_pts_n)
    
            # # additional operation to find split triangle
            bb = bary_minmax(bcc)
            nCs = Cs[bb[0], bb[1]]

            # vertices of split triangle
            v_pts_n = new_vpoints(v_pts, bb) # TODO: get rid of this operation
            # get appropriate coefficients and recompute barycentric coordinates
            cfs =  self.coeffs[i][:,:,trangs]
            bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts_n)

            # evaluate the quadratic Berstein-Bezier polynomial in each split triangle
            inds = range(len(nCs))

            outs = (bcc_n[:,0]**2)[None,:]*cfs[:,nCs[:,0], inds] + \
                   (bcc_n[:,1]**2)[None,:]*cfs[:,nCs[:,1], inds] + \
                   (bcc_n[:,2]**2)[None,:]*cfs[:,nCs[:,2], inds] + \
                 2*(bcc_n[:,1]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,3], inds] +\
                 2*(bcc_n[:,2]*bcc_n[:,1])[None,:]*cfs[:,nCs[:,4], inds] +\
                 2*(bcc_n[:,2]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,5], inds]
            
            norm = np.sqrt(outs[0,:]**2 + outs[1,:]**2 + outs[2,:]**2)
            q_pts_n = (((1/norm)[None,:]*outs)).T
            
            eval_outs[j] = q_pts_n 

            j +=1

        return eval_outs

    def stencil_eval(self, q_pts, st_pts):
        # queries at q_pts and evaluates at st_pts.
        # reduces the computational cost of querying at the expense of
        # performing a small extrapolation for the stencil points

        bcc, trangs, v_pts = self.mesh.query(q_pts.T)
        bmm = bary_minmax(bcc)

        v_pts_n = new_vpoints(v_pts, bmm) # TODO: get rid of this operation
        cfs = self.coeffs[:,:,trangs]

        nCs = Cs[bmm[0], bmm[1]]

        inds = range(len(nCs))
        st_pts_n = st_pts[:]

        for i in range(4):

            bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], st_pts[:,i,:].T)

            # evaluate the quadratic Berstein-Bezier polynomial in each split trianglea
            outs = (bcc_n[:,0]**2)[None,:]*cfs[:,nCs[:,0], inds] + \
                (bcc_n[:,1]**2)[None,:]*cfs[:,nCs[:,1], inds] + \
                (bcc_n[:,2]**2)[None,:]*cfs[:,nCs[:,2], inds] + \
                2*(bcc_n[:,1]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,3], inds] +\
                2*(bcc_n[:,2]*bcc_n[:,1])[None,:]*cfs[:,nCs[:,4], inds] +\
                2*(bcc_n[:,2]*bcc_n[:,0])[None,:]*cfs[:,nCs[:,5], inds]

            st_pts_n[:,i,:] = outs

        return st_pts_n

    def assemble_coefficients(self, i, vals):
        """
        assemble all the coefficients to perform the PS split interpolation.
        """
        # void function replaces the coefficients of the array:
        self.vals[i] = vals
        tri_vals = self.vals[i][:,:,np.array(self.mesh.simplices)]
        ps_split = self.mesh.ps_split


        self.coeffs[i][:,0,:] =  tri_vals[:,0,:,0] 
        self.coeffs[i][:,1,:] =  tri_vals[:,0,:,1]
        self.coeffs[i][:,2,:] =  tri_vals[:,0,:,2]

        self.coeffs[i][:,3,:] = (1/ps_split[1][0][:,1])[None,:]*(utils.dot(ps_split[0][0], tri_vals[:,1::,:,0])/2 - (ps_split[1][0][:,0])[None,:]*tri_vals[:,0,:,0])
        self.coeffs[i][:,4,:] = (1/ps_split[1][6][:,1])[None,:]*(utils.dot(ps_split[0][6], tri_vals[:,1::,:,0])/2 - (ps_split[1][6][:,0])[None,:]*tri_vals[:,0,:,0])
        self.coeffs[i][:,5,:] = (1/ps_split[1][4][:,2])[None,:]*(utils.dot(ps_split[0][4], tri_vals[:,1::,:,0])/2 - (ps_split[1][4][:,0])[None,:]*tri_vals[:,0,:,0])

        self.coeffs[i][:,6,:] = (1/ps_split[1][2][:,1])[None,:]*(utils.dot(ps_split[0][2], tri_vals[:,1::,:,1])/2 - (ps_split[1][2][:,0])[None,:]*tri_vals[:,0,:,1])
        self.coeffs[i][:,7,:] = (1/ps_split[1][7][:,1])[None,:]*(utils.dot(ps_split[0][7], tri_vals[:,1::,:,1])/2 - (ps_split[1][7][:,0])[None,:]*tri_vals[:,0,:,1])
        self.coeffs[i][:,8,:] = (1/ps_split[1][1][:,2])[None,:]*(utils.dot(ps_split[0][1], tri_vals[:,1::,:,1])/2 - (ps_split[1][1][:,0])[None,:]*tri_vals[:,0,:,1])

        self.coeffs[i][:,9,:] = (1/ps_split[1][5][:,1])[None,:]*(utils.dot(ps_split[0][5], tri_vals[:,1::,:,2])/2 - (ps_split[1][5][:,0])[None,:]*tri_vals[:,0,:,2])
        self.coeffs[i][:,10,:] = (1/ps_split[1][8][:,1])[None,:]*(utils.dot(ps_split[0][8], tri_vals[:,1::,:,2])/2 - (ps_split[1][8][:,0])[None,:]*tri_vals[:,0,:,2])
        self.coeffs[i][:,11,:] = (1/ps_split[1][3][:,2])[None,:]*(utils.dot(ps_split[0][3], tri_vals[:,1::,:,2])/2 - (ps_split[1][3][:,0])[None,:]*tri_vals[:,0,:,2])


        self.coeffs[i][:,12,:] = (ps_split[2][0][:,0])[None,:]*self.coeffs[i][:,3,:] + (ps_split[2][0][:,1])[None,:]*self.coeffs[i][:,8,:]
        self.coeffs[i][:,13,:] = (ps_split[2][1][:,1])[None,:]*self.coeffs[i][:,6,:] + (ps_split[2][1][:,2])[None,:]*self.coeffs[i][:,11,:]
        self.coeffs[i][:,14,:] = (ps_split[2][2][:,2])[None,:]*self.coeffs[i][:,9,:] + (ps_split[2][2][:,0])[None,:]*self.coeffs[i][:,5,:]
        self.coeffs[i][:,15,:] = (ps_split[2][0][:,0])[None,:]*self.coeffs[i][:,4,:] + (ps_split[2][0][:,1])[None,:]*self.coeffs[i][:,7,:]
        self.coeffs[i][:,16,:] = (ps_split[2][1][:,1])[None,:]*self.coeffs[i][:,7,:] + (ps_split[2][1][:,2])[None,:]*self.coeffs[i][:,10,:]
        self.coeffs[i][:,17,:] = (ps_split[2][2][:,0])[None,:]*self.coeffs[i][:,4,:] + (ps_split[2][2][:,2])[None,:]*self.coeffs[i][:,10,:]

        #barycentre coords of middle points:
        self.coeffs[i][:,18,:] = (ps_split[1][9][:,0])[None,:]*self.coeffs[i][:,4,:] + (ps_split[1][9][:,1])[None,:]*self.coeffs[i][:,7,:] + (ps_split[1][9][:,2])[None,:]*self.coeffs[i][:,10,:]

        return 

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

# interpolation classes for the velocity field ===================================================================
    
# Vector field interpolation class used during the barotropic vorticity simulations
    
class spline_interp_velocity(object):
    """
    Basic interpolation class for the velocity field.
    Extends to a 2 + 1D interpolant using Lagrange interpolation in time
    
    Inputs: same of sphere_diffeomorphisms class with the additional
            ts value which defines the time grid.    
    """

    def __init__(self, mesh, vals, coeffs, ts):

        self.mesh = mesh
        self.vals = vals
        self.ts = ts
        self.coeffs = coeffs
        self.stepping = False
        self.nVs = 1

        return

    def init_coeffs(self, i):
        # intialize the ith coefficient array:
            self.coeffs[i] = self.assemble_coefficients(self.coeffs[i], i)
            return

    def __call__(self, t, dt, q_pts):
        
        # compute barycentric coordinates and containing triangle
        bcc, trangs, v_pts = self.mesh.query(q_pts.T)
 
        # # additional operation to find split triangle
        bb = bary_minmax(bcc)
        nCs = Cs[bb[0], bb[1]]

        # vertices of split triangle
        v_pts_n = new_vpoints(v_pts, bb) # TODO: get rid of this operation
        # get appropriate coefficients and recompute barycentric coordinates
        bcc_n = utils.bary_coords(v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:], q_pts.T)

        cfs =  self.coeffs[:,:,:,trangs]
        # evaluate the quadratic Berstein-Bezier polynomial in each split triangle
        inds = range(len(nCs))

        outs = (bcc_n[:,0]**2)[None, None, :]*cfs[:,:,nCs[:,0], inds] + \
               (bcc_n[:,1]**2)[None, None, :]*cfs[:,:,nCs[:,1], inds] + \
               (bcc_n[:,2]**2)[None, None, :]*cfs[:,:,nCs[:,2], inds] + \
             2*(bcc_n[:,1]*bcc_n[:,0])[None, None, :]*cfs[:,:,nCs[:,3], inds] +\
             2*(bcc_n[:,2]*bcc_n[:,1])[None, None, :]*cfs[:,:,nCs[:,4], inds] +\
             2*(bcc_n[:,2]*bcc_n[:,0])[None, None, :]*cfs[:,:,nCs[:,5], inds]
        
        # then perform interpolation in time along first dimension:
        if self.nVs == 3:
            tau0, tau1, tau2 = self.ts[0], self.ts[0] + dt, self.ts[0] + 2*dt
            l0 = (t-tau1)*(t-tau2)/(2*dt**2)
            l1 = (t-tau0)*(t-tau2)/(-dt**2)
            l2 = (t-tau0)*(t-tau1)/(2*dt**2)

            return l0*outs[0] + l1*outs[1] + l2*outs[2]


        if self.nVs == 1:

            return outs[0]

        if self.nVs == 2:
            tau = (t - self.ts[0])/dt

            return (1-tau)*outs[0] + tau*outs[1]


        return outs



    def assemble_coefficients(self, coeffs, i):
        """
        assemble all the coefficients to perform the PS split interpolation.
        """
        # void function replaces the coefficients of the array:
        tri_vals = self.vals[i][:,:,np.array(self.mesh.simplices)]
        ps_split = self.mesh.ps_split


        coeffs[:,0,:] =  tri_vals[:,0,:,0] 
        coeffs[:,1,:] =  tri_vals[:,0,:,1]
        coeffs[:,2,:] =  tri_vals[:,0,:,2]

        coeffs[:,3,:] = (1/ps_split[1][0][:,1])[None,:]*(utils.dot(ps_split[0][0], tri_vals[:,1::,:,0])/2 - (ps_split[1][0][:,0])[None,:]*tri_vals[:,0,:,0])
        coeffs[:,4,:] = (1/ps_split[1][6][:,1])[None,:]*(utils.dot(ps_split[0][6], tri_vals[:,1::,:,0])/2 - (ps_split[1][6][:,0])[None,:]*tri_vals[:,0,:,0])
        coeffs[:,5,:] = (1/ps_split[1][4][:,2])[None,:]*(utils.dot(ps_split[0][4], tri_vals[:,1::,:,0])/2 - (ps_split[1][4][:,0])[None,:]*tri_vals[:,0,:,0])

        coeffs[:,6,:] = (1/ps_split[1][2][:,1])[None,:]*(utils.dot(ps_split[0][2], tri_vals[:,1::,:,1])/2 - (ps_split[1][2][:,0])[None,:]*tri_vals[:,0,:,1])
        coeffs[:,7,:] = (1/ps_split[1][7][:,1])[None,:]*(utils.dot(ps_split[0][7], tri_vals[:,1::,:,1])/2 - (ps_split[1][7][:,0])[None,:]*tri_vals[:,0,:,1])
        coeffs[:,8,:] = (1/ps_split[1][1][:,2])[None,:]*(utils.dot(ps_split[0][1], tri_vals[:,1::,:,1])/2 - (ps_split[1][1][:,0])[None,:]*tri_vals[:,0,:,1])

        coeffs[:,9,:] = (1/ps_split[1][5][:,1])[None,:]*(utils.dot(ps_split[0][5], tri_vals[:,1::,:,2])/2 - (ps_split[1][5][:,0])[None,:]*tri_vals[:,0,:,2])
        coeffs[:,10,:] = (1/ps_split[1][8][:,1])[None,:]*(utils.dot(ps_split[0][8], tri_vals[:,1::,:,2])/2 - (ps_split[1][8][:,0])[None,:]*tri_vals[:,0,:,2])
        coeffs[:,11,:] = (1/ps_split[1][3][:,2])[None,:]*(utils.dot(ps_split[0][3], tri_vals[:,1::,:,2])/2 - (ps_split[1][3][:,0])[None,:]*tri_vals[:,0,:,2])


        coeffs[:,12,:] = (ps_split[2][0][:,0])[None,:]*coeffs[:,3,:] + (ps_split[2][0][:,1])[None,:]*coeffs[:,8,:]
        coeffs[:,13,:] = (ps_split[2][1][:,1])[None,:]*coeffs[:,6,:] + (ps_split[2][1][:,2])[None,:]*coeffs[:,11,:]
        coeffs[:,14,:] = (ps_split[2][2][:,2])[None,:]*coeffs[:,9,:] + (ps_split[2][2][:,0])[None,:]*coeffs[:,5,:]
        coeffs[:,15,:] = (ps_split[2][0][:,0])[None,:]*coeffs[:,4,:] + (ps_split[2][0][:,1])[None,:]*coeffs[:,7,:]
        coeffs[:,16,:] = (ps_split[2][1][:,1])[None,:]*coeffs[:,7,:] + (ps_split[2][1][:,2])[None,:]*coeffs[:,10,:]
        coeffs[:,17,:] = (ps_split[2][2][:,0])[None,:]*coeffs[:,4,:] + (ps_split[2][2][:,2])[None,:]*coeffs[:,10,:]

        #barycentre coords of middle points:
        coeffs[:,18,:] = (ps_split[1][9][:,0])[None,:]*coeffs[:,4,:] + (ps_split[1][9][:,1])[None,:]*coeffs[:,7,:] + (ps_split[1][9][:,2])[None,:]*coeffs[:,10,:]

        return coeffs

# interpolation functions: --------------------------------------------------------------------

#Coefficient list, this is needed for some silly indexing.
Cs = np.array([[[0,0,0,0,0,0], [18,1,13,7,6,16], [13,2,18,11,10,16]],
               [[18,14,0,17,5,4],[0,0,0,0,0,0],[14,18,2,17,10,9]],
               [[0,12,18,3,15,4],[1,18,12,7,15,8],[0,0,0,0,0,0]]])

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
    Js = np.vstack(edges_ps[bmm[0], bmm[1]])
    N = np.shape(Js)[0]
    v_pts_n1, v_pts_n2, v_pts_n3 = EE[Js[:,0],range(N),:], EE[Js[:,1],range(N),:], EE[Js[:,2],range(N),:]
    v_pts_n[:,0,:], v_pts_n[:,1,:], v_pts_n[:,2,:] = v_pts_n1, v_pts_n2, v_pts_n3

    return v_pts_n

def bary_minmax(X):
    """
    input: X should be a (N,3) array
    output: (N,) array of index (0,1,2) for min and max barycentric coord
    """

    return np.argmin(X, axis = 1), np.argmax(X, axis = 1)


def grad_HBB(Cfs, bcc_n, Bs):
    bcc_x, bcc_y, bcc_z = Bs[0], Bs[1], Bs[2]

    dp_b1 = 2*Cfs[:,0]*bcc_n[:,0] + 2*Cfs[:,3]*bcc_n[:,1] + 2*Cfs[:,5]*bcc_n[:,2]

    dp_b2 = 2*Cfs[:,1]*bcc_n[:,1] + 2*Cfs[:,3]*bcc_n[:,0] + 2*Cfs[:,4]*bcc_n[:,2]

    dp_b3 = 2*Cfs[:,2]*bcc_n[:,2] + 2*Cfs[:,4]*bcc_n[:,1] + 2*Cfs[:,5]*bcc_n[:,0]

    outx = bcc_x[:,0]*dp_b1 +  bcc_x[:,1]*dp_b2 + bcc_x[:,2]*dp_b3
    outy = bcc_y[:,0]*dp_b1 +  bcc_y[:,1]*dp_b2 + bcc_y[:,2]*dp_b3
    outz = bcc_z[:,0]*dp_b1 +  bcc_z[:,1]*dp_b2 + bcc_z[:,2]*dp_b3

    return [outx, outy, outz]

def d_Proj(X, A):
    outx = (1-X[:,0]**2)*A[0] + (-X[:,0]*X[:,1])*A[1] + (-X[:,0]*X[:,2])*A[2]
    outy = (-X[:,0]*X[:,1])*A[0] + (1-X[:,1]**2)*A[1] + (-X[:,1]*X[:,2])*A[2]
    outz = (-X[:,0]*X[:,2])*A[0] + (-X[:,1]*X[:,2])*A[1] + (1-X[:,2]**2)*A[2]

    return [outx, outy, outz]


