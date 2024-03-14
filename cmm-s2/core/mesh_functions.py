# -----------------------------------------------------------------------------
"""
This script provides all helper functions for mesh generation and relevant
operations.
"""
# -----------------------------------------------------------------------------
import stripy, igl, pdb
import numpy as np
from . import utils
from scipy.spatial import Delaunay, ConvexHull, cKDTree
from numba import njit
# -----------------------------------------------------------------------------

# basic 2D mesh classes

# spherical triangulation--------------------------------------
class spherical_triangulation(object):
    """
    Base class for spherical triangulation
    mesh generation from the stripy package which provides a wrapper
    to the Fortran package STRIPACK of R.J. Renka.
    Modifications have been made to the class simply for the use case here.

    input:
        pts - (N,3) set of points on sphere in Cartesian coords
        refinement_levels - optional number of uniform (midpoint bisection)
                            of the initial triangulation.
    output: spherical triangulation of the points
    """
    def __init__(self, pts, refinement_levels = 0, stencils = None):

        # these are the lat-lon coordinates used by the stripy package
        lons = np.arctan2(pts[:,1], pts[:,0])
        lats = np.arcsin(pts[:,2])
        #
        grid = stripy.sTriangulation(lons, lats,
                                      refinement_levels = refinement_levels,
                                      permute = True, tree = False)

        self.vertices = grid.points
        #enforcing the Delaunay condition might not be smart in the future
        origin = np.array([0,0,0])
        ppoints = np.vstack((origin, grid.points))
        mesh_d = Delaunay(ppoints)
        
        simps0 = mesh_d.simplices - 1
        #now need to remove all the zeros
        simps = np.array([T[T != -1] for T in simps0])
        
        v_pts2 = self.vertices[simps]
        dets2 = utils.det_vec([v_pts2[:,0,:], v_pts2[:,1,:], v_pts2[:,2,:]])
        
        #correct for the orientation
        inds_1 = np.where(dets2 < 0)
        
        #flip the last two vectices to reverse orientation
        simps[inds_1,2:0:-1] = simps[inds_1,1::]
        v_pts1 = self.vertices[simps]
        
        self.mesh_d = mesh_d

        self.simplices = simps
        self.x, self.y, self.z = grid.x, grid.y, grid.z

        # # option to precompute stencil points or pass from another mesh
        if stencils == None:
            self.stencil_pts()
        else:
            self.stencil_pts = stencils
            
        return

    def query(self, q_pts):
        """
        Querying routine to find containing triangle for the query points
        Note: this is the optimized form I found given my resources
        and could/should be optimized further offering significant improvements
        to overall performance.
        TODO: Optimize the performance of this routine

        input:  q_pts - (N,3) array of query points
        output: bcc - (N,3) array of barycentric coordinates
                trangs - (N,) list of triangle indices in self.simplices
                vs - (N,3,3) array of corresponding vertex coordinates
        """
        tri_data = igl.point_mesh_squared_distance(q_pts.T, self.vertices, np.array(self.simplices))
        trangs = tri_data[1]
        tri_out = self.simplices[trangs.reshape(-1)]
        vs = self.vertices[tri_out]
        bcc = utils.bary_coords(vs[:,0,:],vs[:,1,:], vs[:,2,:], q_pts.T)
        
        # TODO: get rid of the necessity of performing this operation
        inds0 = np.where(bcc < 0)
        # extract relevant points
        q_pts2 = q_pts[:, inds0[0]]
        
        indsn0 = self.mesh_d.find_simplex(q_pts2.T/2)
        simps = self.simplices[indsn0]
        vs2 = self.vertices[simps]
        #replace with corrected values
        bcc[inds0[0],:] = utils.bary_coords(vs2[:,0,:],vs2[:,1,:], vs2[:,2,:], q_pts2.T)
        
        trangs[inds0[0]] = indsn0
        vs[inds0[0],:,:] = vs2
        
        return bcc, trangs, vs
    
        # # obtain containing triangle from the igl package helper function
        # tri_data = igl.point_mesh_squared_distance(q_pts.T, self.vertices,
        #                                            np.array(self.simplices))
        # # second output is the containing triangle index
        # trangs = tri_data[1]

        # # compute other relevant quantities
        # tri_out = self.simplices[trangs.reshape(-1)]
        # vs = self.vertices[tri_out]
        # bcc = utils.bary_coords(vs[:,0,:],vs[:,1,:], vs[:,2,:], q_pts.T)

        # return bcc, trangs, vs

    def ps_split(self):

        # these are all recycled quantities used for the Powell-Sabin interpolant
        inds = np.array(self.simplices)
        v_pts = self.vertices[inds]
        v1, v2, v3 = v_pts[:,0,:].copy(), v_pts[:,1,:].copy(), v_pts[:,2,:].copy()
        v4 = utils.div_norm((v1 + v2 + v3)/3).T
        e1, e2, e3 = utils.div_norm(v1/2+v2/2).T, utils.div_norm(v2/2+v3/2).T, utils.div_norm(v3/2 + v1/2).T

        # Hs correspond to tangential projection of vectors along the split
        h12, h21, h23 = sphere_tan_proj(v2-v1,v1).T, sphere_tan_proj(v1-v2,v2).T, sphere_tan_proj(v3-v2,v2).T
        h32, h13, h31 = sphere_tan_proj(v2-v3,v3).T, sphere_tan_proj(v3-v1,v1).T, sphere_tan_proj(v1-v3,v3).T
        h41, h42, h43 = sphere_tan_proj(v4-v1,v1).T, sphere_tan_proj(v4-v2,v2).T, sphere_tan_proj(v4-v3,v3).T

        # Gs barycentric coordinates within each split triangle
        g12, g21 = utils.bary_coords(v1,e1,v4,h12), utils.bary_coords(v2,v4,e1,h21)
        g23, g32 = utils.bary_coords(v2,e2,v4,h23), utils.bary_coords(v3,v4,e2,h32)
        g13, g31 = utils.bary_coords(v1,v4,e3,h13), utils.bary_coords(v3,e3,v4,h31)

        g14 = utils.bary_coords(v1,v4,e3, sphere_tan_proj(v4,v1).T)
        g24 = utils.bary_coords(v2,v4,e1, sphere_tan_proj(v4,v2).T)
        g34 = utils.bary_coords(v3,v4,e2, sphere_tan_proj(v4,v3).T)

        # barycentric coordinates of midpoint
        mid = utils.bary_coords(v1,v2,v3,v4)

        # barycentric coordinates of the edge vectors
        e_1, e_2, e_3 = utils.bary_coords(v1,v2,v3,e1), utils.bary_coords(v1,v2,v3,e2), utils.bary_coords(v1,v2,v3,e3)

        # organize data, naming subject to scrutiny
        Hs = [h12.T, h21.T, h23.T, h32.T, h13.T, h31.T, h41.T, h42.T, h43.T]
        Gs = [g12, g21, g23, g32, g13, g31, g14, g24, g34, mid]
        Es = [e_1, e_2, e_3]

        return Hs, Gs, Es

    def stencil_pts(self, eps = 1e-5):
        # void function to precompute the stencil points on the mesh
        # saves also the orthonormal basis at the vertices for gradient calc.
        # this works exclusively for isosahedral mesh
        verts = self.vertices
        # form direction vectors from every vertex
        a1 = np.array(utils.cross(verts.T,[0,0,1])).T
        a2 = np.array(utils.cross(verts.T, a1.T)).T

        #replace first and last rows <--- this is only valid for icosahedron
        a1[0,:] = utils.cross(verts[0,:], [1/np.sqrt(2),0,1/np.sqrt(2)])
        a1[-1,:] = utils.cross(verts[-1,:], [1/np.sqrt(2),0,-1/np.sqrt(2)])

        a2[0,:] = utils.cross(verts[0,:], a1[0,:])
        a2[-1,:] = utils.cross(verts[-1,:], a1[-1,:])

        #normalize the direction vectors
        a1_n, a2_n = utils.div_norm(a1).T, utils.div_norm(a2).T

        # 4 - point averaged
        eps_1p = verts - a1_n*eps - a2_n*eps
        eps_1m = verts + a1_n*eps - a2_n*eps
        eps_2p = verts - a1_n*eps + a2_n*eps
        eps_2m = verts + a1_n*eps + a2_n*eps

        # define the stencil points for the mesh
        spts = [utils.pi_proj(eps_1p, verts), utils.pi_proj(eps_1m, verts), utils.pi_proj(eps_2p, verts),
                utils.pi_proj(eps_2m, verts)]

        #arrange more conveniently for integration scheme
        spts = [np.array([spts[0][0,:], spts[1][0,:], spts[2][0,:], spts[3][0,:]]),
                np.array([spts[0][1,:], spts[1][1,:], spts[2][1,:], spts[3][1,:]]),
                np.array([spts[0][2,:], spts[1][2,:], spts[2][2,:], spts[3][2,:]])]

        #save directions for later use
        self.tan_vects = [a1_n, a2_n]
        self.s_pts = np.array(spts)

        return


def det2D(v1,v2,v3):
    return  1*v2[:,0]*v3[:,1] + 1*v3[:,0]*v1[:,1] + 1*v1[:,0]*v2[:,1] - (1*v2[:,0]*v1[:,1] + 1*v3[:,0]*v2[:,1] + 1*v1[:,0]*v3[:,1])


class torus_mesh():
    # TODO: include functionality for variable size (changes fft functions)
    def __init__(self, Nx, Ny):
        self.xs = np.linspace(0, 2*np.pi, Nx, endpoint = False)
        self.ys = np.linspace(0, 2*np.pi, Ny, endpoint = False)

        X0 = np.meshgrid(self.xs, self.ys)
        self.vertices = X0
        # initialize the stencil points
        eps = 1e-5
        self.s_pts = [np.array([X0[0] - eps, X0[1] - eps]), np.array([X0[0] - eps, X0[1] + eps]),
                np.array([X0[0] + eps, X0[1] - eps]), np.array([X0[0] + eps, X0[1] + eps])]

        return
    
    def query(self, phi0, theta0):

        phi = phi0 % (2*np.pi)
        theta = theta0 % (2*np.pi)
        dphi = abs(self.xs[1]-self.xs[0])
        dthe = abs(self.ys[1]-self.ys[0])

        ijs = [((phi-self.xs[0])//dphi).astype(int),
                ((theta-self.ys[0])//dthe).astype(int)]

        # if any point landed exactly on boundary:
        ijsnx = np.where(ijs[0] == len(self.xs))
        ijsny = np.where(ijs[1] == len(self.ys))

        # send to 0:
        theta[ijsny] = 0.; phi[ijsnx] = 0.

        # then mod out :
        ijs = [ijs[0] % len(self.xs), ijs[1] % len(self.ys)]


        # ijs = [((phi-self.xs[0])//dphi).astype(int) % (len(self.xs)),
        #        ((theta-self.ys[0])//dthe).astype(int) % (len(self.ys))]

        q_pts = [(phi-self.xs[ijs[0]])/dphi,(theta-self.ys[ijs[1]])/dthe]

        return ijs, q_pts



# 3D mesh classes. ------------------------------

def sphere_tan_proj(a,b):
    """
    a is the vector to be tangentialled
    b is the unit normal vector to the surface at the base point of a.
    """
    c = utils.dot(b.T,a.T)
    #d = np.sqrt(1-dot(b.T,a.T)**2)
    d = np.sqrt(utils.dot(a.T,a.T) - 2*c**2 + utils.dot(b.T,b.T)*c**2)

    return np.array([(a[:,0]-c*b[:,0])/d, (a[:,1]-c*b[:,1])/d, (a[:,2]-c*b[:,2])/d])



# functions related to discretization of velocity field:
def spherical_mesh(phi, theta, N, M):
    """
    generate structured mesh for interpolation of the velocity field
    inputs: phi, theta - Numpy meshgrids of size (N,M).
    theta grid should omit the poles
    outputs a stripy.mesh class based on the meshgrid defined by phi, theta
    """
    phis = list(phi.reshape([N*M,]))
    thetas = list(-theta.reshape([N*M,]) + np.pi/2)
    #convert to coordinates that stripy enjoys
    #add point at the poles
    phis.insert(0,0)
    phis.append(0)
    thetas.insert(0,np.pi/2)
    thetas.append(-np.pi/2)

    return stripy.sTriangulation(lons = np.array(phis),
                                 lats = np.array(thetas),
                                 permute = True)

def full_assembly(N,M):
    simps = grid_assemble(N,M)
    simplices_p1 = [T + [i] for (T,i) in zip(simps, range(len(simps)))]


    # # now populate a meshgrid with these simplices
    msimplices = np.empty([M-1,N,2, 4], dtype = int)

    #populate the data structure
    #first row
    for j in range(N):
        msimplices[0,j,0,:] = simplices_p1[j]
        msimplices[0,j,1,:] = simplices_p1[j]

    #middle section of rectangles
    k = N
    for i in range(1,M-2):
        for j in range(N):
            msimplices[i,j,0,:] = simplices_p1[k]
            msimplices[i,j,1,:] = simplices_p1[k+1]
            k +=2
    #last row
    for j in range(N):
        msimplices[M-2,j,0,:] = simplices_p1[j-N]
        msimplices[M-2,j,1,:] = simplices_p1[j-N]

    return simps, msimplices



def grid_assemble(N,M):
    """
    N,M are the length rows and columns of the overlain meshgrid.
    output: (N*M,3) array of indices for the simplices of the structured triangulation.
    counter clockwise ordering
    """
    #first row
    out = []
    for i in range(N-1):
        out.append([0, i+1, i+2])
    # wrap around
    out.append([0,N,1])
    #middle part
    for i in range(0,M-3):
        for j in range(1,N+1):
            k = j + i*N
            if j != N:
                out.append([k, k+N+1, k+1])
                out.append([k, k+N, k+N+1])
            else:
                out.append([k, k+1, i*N + 1])
                out.append([k, k+N, k+1])

    #last row:
    nn = N**2 -3*N + 1 # if N = M
    nn = (M-3)*N + 1

    for i in range(nn,nn + N-1):
        out.append([i,nn + N, i+1])
    #wrap arround
    out.append([i+1,nn + N, i+1-(N-1)])

    return out

# ==============================================================================



# # -------- older querying routines ----------------------------
# def bcc_simp_trang(points, simplices, q_pts, mesh_d):
    # tri_data = igl.point_mesh_squared_distance(q_pts.T, self.vertices, np.array(self.simplices))
    # trangs = tri_data[1]
    # tri_out = self.simplices[trangs.reshape(-1)]
    # vs = self.vertices[tri_out]
    # bcc = utils.bary_coords(vs[:,0,:],vs[:,1,:], vs[:,2,:], q_pts.T)
    # #
    # inds0 = np.where(bcc < 0)
    # # extract relevant points
    # q_pts2 = q_pts[:, inds0[0]]
    #
    # indsn0 = self.mesh_d.find_simplex(q_pts2.T/2)
    # simps = self.simplices[indsn0]
    # vs2 = self.vertices[simps]
    # #replace with corrected values
    # bcc[inds0[0],:] = utils.bary_coords(vs2[:,0,:],vs2[:,1,:], vs2[:,2,:], q_pts2.T)
    #
    # trangs[inds0[0]] = indsn0
    # vs[inds0[0],:,:] = vs2
    #
    # return bcc, trangs, vs
#
# def c_sim_bcc_scipy(mesh_d, points, simplices, q_pts):
#
#     inds = mesh_d.find_simplex(q_pts.T/2)
#     #subtract one to account for added origin
#     simps = mesh_d.simplices[inds] - 1
#     #now need to remove all the zeros
#     simps0 = np.array([T[T != -1] for T in simps])
#     vs = points[simps0]
#     bcc = bary_coords(vs[:,0,:],vs[:,1,:], vs[:,2,:], q_pts.T)
#
#     return bcc, inds, vs


# vertices = grid.points
# #enforcing the Delaunay condition might not be smart in the future
# origin = np.array([0,0,0])
# ppoints = np.vstack((origin, grid.points))
# mesh_d = Delaunay(ppoints)
#
# simps0 = mesh_d.simplices - 1
# #now need to remove all the zeros
# simps = np.array([T[T != -1] for T in simps0])
#
# v_pts2 = vertices[simps]
# dets2 = utils.det_vec([v_pts2[:,0,:], v_pts2[:,1,:], v_pts2[:,2,:]])
#
# #correct for the orientation
# inds_1 = np.where(dets2 < 0)
#
# #flip the last two vectices to reverse orientation
# simps[inds_1,2:0:-1] = simps[inds_1,1::]
# v_pts1 = vertices[simps]
# dets1 = utils.det_vec([v_pts1[:,0,:], v_pts1[:,1,:], v_pts1[:,2,:]])
#
# # self.mesh_d = mesh_d

# # -------------------------------------------------------
