#/---
"""
An assortment of utility functions
"""
#/---
import numpy as np
import pdb

def sphere2cart(phi, theta):
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return [x,y,z]

def norm(x):
    return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

def cart2sphere(XYZ):
    #This is modified to lambda \in [0,2pi)
    x, y, z = XYZ[0], XYZ[1], XYZ[2]
    phi = (np.arctan2(y,x) + 2*np.pi) % (2*np.pi)
    theta = np.arctan2(np.sqrt(y**2 + x**2), z)
    return [phi, theta]

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]


def tan_proj(X, A):
    outx = (1-X[0]**2)*A[0] + (-X[0]*X[1])*A[1] + (-X[0]*X[2])*A[2]
    outy = (-X[0]*X[1])*A[0] + (1-X[1]**2)*A[1] + (-X[1]*X[2])*A[2]
    outz = (-X[0]*X[2])*A[0] + (-X[1]*X[2])*A[1] + (1-X[2]**2)*A[2]

    return [outx, outy, outz]

def div_norm(x):
    Norm = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
    return np.array([x[:,0]/Norm, x[:,1]/Norm, x[:,2]/Norm])

def det(A):
    """
    should be input as A = [a,b,c] where a,b,c are considered as columns of A
    """
    v_1, v_2, v_3 = A[0], A[1], A[2]
    det = v_1[0]*v_2[1]*v_3[2] + v_2[0]*v_3[1]*v_1[2] + v_1[1]*v_2[2]*v_3[0] - \
          (v_3[0]*v_2[1]*v_1[2] + v_3[1]*v_2[2]*v_1[0] + v_2[0]*v_1[1]*v_3[2])
    return det

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
    order counter-clockwise. v is the transformed point.
    """
    denom = det_vec([v_1, v_2, v_3])
    bcc_outs = np.stack([det_vec([v, v_2, v_3])/denom,
                         det_vec([v_1,v,v_3])/denom,
                         det_vec([v_1, v_2, v])/denom], axis = 1)
    return bcc_outs

def bary_coords_alt(vs, v, outs):
    """
    vs define vertices of the containing triangle
    order counter-clockwise. v is the transformed point.
    """
    denom = np.linalg.det(vs)
    outs[:,0]  = np.linalg.det(np.stack([v, vs[:,1,:], vs[:,2,:]], axis = 2))/denom
    outs[:,1]  = np.linalg.det(np.stack([vs[:,0,:],v, vs[:,2,:]], axis = 2))/denom
    outs[:,2]  = np.linalg.det(np.stack([vs[:,0,:], vs[:,1,:],v], axis = 2))/denom

    return outs

def identity(xyz):
    return [xyz[0], xyz[1], xyz[2]]

def identity_x(xyz):
    return xyz[0]

def grad_x(xyz):
    return np.array(tan_proj(xyz,np.array([0*xyz[0] + 1, 0*xyz[1], 0*xyz[2]])))

def grad_y(xyz):
    return np.array(tan_proj(xyz,np.array([0*xyz[0], 0*xyz[1] + 1, 0*xyz[2]])))

def grad_z(xyz):
    return np.array(tan_proj(xyz,np.array([0*xyz[0], 0*xyz[1], 0*xyz[2] + 1])))

def d_Proj(X, A):
    outx = (1-X[0]**2)*A[0] + (-X[0]*X[1])*A[1] + (-X[0]*X[2])*A[2]
    outy = (-X[0]*X[1])*A[0] + (1-X[1]**2)*A[1] + (-X[1]*X[2])*A[2]
    outz = (-X[0]*X[2])*A[0] + (-X[1]*X[2])*A[1] + (1-X[2]**2)*A[2]

    return [outx, outy, outz]

def cross(a,b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def Mmul(A,B):
    return np.matmul(A,B)

def Rot_x(alpha):
    return np.array([[1,0,0], [0, np.cos(alpha), -np.sin(alpha)],[0, np.sin(alpha), np.cos(alpha)]])

def Rot_y(alpha):
    return np.array([[np.cos(alpha), 0, -np.sin(alpha)],[0,1,0],[np.sin(alpha), 0, np.cos(alpha)]])

def Rot_z(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],[np.sin(alpha), np.cos(alpha), 0],[0,0,1]])

def rotate(L, XYZ):

    outx = L[0,0]*XYZ[0] + L[0,1]*XYZ[1] + L[0,2]*XYZ[2]
    outy = L[1,0]*XYZ[0] + L[1,1]*XYZ[1] + L[1,2]*XYZ[2]
    outz = L[2,0]*XYZ[0] + L[2,1]*XYZ[1] + L[2,2]*XYZ[2]

    return [outx, outy, outz]


def mat_mul(L, XYZ):

    outx = L[0][0]*XYZ[0] + L[0][1]*XYZ[1] + L[0][2]*XYZ[2]
    outy = L[1][0]*XYZ[0] + L[1][1]*XYZ[1] + L[1][2]*XYZ[2]
    outz = L[2][0]*XYZ[0] + L[2][1]*XYZ[1] + L[2][2]*XYZ[2]

    return [outx, outy, outz]


def pi_proj(y,n):
    # inverse of tangent plane projection on unit sphere
    tt = -dot(y.T,n.T) + np.sqrt(dot(y.T,n.T)**2 - dot(y.T,y.T) + 1)
    return (y + tt[:,None]*n).T
