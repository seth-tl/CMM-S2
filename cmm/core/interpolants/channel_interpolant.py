#/--------------
"""
Contains interpolation classes related to the channel flow problem
"""
#/--------------
import numpy as np
from math import factorial
import pdb
import scipy.io
from stripy import _stripack, _ssrfpack
from datetime import datetime
import igl as IGL
from pathos.multiprocessing import ProcessingPool as Pool

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# Tensor Product Interpolants

class Hermite_channel():

    def __init__(self, xs, ys, f, f_x, f_y, f_xy, L_x, L_y):

        """
        phi, theta = sorted 1D-arrays for the points along each axis (not meshgrids)
        f, f_x, f_y, f_xy are meshgrids of function values and derivatives.
        """
        self.xs = xs
        self.ys = ys
        self.f = f.copy()
        self.f_x = f_x.copy()
        self.f_y = f_y.copy()
        self.f_xy = f_xy.copy()
        self.L_x = L_x
        self.L_y = L_y


        return

    def copy(self):
        return Hermite_T2(self.xs, self.ys, self.f, self.f_x, self.f_y,
                          self.f_xy)

    def __call__(self, X):
        return self.eval(X[0],X[1])

    def basis_eval(self, pts, dx):
        p1 = (1-pts)**2
        p2 = pts**2

        out1 = (1+2*pts)*p1
        out2 = p2*(3-2*pts)
        out3 = pts*p1
        out4 = p2*(pts-1)

        return [out1, out2, out3*dx, out4*dx]


    def basis_eval_deriv(self, pts, dx):

        st = 6*pts
        omt = 1-pts
        ttmo = 3*pts-1

        h0p = -st*omt
        p0p = -omt*ttmo
        h1p = -h0p
        p1p = pts*(ttmo-1)

        return [(1/dx)*h0p, (1/dx)*h1p, p0p, p1p]

    def basis_eval_second_deriv(self, pts, dx):

        out1 = 2*(6*pts-3)/(dx**2)
        out2 = (6*pts-4)/dx
        out3 = (6*pts-2)/dx

        return [out1, -out1, out2, out3]

    def Kron(self, B_x, B_y):

        N, M = np.shape(B_x)[1], np.shape(B_x)[2]

        return [B_x[0].reshape([int(N*M),])*B_y[0].reshape([int(N*M),]),
               B_x[1].reshape([int(N*M),])*B_y[0].reshape([int(N*M),]),
               B_x[0].reshape([int(N*M),])*B_y[1].reshape([int(N*M),]),
               B_x[1].reshape([int(N*M),])*B_y[1].reshape([int(N*M),])]

    def Kron1D(self, B_x, B_y):
        #Kron for 1D inputs

        return [B_x[0]*B_y[0], B_x[1]*B_y[0], B_x[0]*B_y[1], B_x[1]*B_y[1]]

    def eval(self, phi0, theta0, deriv = "none"):
        """
        Inputs:
        phi = Meshgrid of query points in azimuthal angle
        theta = meshgrid of query points in polar angle
        deriv = string of either "phi", "theta", "mix"
        denoting which partial derivative one wants to evaluate, default is
        simply the function value.
        """
        NN = len(phi0[0,:])
        MM = len(theta0[:,0])
        phi = phi0.copy()
        theta = theta0.copy()
        dphi = abs(self.xs[1]-self.xs[0])
        dthe = abs(self.ys[1]-self.ys[0])

        phi = phi % self.L_x

        ijs = [((phi-self.xs[0])//dphi).astype(int),
               ((theta-self.ys[0])//dthe).astype(int) % (len(self.ys))]

        # For the endpoint:
        #if point lands on top boundary
        ijs[1][np.where(ijs[1] == len(self.ys)-1)] = len(self.ys)-2

        ijsnx = np.where(ijs[0] == len(self.xs))
        phi[ijsnx] = self.xs[0];

        # then mod out length
        ijs[0] = ijs[0] % len(self.xs)

        Evals = [(phi-self.xs[ijs[0]])/dphi,(theta-self.ys[ijs[1]])/dthe]


        ijsp1 = [ijs[0] + 1, ijs[1] + 1]

        # periodicity in the x coordinate
        ijsp1[0] = ijsp1[0] % (len(self.xs))  #[where(ijsp1[0] == len(self.phi))] = 0

        if deriv == "none":
            B_phi = self.basis_eval(Evals[0], dphi)
            B_theta = self.basis_eval(Evals[1], dthe)

        if deriv == "phi":
            B_phi = self.basis_eval_deriv(Evals[0], dphi, deriv = "first")
            B_theta = self.basis_eval1d(Evals[1], dthe)

        if deriv == "theta":
            B_phi = self.basis_eval(Evals[0], dphi)
            B_theta = self.basis_eval_deriv(Evals[1], dthe, deriv = "first")

        if deriv == "mix":
            B_phi = self.basis_eval_deriv(Evals[0], dphi, deriv = "first")
            B_theta = self.basis_eval_deriv(Evals[1], dthe, deriv = "first")


        #Build arrays containing all necessary values
        # Arrays will have size [N**2, 16]

        ff = np.array([np.array(self.f[ijs[1], ijs[0]]).reshape([NN*MM,]), np.array(self.f[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              np.array(self.f[ijsp1[1], ijs[0]]).reshape([NN*MM,]), np.array(self.f[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              np.array(self.f_x[ijs[1], ijs[0]]).reshape([NN*MM,]), np.array(self.f_x[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              np.array(self.f_x[ijsp1[1], ijs[0]]).reshape([NN*MM,]), np.array(self.f_x[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              np.array(self.f_y[ijs[1], ijs[0]]).reshape([NN*MM,]), np.array(self.f_y[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              np.array(self.f_y[ijsp1[1], ijs[0]]).reshape([NN*MM,]), np.array(self.f_y[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              np.array(self.f_xy[ijs[1], ijs[0]]).reshape([NN*MM,]), np.array(self.f_xy[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              np.array(self.f_xy[ijsp1[1], ijs[0]]).reshape([NN*MM,]), np.array(self.f_xy[ijsp1[1], ijsp1[0]]).reshape([NN*MM,])])


        B_tmp = np.array([self.Kron(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[0], B_theta[1]]) + \
                      self.Kron(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[0], B_theta[1]]) + \
                      self.Kron(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[2], B_theta[3]]) + \
                      self.Kron(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[2], B_theta[3]])])

        U = np.sum(B_tmp[0].T*ff.T, axis = 1)
        # adjust values to incorporate the metric
        return U.reshape([MM, NN])


    def eval_grad(self, phi0, theta0):
        """
        Inputs:
        phi = Meshgrid of query points in azimuthal angle
        theta = meshgrid of query points in polar angle
        deriv = string of either "phi", "theta", "mix"
        denoting which partial derivative one wants to evaluate, default is
        simply the function value.
        """
        return array([self.eval(phi0, theta0, deriv= "phi"), self.eval(phi0, theta0, deriv= "theta")])



    def stencil_eval(self, q_pts, spts):
        """
        Inputs:
        phi = Meshgrid of query points in azimuthal angle
        theta = meshgrid of query points in polar angle
        deriv = string of either "phi", "theta", "mix"
        denoting which partial derivative one wants to evaluate, default is
        simply the function value.
        """

        phi = q_pts[0]; theta = q_pts[1];
        phi_s0 = spts[0]; theta_s0 = spts[1];
        phi0 = phi.copy()
        theta0 = theta.copy()
        # "extending outside domain"
        # theta[theta <0] = 0.; theta[theta>self.L_y] = self.L_y;
        # theta_s[theta_s <0] = 0; theta_s[theta_s>self.L_y] = self.L_y;

        dphi = np.absolute(self.xs[1]-self.xs[0])
        dthe = np.absolute(self.ys[1]-self.ys[0])

        delta_phi = phi_s0 - phi0
        delta_theta = theta_s0 - theta0

        #periodicize the x-direction
        phi0 = phi0 % self.L_x

        ijs = [((phi0-self.xs[0])//dphi).astype(int) % (len(self.xs)),
               ((theta0-self.ys[0])//dthe).astype(int) % (len(self.ys))]
        # For the endpoint:
        #if point lands on top boundary
        ijs[1][np.where(ijs[1] == len(self.ys)-1)] = len(self.ys)-2

        #pdb.set_trace()
        #also compute position within the cell
        phi_c = (phi0-self.xs[ijs[0]])/dphi
        theta_c = (theta0-self.ys[ijs[1]])/dthe

        Evals = [phi_c + delta_phi/dphi, theta_c + delta_theta/dthe]

        # For the endpoint:
        ijsp1 = [ijs[0] + 1, ijs[1] + 1]
        ijsp1[0] = ijsp1[0] % (len(self.xs))  #[where(ijsp1[0] == len(self.phi))] = 0
        # ijsp1[1] = ijsp1[1] % (len(self.theta)) #[where(ijsp1[1] == len(self.phi))] = 0
        #----------------------------------

        B_phi = self.basis_eval(Evals[0], dphi)
        B_theta = self.basis_eval(Evals[1], dthe)

        #Build arrays containing all necessary values
        # Arrays will have size [N**2, 16]
        ff = np.array([np.array(self.f[ijs[1], ijs[0]]), np.array(self.f[ijs[1], ijsp1[0]]),
              np.array(self.f[ijsp1[1], ijs[0]]), np.array(self.f[ijsp1[1], ijsp1[0]]),
              np.array(self.f_x[ijs[1], ijs[0]]), np.array(self.f_x[ijs[1], ijsp1[0]]),
              np.array(self.f_x[ijsp1[1], ijs[0]]), np.array(self.f_x[ijsp1[1], ijsp1[0]]),
              np.array(self.f_y[ijs[1], ijs[0]]), np.array(self.f_y[ijs[1], ijsp1[0]]),
              np.array(self.f_y[ijsp1[1], ijs[0]]), np.array(self.f_y[ijsp1[1], ijsp1[0]]),
              np.array(self.f_xy[ijs[1], ijs[0]]), np.array(self.f_xy[ijs[1], ijsp1[0]]),
              np.array(self.f_xy[ijsp1[1], ijs[0]]), np.array(self.f_xy[ijsp1[1], ijsp1[0]])])


        B_tmp = np.array([self.Kron1D(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[0], B_theta[1]]) + \
                      self.Kron1D(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[0], B_theta[1]]) + \
                      self.Kron1D(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[2], B_theta[3]]) + \
                      self.Kron1D(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[2], B_theta[3]])])


        # adjust values to incorporate the metric
        return np.sum(B_tmp[0].T*ff.T, axis = 1) # U.reshape([MM, NN])


    def inds(self, phi0, theta0, phi_s0, theta_s0):
        """
        input a value (phi,theta) to interpolate
        phi0, theta0 should be meshgrids with the same amount of points and
        square.

        phi_s0, theta_s0: Numpy array of 4 meshgrids of the stencils points
        each grid represents one of the four corners.

        output: list of indices, base points of position in the cell also
        position of the stencil point relative to the cell
        """

        phi = phi0.copy()
        theta = theta0.copy()
        delta_phi = phi_s0 - phi
        delta_theta = theta_s0 - theta
        dphi = abs(self.xs[1]-self.xs[0])
        dthe = abs(self.ys[1]-self.ys[0])

        #periodicize the x-direction
        phi = phi % self.L_x

        ijs = [((phi-self.xs[0])//dphi).astype(int) % (len(self.xs)),
               ((theta-self.ys[0])//dthe).astype(int) % (len(self.ys))]

        # For the endpoint:
        #if point lands on top boundary
        ijs[1][np.where(ijs[1] == len(self.ys)-1)] = len(self.ys)-2

        #pdb.set_trace()
        #also compute position within the cell
        phi_c = (phi-self.xs[ijs[0]])/dphi
        theta_c = (theta-self.ys[ijs[1]])/dthe


        # for the stencils w.r.t the base point of the grid point for
        # which they are stencilling

        phi_S = phi_c + delta_phi/dphi
        theta_S = theta_c + delta_theta/dthe

        return ijs, [phi_c, theta_c], [phi_S, theta_S]



class channel_diffeomorphism():
    # X direction is the displacement map
    # Y direction is map from S^1 \times [0,L] to [0,l]
    def __init__(self, X, Y, mesh):
        self.Chi_x = X
        self.Chi_y = Y
        self.mesh = mesh
        return

    def __call__(self,X):
        out_x = (X[0] + self.Chi_x(X)) % (self.Chi_x.L_x)
        out_y = self.Chi_y(X)
        return [out_x, out_y]

    def stencil_eval(self, q_pts, spts, eps = 1e-5):
        # interior points in each direction
        # L_x = self.Chi_x.L_x;
        outs_x_int = [self.Chi_x.stencil_eval(q_pts, spts[0]),
                      self.Chi_x.stencil_eval(q_pts, spts[1]),
                      self.Chi_x.stencil_eval(q_pts, spts[2]),
                      self.Chi_x.stencil_eval(q_pts, spts[3])]

        outs_y_int = [self.Chi_y.stencil_eval(q_pts, spts[0]),
                      self.Chi_y.stencil_eval(q_pts, spts[1]),
                      self.Chi_y.stencil_eval(q_pts, spts[2]),
                      self.Chi_y.stencil_eval(q_pts, spts[3])]


        return [outs_x_int, outs_y_int] #, [outs_x_b, outs_y_b]]



class Bilinear_Cylinder():

    def __init__(self, f, xs, ys, Lx, Ly):
        self.xs = xs
        self.ys = ys
        self.f = f
        self.L_x = Lx; self.L_y = Ly;

        return

    def __call__(self, X):
        return self.eval(X[0], X[1])

    def eval(self, x0, y0):
        NN = len(x0[0,:])
        MM = len(y0[:,0])

        x= x0 % self.L_x
        y = y0 % (2*pi)
        dx = abs(self.xs[1]-self.xs[0])
        dy = abs(self.ys[1]-self.ys[0])

        ijs = [((x-self.xs[0])//dx).astype(int), ((y-self.ys[0])//dy).astype(int)]

        # evals = [(phi-self.xs[ijs[0]])/dphi,(theta-self.ys[ijs[1]])/dthe]
        ijs[1][np.where(ijs[1] == len(self.ys)-1)] = len(self.ys)-2

        ijsnx = np.where(ijs[0] == len(self.xs))

        if len(ijsnx[0]) != 0:
            x[ijsnx] = self.xs[0];

        # For the endpoint:
        ijsp1 = [(ijs[0] + 1) % len(self.xs), ijs[1] + 1]

        ff = [self.f[ijs[1], ijs[0]], self.f[ijs[1], ijsp1[0]],
              self.f[ijsp1[1], ijs[0]], self.f[ijsp1[1], ijsp1[0]]]

        x1 = self.xs[ijs[0]]; x2 = self.xs[ijsp1[0]]; y1 = self.ys[ijs[1]]; y2 = self.ys[ijsp1[1]];

        ff1 = [ff[0]*(y2 - y) + ff[1]*(y-y1), ff[2]*(y2 - y) + ff[3]*(y-y1)]

        return (1/(dx*dy))*(ff1[0]*(x2-x) + ff1[1]*(x-x1))
