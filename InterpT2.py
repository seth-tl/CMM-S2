from numpy import cos, sin, absolute, matmul, array, dot, empty, shape, sum, where, pi, exp, log, sign, meshgrid
from datetime import datetime
import pdb

#-----------------------------------------------------------------------------

"""
What does this script do?

Base class for Hermite interpolation on the Torus

"""
#------------------------------------------------------------------------------

def Mmul(A,b):
    return  matmul(A,b)

def tic(x):
    return datetime.now()
def toc(time):
    print(datetime.now()-time)
    return

#-------------------------------------------------------------------------------


def Eval_Tuple(H, X):

    return [H[0].eval(phi = X[0], theta = X[1], phi_s = X[0], theta_s = X[0], stencil = False),
            H[1].eval(phi = X[0], theta = X[1], phi_s = X[0], theta_s = X[0], stencil = False)]


#-------------------------------------------------------------------------------

class hermite_density():

    def __init__(self, Hphi, H_the):

        self.Hx = Hphi
        self.Hy = H_the
        return

    def __call__(self, x,y):
        dHx = [1 + self.Hx.eval(x,y,deriv="phi"), self.Hx.eval(x,y,deriv="theta")]
        dHy = [self.Hy.eval(x,y,deriv="phi"), 1+self.Hy.eval(x,y,deriv="theta")]
        return absolute(dHx[0]*dHy[1] - dHx[1]*dHy[0])

class Hermite_Map():
    # H_phi, H_the are the approximations of the displacement map
    def __init__(self, A, B, identity = False):

        if identity == False:
            self.Chi_x = A
            self.Chi_y = B
        else:
            XX = meshgrid(A,B)
            self.Chi_x = Hermite_T2(phi = A, theta = B,
                                  f = 0*XX[0], f_x = 0*XX[0],
                                  f_y = 0*XX[0], f_xy = 0*XX[0])

            self.Chi_y = Hermite_T2(phi = A, theta = B,
                                  f = 0*XX[1], f_x = 0*XX[1],
                                  f_y = 0*XX[1], f_xy = 0*XX[1])


        return

    def __call__(self,X):
        out_x = (X[0] + self.Chi_x.eval(X[0],X[1])) % (2*pi)
        out_y = (X[1] + self.Chi_y.eval(X[0],X[1])) % (2*pi)
        return [out_x, out_y]

    def grad(self, X):
        dX = self.Chi_x.eval_grad(X[0], X[1])
        dY = self.Chi_y.eval_grad(X[0], X[1])
        return [dX, dY]

    def density(self, X):
        grad = self.grad(X)
        return (1 + grad[0][0])*(1 + grad[1][1]) - grad[0][1]*grad[1][0] 

class Hermite_T2():

    def __init__(self, phi, theta, f, f_x, f_y, f_xy):

        """
        phi, theta = sorted 1D-arrays for the points along each axis (not meshgrids)
        f, f_x, f_y, f_xy are meshgrids of function values and derivatives.
        """
        self.phi = phi
        self.theta = theta
        self.f = f.copy()
        self.f_x = f_x.copy()
        self.f_y = f_y.copy()
        self.f_xy = f_xy.copy()


        return

    def copy(self):
        return Hermite_T2(self.phi, self.theta, self.f, self.f_x, self.f_y,
                          self.f_xy)

    def __call__(self, x,y):
        return self.eval(x,y)

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

        N, M = shape(B_x)[1], shape(B_x)[2]

        return [B_x[0].reshape([int(N*M),])*B_y[0].reshape([int(N*M),]),
               B_x[1].reshape([int(N*M),])*B_y[0].reshape([int(N*M),]),
               B_x[0].reshape([int(N*M),])*B_y[1].reshape([int(N*M),]),
               B_x[1].reshape([int(N*M),])*B_y[1].reshape([int(N*M),])]

    def Kron1D(self, B_x, B_y):
        #Kron for 1D inputs

        return [B_x[0]*B_y[0], B_x[1]*B_y[0], B_x[0]*B_y[1], B_x[1]*B_y[1]]

    def eval(self, phi0, theta0, deriv = "zero"):
        """
        Inputs:
        phi = Meshgrid of query points in azimuthal angle
        theta = meshgrid of query points in polar angle
        deriv = string of either "phi", "theta", "mix"
        denoting which partial derivative one wants to evaluate, default is
        simply the function value.
        """
        #pdb.set_trace()
        NN = len(phi0[0,:])
        MM = len(theta0[:,0])

        phi = phi0.copy()
        theta = theta0.copy()
        dphi = abs(self.phi[1]-self.phi[0])
        dthe = abs(self.theta[1]-self.theta[0])

        phi = phi % (2*pi)
        theta = theta % (2*pi)

        ijs = [((phi-self.phi[0])//dphi).astype(int) % (len(self.phi)),
               ((theta-self.theta[0])//dthe).astype(int) % (len(self.theta))]

        Evals = [(phi-self.phi[ijs[0]])/dphi,(theta-self.theta[ijs[1]])/dthe]

        # For the endpoint:
        ijsp1 = [ijs[0] + 1, ijs[1] + 1]
        ijsp1[0] = ijsp1[0] % (len(self.phi))  #[where(ijsp1[0] == len(self.phi))] = 0
        ijsp1[1] = ijsp1[1] % (len(self.theta)) #[where(ijsp1[1] == len(self.phi))] = 0

        if deriv == "zero":
            B_phi = self.basis_eval(Evals[0], dphi)
            B_theta = self.basis_eval(Evals[1], dthe)

        if deriv == "dx":
            B_phi = self.basis_eval_deriv(Evals[0], dphi)
            B_theta = self.basis_eval(Evals[1], dthe)

        if deriv == "dy":
            B_phi = self.basis_eval(Evals[0], dphi)
            B_theta = self.basis_eval_deriv(Evals[1], dthe)

        if deriv == "dxdy":
            B_phi = self.basis_eval_deriv(Evals[0], dphi)
            B_theta = self.basis_eval_deriv(Evals[1], dthe)


        #Build arrays containing all necessary values
        # Arrays will have size [N**2, 16]

        ff = array([array(self.f[ijs[1], ijs[0]]).reshape([NN*MM,]), array(self.f[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f[ijsp1[1], ijs[0]]).reshape([NN*MM,]), array(self.f[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_x[ijs[1], ijs[0]]).reshape([NN*MM,]), array(self.f_x[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_x[ijsp1[1], ijs[0]]).reshape([NN*MM,]), array(self.f_x[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_y[ijs[1], ijs[0]]).reshape([NN*MM,]), array(self.f_y[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_y[ijsp1[1], ijs[0]]).reshape([NN*MM,]), array(self.f_y[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_xy[ijs[1], ijs[0]]).reshape([NN*MM,]), array(self.f_xy[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_xy[ijsp1[1], ijs[0]]).reshape([NN*MM,]), array(self.f_xy[ijsp1[1], ijsp1[0]]).reshape([NN*MM,])])


        B_tmp = array([self.Kron(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[0], B_theta[1]]) + \
                      self.Kron(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[0], B_theta[1]]) + \
                      self.Kron(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[2], B_theta[3]]) + \
                      self.Kron(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[2], B_theta[3]])])

        U = sum(B_tmp[0].T*ff.T, axis = 1)
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
        return array([self.eval(phi0, theta0, deriv= "dx"), self.eval(phi0, theta0, deriv= "dy")])

    def eval_curl(self, phi0, theta0):
        """
        Inputs:
        phi = Meshgrid of query points in azimuthal angle
        theta = meshgrid of query points in polar angle
        deriv = string of either "phi", "theta", "mix"
        denoting which partial derivative one wants to evaluate, default is
        simply the function value.
        """
        return array([-self.eval(phi0, theta0, deriv= "dy"), self.eval(phi0, theta0, deriv= "dx")])


    def stencil_eval(self, phi, theta, phi_s, theta_s):
        """
        Inputs:
        phi = Meshgrid of query points in azimuthal angle
        theta = meshgrid of query points in polar angle
        deriv = string of either "phi", "theta", "mix"
        denoting which partial derivative one wants to evaluate, default is
        simply the function value.
        """
        #pdb.set_trace()
        NN = len(phi[0,:])
        MM = len(theta[:,0])

        dphi = absolute(self.phi[1]-self.phi[0])
        dthe = absolute(self.theta[1]-self.theta[0])

        phi0 = phi.copy()
        theta0 = theta.copy()
        # pdb.set_trace()
        delta_phi = phi_s - phi0
        delta_theta = theta_s - theta0


        phi0 = phi0 % (2*pi)
        theta0 = theta0 % (2*pi)

        ijs = [((phi0-self.phi[0])//dphi).astype(int) % (len(self.phi)),
                ((theta0-self.theta[0])//dthe).astype(int) % (len(self.theta))]

        #also compute position within the cell
        phi_c = (phi0-self.phi[ijs[0]])/dphi
        theta_c = (theta0-self.theta[ijs[1]])/dthe

        # for the stencils w.r.t the base point of the grid point for
        # which they are stencilling

        Evals = [phi_c + delta_phi/dphi, theta_c + delta_theta/dthe]

        # For the endpoint:
        ijsp1 = [(ijs[0] + 1) % (len(self.phi)), (ijs[1] + 1) % (len(self.theta))]

        # Create two lists, one for each basis element evaluated at coords

        B_phi = self.basis_eval(Evals[0], dphi)
        B_theta = self.basis_eval(Evals[1], dthe)

        #Build arrays containing all necessary values
        # Arrays will have size [N**2, 16]

        ff = array([array(self.f[ijs[1], ijs[0]]).reshape([NN*MM,]), array(self.f[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f[ijsp1[1], ijs[0]]).reshape([NN*MM,]), array(self.f[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_x[ijs[1], ijs[0]]).reshape([NN*MM,]), array(self.f_x[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_x[ijsp1[1], ijs[0]]).reshape([NN*MM,]), array(self.f_x[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_y[ijs[1], ijs[0]]).reshape([NN*MM,]), array(self.f_y[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_y[ijsp1[1], ijs[0]]).reshape([NN*MM,]), array(self.f_y[ijsp1[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_xy[ijs[1], ijs[0]]).reshape([NN*MM,]), array(self.f_xy[ijs[1], ijsp1[0]]).reshape([NN*MM,]),
              array(self.f_xy[ijsp1[1], ijs[0]]).reshape([NN*MM,]), array(self.f_xy[ijsp1[1], ijsp1[0]]).reshape([NN*MM,])])


        B_tmp = array([self.Kron(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[0], B_theta[1]]) + \
                      self.Kron(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[0], B_theta[1]]) + \
                      self.Kron(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[2], B_theta[3]]) + \
                      self.Kron(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[2], B_theta[3]])])

        U = sum(B_tmp[0].T*ff.T, axis = 1)
        # adjust values to incorporate the metric
        return U.reshape([MM, NN])


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
        delta_phi = phi_s0 - phi0
        delta_theta = theta_s0 - theta0
        dphi = abs(self.phi[1]-self.phi[0])
        dthe = abs(self.theta[1]-self.theta[0])

        phi = phi % (2*pi)
        theta = theta % (2*pi)

        phi_l = ((phi-self.phi[0])//dphi).astype(int) % (len(self.phi))
        theta_l = ((theta-self.theta[0])//dthe).astype(int) % (len(self.theta))

        #also compute position within the cell
        phi_c = (phi-self.phi[phi_l])/dphi
        theta_c = (theta-self.theta[theta_l])/dthe

        # for the stencils w.r.t the base point of the grid point for
        # which they are stencilling

        phi_S = phi_c + delta_phi/dphi
        theta_S = theta_c + delta_theta/dthe

        return [phi_l, theta_l], [phi_c, theta_c], [phi_S, theta_S]


# ---

#         st = 6*xp
#         omt = 1-xp
#         ttmo = 3*xp-1
#
#         h0p = -st*omt
#         p0p = -omt*ttmo
#         h1p = -h0p
#         p1p = xp*(ttmo-1)
#
#         return [(1/dx)*h0p, (1/dx)*h1p, p0p, p1p]
#



    # def basis_eval1d(self, xp, dx, deriv = "zero"):
    #

    #     if deriv == "zero":
    #         omt2 = (1-xp)**2
    #         t2 = xp**2
    #
    #         h0p = (1+2*xp)*omt2
    #         p0p = xp*omt2
    #         h1p = t2*(3-2*xp)
    #         p1p = t2*(xp-1)
    #
    #         return [h0p, h1p, p0p*dx, p1p*dx]
    #
    #     xp = array(xp.copy())
    #
    #     if deriv == "zero":
    #         omt2 = (1-xp)**2
    #         t2 = xp**2
    #
    #         h0p = (1+2*xp)*omt2
    #         p0p = xp*omt2
    #         h1p = t2*(3-2*xp)
    #         p1p = t2*(xp-1)
    #
    #         return [h0p, h1p, p0p*dx, p1p*dx]
    #
    #     if deriv == "first":
    #

    #     if deriv == "second":
    #         op = 6*xp-3
    #         h0p = 2*op
    #         p0p = op-1
    #         h1p = -h0p
    #         p1p = op+1
    #
    #         return [h0p/dx**2, h1p/dx**2, p0p/dx, p1p/dx]



    #
    # def eval1D(self, phi, theta, phi_s, theta_s, deriv = "zero", stencil = True):
    #     """
    #     Inputs:
    #     phi = list of query points in azimuthal angle
    #     theta = list of query points in polar angle
    #     must be of equal length
    #
    #     deriv = string of either "phi", "theta", "mix"
    #     denoting which partial derivative one wants to evaluate, default is
    #     simply the function value.
    #     """
    #     NN = len(phi)
    #
    #     dphi = absolute(self.phi[1]-self.phi[0])
    #     dthe = absolute(self.theta[1]-self.theta[0])
    #
    #     ijs, coords, S_coords = self.inds(phi,theta, phi_s, theta_s)
    #
    #     # To obtain the (i+1,j), (i+1,j+1) values, wraps around for the endpoints
    #     ijsp1 = [ijs[0] + 1, ijs[1] + 1]
    #     ijsp1[0] = ijsp1[0] % (len(self.phi))  #[where(ijsp1[0] == len(self.phi))] = 0
    #     ijsp1[1] = ijsp1[1] % (len(self.theta)) #[where(ijsp1[1] == len(self.phi))] = 0
    #
    #
    #     if stencil == True:
    #         Evals = S_coords
    #     else:
    #         Evals = coords
    #     # Create two lists, one for each basis element evaluated at coords
    #
    #     if deriv == "zero":
    #         B_phi = self.basis_eval1d(Evals[0], dphi)
    #         B_theta = self.basis_eval1d(Evals[1], dthe)
    #
    #     if deriv == "phi":
    #         B_phi = self.basis_eval1d(Evals[0], dphi, deriv = "first")
    #         B_theta = self.basis_eval1d(Evals[1], dthe)
    #
    #     if deriv == "theta":
    #         B_phi = self.basis_eval1d(Evals[0], dphi)
    #         B_theta = self.basis_eval1d(Evals[1], dthe, deriv = "first")
    #
    #     if deriv == "mix":
    #         B_phi = self.basis_eval1d(Evals[0], dphi, deriv = "first")
    #         B_theta = self.basis_eval1d(Evals[1], dthe, deriv = "first")
    #
    #
    #     #Build arrays containing all necessary values
    #     # Arrays will have size [N**2, 16]
    #
    #     ff = array([array(self.f[ijs[1], ijs[0]]).reshape([NN,]), array(self.f[ijs[1], ijsp1[0]]).reshape([NN,]),
    #           array(self.f[ijsp1[1], ijs[0]]).reshape([NN,]), array(self.f[ijsp1[1], ijsp1[0]]).reshape([NN,]),
    #           array(self.f_x[ijs[1], ijs[0]]).reshape([NN,]), array(self.f_x[ijs[1], ijsp1[0]]).reshape([NN,]),
    #           array(self.f_x[ijsp1[1], ijs[0]]).reshape([NN,]), array(self.f_x[ijsp1[1], ijsp1[0]]).reshape([NN,]),
    #           array(self.f_y[ijs[1], ijs[0]]).reshape([NN,]), array(self.f_y[ijs[1], ijsp1[0]]).reshape([NN,]),
    #           array(self.f_y[ijsp1[1], ijs[0]]).reshape([NN,]), array(self.f_y[ijsp1[1], ijsp1[0]]).reshape([NN,]),
    #           array(self.f_xy[ijs[1], ijs[0]]).reshape([NN,]), array(self.f_xy[ijs[1], ijsp1[0]]).reshape([NN,]),
    #           array(self.f_xy[ijsp1[1], ijs[0]]).reshape([NN,]), array(self.f_xy[ijsp1[1], ijsp1[0]]).reshape([NN,])])
    #
    #
    #     B_tmp = array([self.Kron1D(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[0], B_theta[1]]) + \
    #                   self.Kron1D(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[0], B_theta[1]]) + \
    #                   self.Kron1D(B_x = [B_phi[0], B_phi[1]], B_y = [B_theta[2], B_theta[3]]) + \
    #                   self.Kron1D(B_x = [B_phi[2], B_phi[3]], B_y = [B_theta[2], B_theta[3]])])
    #
    #     U = sum(B_tmp[0].T*ff.T, axis = 1)
    #     # adjust values to incorporate the metric
    #     return U
    #

        # if deriv == "zero" or deriv == "theta":
        #     return U.reshape([MM, NN])
        # if deriv == "phi":
        #     return U.reshape([MM, NN])
        # if deriv == "mix":
        #     return U.reshape([MM, NN])
