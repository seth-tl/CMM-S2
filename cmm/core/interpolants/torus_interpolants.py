#/----
"""
Base classes for Hermite interpolants on periodic domains
"""
#/----
from numpy import absolute, array, shape, sum, where, pi, meshgrid
import numpy as np
import pdb
#-----------------------------------------------------------------------------

class hermite_density():

    def __init__(self, Hmap):

        self.Hmap = Hmap
        return

    def __call__(self,x,y):

        rev_remaps = self.Hmap[::-1]
        J = 1
        pts = [x.copy(),y.copy()]
        for Chi in rev_remaps:
            J *= Chi.density(pts)
            pts = Chi(pts)

        return J

class Hermite_Map():
    # H_phi, H_the are the approximations of the displacement map
    def __init__(self, mesh, Chix, Chiy, identity = False):
        self.mesh = mesh

        if identity == False:
            #simply inherit the interpolants
            self.Chi_x = Chix
            self.Chi_y = Chiy

        else:
            XX = self.mesh.vertices
            self.Chi_x = Hermite_T2(mesh = self.mesh,
                                  f = 0*XX[0], f_x = 0*XX[0],
                                  f_y = 0*XX[0], f_xy = 0*XX[0])

            self.Chi_y = Hermite_T2(mesh = self.mesh,
                                  f = 0*XX[1], f_x = 0*XX[1],
                                  f_y = 0*XX[1], f_xy = 0*XX[1])


        return

    def __call__(self,X):
        out_x = (X[0] + self.Chi_x.eval(X[0],X[1])) % (2*pi)
        out_y = (X[1] + self.Chi_y.eval(X[0],X[1])) % (2*pi)     

        return np.array([out_x, out_y])
    
    def eval_displacement(self, X):
        out_x = self.Chi_x.eval(X[0],X[1])
        out_y = self.Chi_y.eval(X[0],X[1])

        return np.array([out_x, out_y])

    def grad(self, X):
        dX = self.Chi_x.eval_grad(X[0], X[1])
        dY = self.Chi_y.eval_grad(X[0], X[1])
        return [dX, dY]

    def density(self, X):
        grad = self.grad(X)
        return (1 + grad[0][0])*(1 + grad[1][1]) - grad[0][1]*grad[1][0]



class Hermite_velocity():

    def __init__(self, Vx, Vy):

        self.Vx = Vx
        self.Vy = Vy
        return
    
    def eval(self, xy):
        return array([self.Vx(xy[0], xy[1]), self.Vy(xy[0], xy[1])]) 

class Bilinear_Map():

    def __init__(self, X1, X2):
        """
        X1, X2 - Bilinear_T2 interpolants
        """
        self.X = X1
        self.Y = X2

        return
    def __call__(self, xy):
        
        x = (xy[0] + self.X(xy)) % (2*np.pi)
        y = (xy[1] + self.Y(xy)) % (2*np.pi)

        return np.array([x,y])

class Bilinear_T2():

    def __init__(self, xs, ys, f):
        self.xs = xs
        self.ys = ys
        self.f = f

        return
    def __call__(self, qpts):
        return self.eval(qpts[0], qpts[1])
    
    def eval(self, x, y):

        dx = self.xs[1]-self.xs[0]; dy = self.ys[1]-self.ys[0]

        x0 = (x//dx).astype(int); x1 = x0 + 1
        y0 = (y//dy).astype(int); y1 = y0 + 1

        # Handle periodicity in both directions
        x0[x0 < 0] = len(self.xs) - 1
        x1[x1 >= len(self.xs)] = 0

        y0[y0 < 0] = len(self.ys) - 1
        y1[y1 >= len(self.ys)] = 0

        x0 = np.clip(x0, 0, len(self.xs) - 1)
        x1 = np.clip(x1, 0, len(self.xs) - 1)
        y0 = np.clip(y0, 0, len(self.ys) - 1)
        y1 = np.clip(y1, 0, len(self.ys) - 1)

        q11 = self.f[y0, x0]
        q12 = self.f[y1, x0]
        q21 = self.f[y0, x1]
        q22 = self.f[y1, x1]

        fx = (x - self.xs[x0]) / (self.xs[x1] - self.xs[x0])
        fy = (y - self.ys[y0]) / (self.ys[y1] - self.ys[y0])

        out = (1 - fx)*(1 - fy)*q11 + fx*(1 - fy)*q21 + (1 - fx)*fy*q12 + fx*fy*q22

        return out

class Hermite_T2():

    def __init__(self, mesh, f, f_x, f_y, f_xy):

        """
        phi, theta = sorted 1D-arrays for the points along each axis (not meshgrids)
        f, f_x, f_y, f_xy are meshgrids of function values and derivatives.
        """
        self.mesh = mesh
        self.f = f.copy()
        self.f_x = f_x.copy()
        self.f_y = f_y.copy()
        self.f_xy = f_xy.copy()

        return

    def copy(self):
        return Hermite_T2(self.mesh, self.f, self.f_x, self.f_y,
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
        NN = len(phi0[0,:])
        MM = len(theta0[:,0])
        dx = self.mesh.xs[1] - self.mesh.xs[0]
        dy = self.mesh.ys[1] - self.mesh.ys[0]

        ijs, Evals = self.mesh.query(phi0, theta0)

        # For the endpoint:
        ijsp1 = [ijs[0] + 1, ijs[1] + 1]
        ijsp1[0] = ijsp1[0] % (len(self.mesh.xs))  
        ijsp1[1] = ijsp1[1] % (len(self.mesh.ys)) 

        if deriv == "zero":
            B_phi = self.basis_eval(Evals[0], dx)
            B_theta = self.basis_eval(Evals[1], dy)

        elif deriv == "dx":
            B_phi = self.basis_eval_deriv(Evals[0], dx)
            B_theta = self.basis_eval(Evals[1], dy)

        elif deriv == "dy":
            B_phi = self.basis_eval(Evals[0], dx)
            B_theta = self.basis_eval_deriv(Evals[1], dy)

        elif deriv == "dxdy":
            B_phi = self.basis_eval_deriv(Evals[0], dx)
            B_theta = self.basis_eval_deriv(Evals[1], dy)


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

    def eval_for_map(self, phi0, theta0, ijs, Evals, deriv = "zero"):
        NN = len(phi0[0,:])
        MM = len(theta0[:,0])
        dx = self.mesh.xs[1] - self.mesh.xs[0]
        dy = self.mesh.ys[1] - self.mesh.ys[0]

        ijs, Evals = self.mesh.query(phi0, theta0)

        # For the endpoint:
        ijsp1 = [ijs[0] + 1, ijs[1] + 1]
        ijsp1[0] = ijsp1[0] % (len(self.mesh.xs))  
        ijsp1[1] = ijsp1[1] % (len(self.mesh.ys)) 

        if deriv == "zero":
            B_phi = self.basis_eval(Evals[0], dx)
            B_theta = self.basis_eval(Evals[1], dy)

        elif deriv == "dx":
            B_phi = self.basis_eval_deriv(Evals[0], dx)
            B_theta = self.basis_eval(Evals[1], dy)

        elif deriv == "dy":
            B_phi = self.basis_eval(Evals[0], dx)
            B_theta = self.basis_eval_deriv(Evals[1], dy)

        elif deriv == "dxdy":
            B_phi = self.basis_eval_deriv(Evals[0], dx)
            B_theta = self.basis_eval_deriv(Evals[1], dy)


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

        return array([self.eval(phi0, theta0, deriv= "dx"), self.eval(phi0, theta0, deriv= "dy")])

    def eval_curl(self, phi0, theta0):

        return array([-self.eval(phi0, theta0, deriv= "dy"), self.eval(phi0, theta0, deriv= "dx")])


    def stencil_eval(self, phi, theta, phi_s, theta_s):

        NN = len(phi[0,:])
        MM = len(theta[:,0])

        dphi = absolute(self.mesh.xs[1]-self.mesh.xs[0])
        dthe = absolute(self.mesh.ys[1]-self.mesh.ys[0])
        delta_phi = phi_s - phi
        delta_theta = theta_s - theta

        phi0 = phi % (2*pi)
        theta0 = theta % (2*pi)


        ijs = [((phi0-self.mesh.xs[0])//dphi).astype(int),
                ((theta0-self.mesh.ys[0])//dthe).astype(int)]

        # if any point landed exactly on boundary:
        ijsnx = where(ijs[0] == len(self.mesh.xs))
        ijsny = where(ijs[1] == len(self.mesh.ys))

        # send to 0:
        theta0[ijsny] = 0.; phi0[ijsnx] = 0.

        # then mod out :
        ijs = [ijs[0] % len(self.mesh.xs), ijs[1] % len(self.mesh.ys)]

        #also compute position within the cell
        phi_c = ((phi0-self.mesh.xs[ijs[0]])/dphi) 
        theta_c = ((theta0-self.mesh.ys[ijs[1]])/dthe) 

        # for the stencils w.r.t the base point of the grid point for
        # which they are stencilling
        Evals = [phi_c + delta_phi/dphi, theta_c + delta_theta/dthe]

        # For the endpoint:
        ijsp1 = [(ijs[0] + 1) % (len(self.mesh.xs)), (ijs[1] + 1) % (len(self.mesh.ys))]

        # Create a list for each basis element evaluated at coords
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


    # def inds(self, phi0, theta0, phi_s0, theta_s0):
    #     """
    #     input a value (phi,theta) to interpolate
    #     phi0, theta0 should be meshgrids with the same amount of points and
    #     square.
    #
    #     phi_s0, theta_s0: Numpy array of 4 meshgrids of the stencils points
    #     each grid represents one of the four corners.
    #
    #     output: list of indices, base points of position in the cell also
    #     position of the stencil point relative to the cell
    #     """
    #
    #     phi = phi0.copy()
    #     theta = theta0.copy()
    #     delta_phi = phi_s0 - phi0
    #     delta_theta = theta_s0 - theta0
    #     dphi = abs(self.phi[1]-self.phi[0])
    #     dthe = abs(self.theta[1]-self.theta[0])
    #
    #     phi = phi % (2*pi)
    #     theta = theta % (2*pi)
    #
    #     phi_l = ((phi-self.phi[0])//dphi).astype(int) % (len(self.phi))
    #     theta_l = ((theta-self.theta[0])//dthe).astype(int) % (len(self.theta))
    #
    #     #also compute position within the cell
    #     phi_c = (phi-self.phi[phi_l])/dphi
    #     theta_c = (theta-self.theta[theta_l])/dthe
    #
    #     # for the stencils w.r.t the base point of the grid point for
    #     # which they are stencilling
    #
    #     phi_S = phi_c + delta_phi/dphi
    #     theta_S = theta_c + delta_theta/dthe
    #
    #     return [phi_l, theta_l], [phi_c, theta_c], [phi_S, theta_S]


def Eval_Tuple(H, X):

    return [H[0].eval(phi = X[0], theta = X[1], phi_s = X[0], theta_s = X[0], stencil = False),
            H[1].eval(phi = X[0], theta = X[1], phi_s = X[0], theta_s = X[0], stencil = False)]

