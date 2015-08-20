# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from collapsed_mixture import CollapsedMixture
import GPy
from GPy.util.linalg import mdot, pdinv, backsub_both_sides, dpotrs, jitchol, dtrtrs
from utilities import multiple_mahalanobis

class OMGP(CollapsedMixture):
    """ OMGP Model
    """
    def __init__(self, X, Y, K=2, kernels=None, alpha=1., prior_Z='symmetric', name='OMGP'):

        N, self.D = Y.shape
        self.Y = Y
        self.X = X

        self.s2 = 1.

        if kernels == None:
            self.kern = []
            for i in range(K):
                self.kern.append(GPy.kern.RBF(input_dim=1))

        CollapsedMixture.__init__(self, N, K, prior_Z, alpha, name)

        self.link_parameters(*self.kern)

    def parameters_changed(self):
        """ Set the kernel parameters
        """
        self.update_kern_grads()

    def do_computations(self):
        """
        Here we do all the computations that are required whenever the kernels
        or the variational parameters are changed.
        """
        if len(self.kern) < self.K:
            self.kern.append(GPy.kern.RBF(input_dim=1))
            self.link_parameter(self.kern[-1])

    def update_kern_grads(self):
        """
        Set the derivative of the lower bound wrt the (kernel) parameters
        """
        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            B_inv = np.diag(1. / (self.phi[:, i] / self.s2))

            # Should work out a ~Cholesky way of doing this due to stability
            alpha = np.linalg.solve(K + B_inv, self.Y)
            K_B_inv = pdinv(K + B_inv)[0]

            # Also not completely sure this actually is dL_dK
            dL_dK = np.outer(alpha, alpha) - K_B_inv
            kern.update_gradients_full(dL_dK=dL_dK, X=self.X)

    def bound(self):
        """
        Compute the lower bound on the marginal likelihood (conditioned on the
        GP hyper parameters).
        """
        GP_bound = 0.0

        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            B_inv = np.diag(1. / (self.phi[:, i] / self.s2))

            # Data fit, numerically unstable?
            alpha = np.linalg.solve(K + B_inv, self.Y)
            GP_bound += -0.5 * np.dot(self.Y.T, alpha)

            # Penalty
            GP_bound += -0.5 * np.linalg.slogdet(K + B_inv)[1]

            # Constant
            GP_bound += self.N / 2.0 * np.log(2. * np.pi)


        mixing_bound = self.mixing_prop_bound() + self.H
        norm_bound = -0.5 * self.D * (self.phi * np.log(2 * np.pi * self.s2)).sum()

        return  GP_bound + mixing_bound + norm_bound

    def vb_grad_natgrad(self):
        """
        Natural Gradients of the bound with respect to phi, the variational
        parameters controlling assignment of the data to GPs
        """
        ynmk2 = np.zeros_like(self.phi)
        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            I = np.eye(self.N)
            B12 = np.sqrt(np.diag(self.phi[:, i] / self.s2))

            # R = jitchol(I + B12.dot(K.dot(B12))).T

            B_inv = np.diag(1. / (self.phi[:, i] / self.s2))            
            R = jitchol((K + B_inv)).T

            muk = np.atleast_2d(self.predict(self.X, i))
            ynmk2[:, i:(i + 1)] = multiple_mahalanobis(self.Y, muk.T, R)

        grad_phi = (self.mixing_prop_bound_grad() - 
                     0.0) + \
                   (self.Hgrad - 0.5 * ynmk2)

        natgrad = grad_phi - np.sum(self.phi * grad_phi, 1)[:, None]
        grad = natgrad * self.phi

        return grad.flatten(), natgrad.flatten()

    def predict(self, Xnew, i):
        """ Predictive mean for a given component
        """
        kern = self.kern[i]
        K = kern.K(self.X)
        kx = kern.K(self.X, Xnew)

        # This works but should Cholesky for stability
        B_inv = np.diag(1. / (self.phi[:, i] / self.s2))
        mu = kx.T.dot(np.linalg.solve((K + B_inv), self.Y))

        return mu

    def predict_components(self, Xnew):
        """The predictive density under each component"""
        mus = []
        for i in range(len(self.kern)):
            mu = self.predict(Xnew, i)
            mus.append(mu)

        return np.array(mus)[:, :, 0].T
