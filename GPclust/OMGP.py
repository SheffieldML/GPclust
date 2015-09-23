# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from collapsed_mixture import CollapsedMixture
import GPy
from GPy.util.linalg import mdot, pdinv, backsub_both_sides, dpotrs, jitchol, dtrtrs
from utilities import multiple_mahalanobis
from scipy import linalg

class OMGP(CollapsedMixture):
    """ OMGP Model
    """
    def __init__(self, X, Y, K=2, kernels=None, alpha=1., prior_Z='symmetric', name='OMGP'):

        N, self.D = Y.shape
        self.Y = Y
        self.X = X

        if kernels == None:
            self.kern = []
            for i in range(K):
                self.kern.append(GPy.kern.RBF(input_dim=1))

        CollapsedMixture.__init__(self, N, K, prior_Z, alpha, name)
        
        self.link_parameter(GPy.core.parameterization.param.Param('variance', 0.01))
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
            B_inv = np.diag(1. / (self.phi[:, i] / self.variance))

            alpha = linalg.cho_solve(linalg.cho_factor(K + B_inv), self.Y)
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
            B_inv = np.diag(1. / ((self.phi[:, i]+1e-6) / self.variance))

            # Data fit
            alpha = linalg.cho_solve(linalg.cho_factor(K + B_inv), self.Y)

            GP_bound += -0.5 * np.dot(self.Y.T, alpha)

            # Penalty
            GP_bound += -0.5 * np.linalg.slogdet(K + B_inv)[1]

            # Constant, weighted by  model assignment per point
            GP_bound += -0.5 * (self.phi[:, i] * np.log(2 * np.pi * self.variance)).sum()

        return  GP_bound + self.mixing_prop_bound() + self.H

    def vb_grad_natgrad(self):
        """
        Natural Gradients of the bound with respect to phi, the variational
        parameters controlling assignment of the data to GPs
        """
        grad_Lm = np.zeros_like(self.phi)
        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            I = np.eye(self.N)

            B_inv = np.diag(1. / ((self.phi[:, i]+1e-6) / self.variance))
            alpha = np.linalg.solve(K + B_inv, self.Y)
            K_B_inv = pdinv(K + B_inv)[0]
            dL_dB = np.outer(alpha, alpha) - K_B_inv

            for n in range(self.phi.shape[0]):
                grad_B_inv = np.zeros_like(B_inv)
                grad_B_inv[n, n] = -self.variance / (self.phi[n, i] ** 2 + 1e-6)
                grad_Lm[n, i] = 0.5 * np.trace(np.dot(dL_dB, grad_B_inv))

        grad_phi = grad_Lm + self.mixing_prop_bound_grad() + self.Hgrad

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
        B_inv = np.diag(1. / (self.phi[:, i] / self.variance))
        mu = kx.T.dot(np.linalg.solve((K + B_inv), self.Y))

        return mu

    def predict_components(self, Xnew):
        """The predictive density under each component"""
        mus = []
        for i in range(len(self.kern)):
            mu = self.predict(Xnew, i)
            mus.append(mu)

        return np.array(mus)[:, :, 0].T
