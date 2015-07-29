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

        # self.do_computations()

    def do_computation(self):
        """
        Here we do all the computations that are required whenever the kernels
        or the variational parameters are changed.
        """
        pass

    def bound(self):
        """
        Compute the lower bound on the marginal likelihood (conditioned on the
        GP hyper parameters).
        """
        return  -0.5 * ( \
                    self.N * self.D * np.log(2. * np.pi) \
                    + 0 ) \
                + self.mixing_prop_bound() \
                + self.H

    def predict_components(self, Xnew):
        """The predictive density under each component"""
        mus = []
        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            kx = kern.K(self.X, Xnew)

            # This Cholesky version isn't working... Fix later
            # I = np.eye(self.N)
            # B12 = np.sqrt(np.diag(self.phi[:, i] / self.s2))
            # R = jitchol(I + B12.dot(K.dot(B12)))
            # B12y = B12.dot(self.Y)
            # tmp1 = np.linalg.solve(R.T, B12y)
            # tmp2 = np.linalg.solve(R, tmp1)
            # mu = kx.T.dot(B12.dot(tmp2))

            # Don't do this due to numerical stability!
            B_inv = np.diag(1. / (self.phi[:, i] / self.s2))
            self.K = K
            self.B_inv = B_inv
            mu = kx.T.dot(np.linalg.solve((K + B_inv), self.Y))

            mus.append(mu)

        return np.array(mus)
