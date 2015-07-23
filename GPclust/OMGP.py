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
    def __init__(self, X, Y, K=2, alpha=1., prior_Z='symmetric', name='OMGP'):

        N, self.D = Y.shape
        self.Y = Y
        self.X = X

        self.kern = GPy.kern.RBF(input_dim=1) ** GPy.kern.Coregionalize(1, output_dim=K)

        CollapsedMixture.__init__(self, N, K, prior_Z, alpha, name)

        # Initialize kernels
        Xtot = np.hstack((self.X, np.atleast_2d(np.argmax(self.phi, 1)).T))
        self.Sy = self.kern.K(Xtot)

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
        return  -0.5 * ( self.N * self.D * np.log(2. * np.pi) \
                    + 0 ) \
                + self.mixing_prop_bound() \
                + self.H

    def _raw_predict(self, _Xnew):
        Kx = self.kern.K(_Xnew, self.X).T
