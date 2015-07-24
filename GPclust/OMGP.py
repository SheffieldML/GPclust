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

        # Initialize kernel
        self.Xtot = np.hstack((self.X, np.atleast_2d(np.argmax(self.phi, 1)).T))
        self.Sy = self.kern.K(self.Xtot)
        # self.Sy_inv, self.Sy_chol, self.Sy_chol_inv, self.Sy_logdet = pdinv(self.Sy + np.eye(self.D) * 1e-6)

        # self.do_computations()

    def do_computation(self):
        """
        Here we do all the computations that are required whenever the kernels
        or the variational parameters are changed.
        """
        # sufficient stats.
        self.ybark = np.dot(self.phi.T,self.Y).T

        # compute posterior variances of each cluster (lambda_inv)
        tmp = backsub_both_sides(self.Sy_chol, self.Sf, transpose='right')
        self.Cs = [np.eye(self.D) + tmp * phi_hat_i for phi_hat_i in self.phi_hat]

        self._C_chols = [jitchol(C) for C in self.Cs]
        self.log_det_diff = np.array([2.*np.sum(np.log(np.diag(L))) for L in self._C_chols])
        tmp = [dtrtrs(L, self.Sy_chol.T, lower=1)[0] for L in self._C_chols]
        self.Lambda_inv = np.array([( self.Sy - np.dot(tmp_i.T, tmp_i) )/phi_hat_i if (phi_hat_i>1e-6) else self.Sf for phi_hat_i, tmp_i in zip(self.phi_hat, tmp)])

        #posterior mean and other useful quantities
        self.Syi_ybark, _ = dpotrs(self.Sy_chol, self.ybark, lower=1)
        self.Syi_ybarkybarkT_Syi = self.Syi_ybark.T[:,None,:]*self.Syi_ybark.T[:,:,None]
        self.muk = (self.Lambda_inv*self.Syi_ybark.T[:,:,None]).sum(1).T

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
