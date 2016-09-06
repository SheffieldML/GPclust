# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np

from tf_utilities import  ln_dirichlet_C, softmax

from scipy.special import gammaln, digamma
from collapsed_vb import CollapsedVB

class CollapsedMixture(CollapsedVB):
    """
    A base class for collapsed mixture models based on the CollapsedVB class

    We inherrit from this to build mixtures of Gaussians, mixures of GPs etc.

    This handles the mixing proportion part of the model,
    as well as providing generic functions for a merge-split approach
    """
    def __init__(self, num_data, num_clusters, prior_Z='symmetric', alpha=1.0, name='col_mix'):
        """
        Arguments
        =========
        num_data: the number of data
        num_clusters: the (initial) number of cluster (or truncation)
        prior_Z  - either 'symmetric' or 'DP', specifies whether to use a symmetric Dirichlet prior for the clusters, or a (truncated) Dirichlet Process.
        alpha: parameter of the Dirichelt (process)

        """
        CollapsedVB.__init__(self, name)
        self.num_data, self.num_clusters = num_data, num_clusters
        assert prior_Z in ['symmetric','DP']
        self.prior_Z = prior_Z
        self.alpha = alpha

        #random initial conditions for the vb parameters
        self.phi_ = np.random.randn(self.num_data, self.num_clusters)
        self.phi, logphi, self.H = softmax(self.phi_)
        self.phi_hat = self.phi.sum(0)
        self.Hgrad = -logphi
        if self.prior_Z == 'DP':
            self.phi_tilde_plus_hat = self.phi_hat[::-1].cumsum()[::-1]
            self.phi_tilde = self.phi_tilde_plus_hat - self.phi_hat

    def set_vb_param(self,phi_):
        """
        Accept a vector representing the variatinoal parameters, and reshape it into self.phi
        """
        self.phi_ = phi_.reshape(self.num_data, self.num_clusters)
        self.phi, logphi, self.H = softmax(self.phi_)
        self.phi_hat = self.phi.sum(0)
        self.Hgrad = -logphi
        if self.prior_Z == 'DP':
            self.phi_tilde_plus_hat = self.phi_hat[::-1].cumsum()[::-1]
            self.phi_tilde = self.phi_tilde_plus_hat - self.phi_hat

        self.do_computations()

    def get_vb_param(self):
        return self.phi_.flatten()

    def mixing_prop_bound(self):
        """
        The portion of the bound which is provided by the mixing proportions
        """
        if self.prior_Z=='symmetric':
            return ln_dirichlet_C(np.ones(self.num_clusters)*self.alpha) -ln_dirichlet_C(self.alpha + self.phi_hat)
        elif self.prior_Z=='DP':
            A = gammaln(1. + self.phi_hat)
            B = gammaln(self.alpha + self.phi_tilde)
            C = gammaln(self.alpha + 1. + self.phi_tilde_plus_hat)
            D = self.num_clusters*(gammaln(1.+self.alpha) - gammaln(self.alpha))
            return A.sum() + B.sum() - C.sum() + D
        else:
            raise NotImplementedError("invalid mixing proportion prior type: %s" % self.prior_Z)

    def mixing_prop_bound_grad(self):
        """
        The gradient of the portion of the bound which arises from the mixing
        proportions, with respect to self.phi
        """
        if self.prior_Z=='symmetric':
            return digamma(self.alpha + self.phi_hat)
        elif self.prior_Z=='DP':
            A = digamma(self.phi_hat + 1.)
            B = np.hstack((0, digamma(self.phi_tilde + self.alpha)[:-1].cumsum()))
            C = digamma(self.phi_tilde_plus_hat + self.alpha + 1.).cumsum()
            return A + B - C
        else:
            raise NotImplementedError("invalid mixing proportion prior type: %s"%self.prior_Z)

    def reorder(self):
        """
        Re-order the clusters so that the biggest one is first. This increases
        the bound if the prior type is a DP.
        """
        if self.prior_Z=='DP':
            i = np.argsort(self.phi_hat)[::-1]
            self.set_vb_param(self.phi_[:,i])

    def remove_empty_clusters(self,threshold=1e-6):
        """Remove any cluster which has no data assigned to it"""
        i = self.phi_hat>threshold
        phi_ = self.phi_[:,i]
        self.num_clusters = i.sum()
        self.set_vb_param(phi_)

    def try_split(self, indexK=None, threshold=0.9, verbose=True, maxiter=100, optimize_params=None):
        """
        Re-initialize one of the clusters as two clusters, optimize, and keep
        the solution if the bound is increased. Kernel parameters stay constant.

        Arguments
        ---------
        indexK: (int) the index of the cluster to split
        threshold: float [0,1], to assign data to the splitting cluster
        verbose: whether to print status

        Returns
        -------
        Success: (bool)

        """
        if indexK is None:
            indexK = np.random.multinomial(1,self.phi_hat/self.num_data).argmax()
        if indexK > (self.num_clusters-1):
            return False #index exceed no. clusters
        elif self.phi_hat[indexK]<1:
            return False # no data to split

        #ensure there's something to split
        if np.sum(self.phi[:,indexK]>threshold) <2:
            return False

        if verbose:print("\nattempting to split cluster ", indexK)

        bound_old = self.bound()
        phi_old = self.get_vb_param().copy()
        self._optimizer_copy_transformed = False  # Redo transform in case parameters have been unlinked
        param_old = self.optimizer_array.copy()
        old_num_clusters = self.num_clusters

        #re-initalize
        self.num_clusters += 1
        self.phi_ = np.hstack((self.phi_,self.phi_.min(1)[:,None]))
        indexN = np.nonzero(self.phi[:,indexK] > threshold)[0]

        # this procedure equally assigns data to the new and old clusters, 
        # aside from one random point, which is in the new cluster
        special = np.random.permutation(indexN)[0]
        self.phi_[indexN,-1] = self.phi_[indexN,indexK].copy()
        self.phi_[special,-1] = np.max(self.phi_[special])+10
        self.set_vb_param(self.get_vb_param())

        self.optimize(maxiter=maxiter, verbose=verbose)
        self.remove_empty_clusters()
        bound_new = self.bound()

        bound_increase = bound_new - bound_old

        if (bound_increase < 1e-3):
            self.num_clusters = old_num_clusters
            self.set_vb_param(phi_old)
            self.optimizer_array = param_old
            if verbose:print("split failed, bound changed by: ",bound_increase, '(K=%s)' % self.num_clusters)

            return False

        else:
            if verbose:print("split suceeded, bound changed by: ", bound_increase, ',', self.num_clusters-old_num_clusters,' new clusters', '(K=%s)' % self.num_clusters)
            if verbose:print("optimizing new split to convergence:")
            if optimize_params:
                self.optimize(**optimize_params)
            else:
                self.optimize(maxiter=5000, verbose=verbose)

            return True

    def systematic_splits(self, verbose=True):
        """
        perform recursive splits on each of the existing clusters
        """
        for kk in range(self.num_clusters):
            self.recursive_splits(kk, verbose=verbose)

    def recursive_splits(self,k=0, verbose=True, optimize_params=None):
        """
        A recursive function which attempts to split a cluster (indexed by k), and if sucessful attempts to split the resulting clusters
        """
        success = self.try_split(k, verbose=verbose, optimize_params=optimize_params)
        if success:
            if not k==(self.num_clusters-1):
                self.recursive_splits(self.num_clusters-1, verbose=verbose, optimize_params=optimize_params)
            self.recursive_splits(k, verbose=verbose, optimize_params=optimize_params)
