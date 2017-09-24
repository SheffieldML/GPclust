# Copyright (c) 2012, 2013, 2014, 2016 James Hensman, Dan Marthaler
# Licensed under the GPL v3 (see LICENSE.txt)
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import gpflow
from .tf_utilities import ln_dirichlet_C
from .collapsed_vb import CollapsedVB


class CollapsedMixture(CollapsedVB):
    """
    A base class for collapsed mixture models based on the CollapsedVB class

    We inherrit from this to build mixtures of Gaussians, mixures of GPs etc.

    This handles the mixing proportion part of the model,
    as well as providing generic functions for a merge-split approach
    """
    def __init__(self, num_data, num_clusters, prior_Z='symmetric', alpha=1.0):
        """
        Arguments
        =========
        num_data: the number of data
        num_clusters: the (initial) number of cluster (or truncation)
        prior_Z  - either 'symmetric' or 'DP', specifies whether to use a symmetric
                   Dirichlet prior for the clusters, or a (truncated) Dirichlet Process.
        alpha: parameter of the Dirichlet (process)

        """
        CollapsedVB.__init__(self)
        self.num_data, self.num_clusters = num_data, num_clusters
        assert prior_Z in ['symmetric', 'DP']
        self.prior_Z = prior_Z
        self.alpha = alpha

        # hold the variational parameters in a DataHolder
        self.logphi = gpflow.param.DataHolder(np.random.randn(self.num_data, self.num_clusters), on_shape_change='pass')

    def set_vb_param(self, logphi):
        """
        Accept a vector representing the variational parameters, and reshape it into self.phi
        """
        self.logphi = logphi.reshape(self.num_data, self.num_clusters)

    def get_vb_param(self):
        return self.logphi.value.flatten()

    def build_KL_Z(self):
        """
        The portion of the bound which is provided by the mixing proportions

        This function returns a tensorflow expression to be used inside build_likelihood
        """
        phi = tf.nn.softmax(self.logphi)
        phi_hat = tf.reduce_sum(phi, 0)
        entropy = -tf.reduce_sum(tf.multiply(phi,tf.nn.log_softmax(self.logphi)))
        if self.prior_Z == 'symmetric':
            return -ln_dirichlet_C(np.ones(self.num_clusters) * self.alpha)\
                + ln_dirichlet_C(self.alpha + phi_hat)\
                - entropy
        elif self.prior_Z == 'DP':
            alpha = tf.to_double(self.alpha)
            phi_tilde_plus_hat = tf.reverse(tf.cumsum(tf.reverse(phi_hat,[True])), [True])
            phi_tilde = phi_tilde_plus_hat - phi_hat
            A = tf.lgamma(1. + phi_hat)
            B = tf.lgamma(alpha + phi_tilde)
            C = tf.lgamma(alpha + 1. + phi_tilde_plus_hat)
            D = tf.to_double(self.num_clusters)*(tf.lgamma(1. + alpha)-tf.lgamma(alpha))
            return -tf.reduce_sum(A)\
                - tf.reduce_sum(B)\
                + tf.reduce_sum(C)\
                - D\
                - entropy
        else:
            raise NotImplementedError("invalid mixing proportion prior type: %s" % self.prior_Z)

    @gpflow.param.AutoFlow()
    def get_phihat(self):
        phi = tf.nn.softmax(self.logphi)
        phi_hat = tf.reduce_sum(phi, 0)
        return phi_hat

    @gpflow.param.AutoFlow()
    def log_likelihood(self):
        return self.build_likelihood()

    @gpflow.param.AutoFlow()
    def get_phi(self):
        phi = tf.nn.softmax(self.logphi)
        return phi

    def bound(self):
        return self.compute_log_likelihood()

    @gpflow.param.AutoFlow()
    def vb_bound_grad_natgrad(self):
        """
        Natural Gradients of the bound with respect to the variational
        parameters controlling assignment of the data to clusters
        """
        bound = self.build_likelihood()
        grad, = tf.gradients(bound, self.logphi)
        natgrad = grad / (tf.nn.softmax(self.logphi) + 1e-6)
        grad, natgrad = tf.clip_by_value(grad, -100, 100), tf.clip_by_value(natgrad, -100, 100)
        return bound, tf.reshape(grad, [-1]), tf.reshape(natgrad, [-1])

    def reorder(self):
        """
        Re-order the clusters so that the biggest one is first. This increases
        the bound if the prior type is a DP.
        """
        if self.prior_Z == 'DP':
            phi_hat = self.get_phihat()
            i = np.argsort(phi_hat)[::-1]
            self.set_vb_param(self.logphi.value[:, i])

    def remove_empty_clusters(self, threshold=1e-6):
        """Remove any cluster which has no data assigned to it"""
        i = self.get_phihat() > threshold
        new_logphi = self.logphi.value[:, i]
        self.num_clusters = i.sum()
        self.set_vb_param(new_logphi)

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
        phi = self.get_phi()
        phi_hat = self.get_phihat()
        if indexK is None:
            indexK = np.random.multinomial(1, phi_hat/self.num_data).argmax()
        if indexK > (self.num_clusters-1):
            return False  # index exceed no. clusters
        elif phi_hat[indexK] < 1:
            return False  # no data to split

        # ensure there's something to split
        if np.sum(phi[:, indexK] > threshold) < 2:
            return False

        if verbose:
            print("\nattempting to split cluster ", indexK)

        bound_old = self.bound()
        logphi_old = self.logphi.value
        param_old = self.get_parameter_dict()
        old_num_clusters = self.num_clusters

        # re-initalize
        self.num_clusters += 1
        logphi = np.hstack((logphi_old, logphi_old.min(1)[:, None]))
        indexN = np.nonzero(phi[:, indexK] > threshold)[0]

        # this procedure equally assigns data to the new and old clusters,
        # aside from one random point, which is in the new cluster
        special = np.random.permutation(indexN)[0]
        logphi[indexN, -1] = logphi[indexN, indexK].copy()
        logphi[special, -1] = np.max(logphi[special])+10
        self.set_vb_param(logphi)

        self.optimize(maxiter=maxiter, verbose=verbose)
        self.remove_empty_clusters()
        bound_new = self.bound()

        bound_increase = bound_new - bound_old

        if (bound_increase < 1e-3):
            self.num_clusters = old_num_clusters
            self.set_vb_param(logphi_old)
            self.set_parameter_dict(param_old)
            if verbose:
                print("split failed, bound changed by: ", bound_increase, '(K=%s)' % self.num_clusters)

            return False

        else:
            if verbose:
                print("split suceeded, bound changed by: ", bound_increase,
                      ',', self.num_clusters-old_num_clusters, ' new clusters', '(K=%s)' % self.num_clusters)
                print("optimizing new split to convergence:")
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

    def recursive_splits(self, k=0, verbose=True, optimize_params=None):
        """
        A recursive function which attempts to split a cluster
        (indexed by k), and if sucessful attempts to split the resulting clusters
        """
        success = self.try_split(k, verbose=verbose, optimize_params=optimize_params)
        if success:
            if not k == (self.num_clusters-1):
                self.recursive_splits(self.num_clusters-1, verbose=verbose, optimize_params=optimize_params)
            self.recursive_splits(k, verbose=verbose, optimize_params=optimize_params)
