# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from .collapsed_mixture import CollapsedMixture
import GPflow
import tensorflow as tf


class MOHGP(CollapsedMixture):
    """
    A Hierarchical Mixture of Gaussian Processes

    A hierarchy is formed by using a GP model the mean function of a cluster,
    and further GPs to model the deviation of each time-course in the cluster
    from the mean.

    Arguments
    =========
    X        - The times of observation of the time series in a (Tx1) np.array
    Y        - A np.array of the observed time-course values: each row contains
               a time series, each column represents a unique time point
    kernF    - A GPflow kernel to model the mean function of each cluster
    kernY    - A GPflow kernel to model the deviation of each of the time courses
               from the mean of the cluster
    alpha    - The a priori Dirichlet concentrationn parameter (default 1.)
    prior_Z  - Either 'symmetric' or 'dp', specifies whether to use a symmetric Dirichlet
               prior for the clusters, or a (truncated) Dirichlet Process.
    name     - A convenient string for printing the model (default MOHGP)

    """
    def __init__(self, X, kernF, kernY, Y, num_clusters=2, alpha=1., prior_Z='symmetric'):

        num_data, self.D = Y.shape
        self.Y = GPflow.param.DataHolder(Y, on_shape_change='raise')
        self.X = GPflow.param.DataHolder(X, on_shape_change='pass')
        #assert X.shape[1] == self.D, "input data don't match observations"

        CollapsedMixture.__init__(self, num_data, num_clusters, prior_Z, alpha)

        self.kernF = kernF
        self.kernY = kernY
        self.LOG2PI = np.log(2.*np.pi)

        # Computations that can be done outside the optimisation loop
        self.YTY = GPflow.param.DataHolder(np.dot(Y.T, Y))

    def build_likelihood(self):

        tile_shape = tf.stack([tf.shape(self.logphi)[1], 1, 1])

        Sf = self.kernF.K(self.X)
        Sy = self.kernY.K(self.X)

        Sy_chol = tf.cholesky(Sy + tf.eye(self.D) * 1e-6)
        Sy_logdet = 2.*tf.reduce_sum(tf.log(tf.diag_part(Sy_chol)))
        tmp = tf.matrix_triangular_solve(Sy_chol, tf.eye(self.D), lower=True)
        Sy_inv = tf.matrix_triangular_solve(tf.transpose(Sy_chol), tmp, lower=False)

        phi = tf.nn.softmax(self.logphi)
        ybark = tf.transpose(tf.matmul(tf.transpose(phi), self.Y))
        phi_hat = tf.reduce_sum(phi, 0)

        # compute posterior variances of each cluster (lambda_inv)
        tmp1 = tf.matrix_triangular_solve(Sy_chol, Sf, lower=True)
        tmp = tf.matrix_triangular_solve(Sy_chol, tf.transpose(tmp1), lower=True)

        Cs = tf.expand_dims(tf.eye(self.D), 0) + tf.expand_dims(tmp, 0) * tf.reshape(phi_hat, tile_shape)
        C_chols = tf.cholesky(Cs + tf.expand_dims(tf.eye(self.D), 0) * 1e-6)
        log_det_diff_sum = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(C_chols)))

        L_tiled = tf.tile(tf.expand_dims(tf.transpose(Sy_chol), 0), tile_shape)
        tmp = tf.matrix_triangular_solve(C_chols, L_tiled, lower=True)
        tmp2 = tf.matmul(tf.matrix_transpose(tmp), tmp)
        Lambda_inv = (tf.expand_dims(Sy, 0) - tmp2) / tf.reshape(phi_hat, tile_shape)

        Li_ybark = tf.matrix_triangular_solve(Sy_chol, ybark, lower=True)
        Syi_ybark = tf.matrix_triangular_solve(tf.transpose(Sy_chol), Li_ybark, lower=False)
        Syi_ybarkybarkT_Syi = tf.expand_dims(tf.transpose(Syi_ybark), 1) * tf.expand_dims(tf.transpose(Syi_ybark), 2)

        return -0.5 * (self.num_data * self.D * self.LOG2PI +
                       log_det_diff_sum + self.num_data * Sy_logdet +
                       tf.reduce_sum(self.YTY * Sy_inv) -
                       tf.reduce_sum(Syi_ybarkybarkT_Syi * Lambda_inv))\
            - self.build_KL_Z()

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_components(self, Xnew):
        """The predictive density under each component"""
        tile_shape = tf.stack([tf.shape(self.logphi)[1], 1, 1])

        Sf = self.kernF.K(self.X)
        Sf_tiled = tf.tile(tf.expand_dims(Sf, 0), tile_shape)

        Sy = self.kernY.K(self.X)
        Sy_chol = tf.cholesky(Sy + tf.eye(self.D) * 1e-6)
        Sy_chol_inv = tf.matrix_triangular_solve(Sy_chol, tf.eye(self.D), lower=True)
        Sy_inv = tf.matrix_triangular_solve(tf.transpose(Sy_chol), Sy_chol_inv, lower=False)
        Sy_chol_inv_tiled = tf.tile(tf.expand_dims(Sy_chol_inv, 0), tile_shape)
        Sy_inv_tiled = tf.tile(tf.expand_dims(Sy_inv, 0), tile_shape)

        phi = tf.nn.softmax(self.logphi)
        phi_hat = tf.reduce_sum(phi, 0)

        ybark = tf.transpose(tf.matmul(tf.transpose(phi), self.Y))

        tmp1 = tf.matrix_triangular_solve(Sy_chol, Sf, lower=True)
        tmp = tf.matrix_triangular_solve(Sy_chol, tf.transpose(tmp1), lower=True)
        Cs = tf.expand_dims(tf.eye(self.D), 0) + tf.expand_dims(tmp, 0) * tf.reshape(phi_hat, tile_shape)

        C_chols = tf.cholesky(Cs + tf.expand_dims(tf.eye(self.D), 0) * 1e-6)

        tmp = tf.matrix_triangular_solve(C_chols, Sy_chol_inv_tiled, lower=True)
        B_invs = tf.matmul(tf.matrix_transpose(tmp), tmp) * tf.reshape(phi_hat, tile_shape)

        kx = self.kernF.K(self.X, Xnew)
        kx_tiled = tf.tile(tf.expand_dims(kx, 0), tile_shape)
        kxx = self.kernF.K(Xnew) + self.kernY.K(Xnew)

        tmp = tf.expand_dims(tf.eye(self.D), 0) - tf.matmul(B_invs, Sf_tiled)

        tmp = tf.matmul(tf.matrix_transpose(kx_tiled), tmp)
        tmp = tf.matmul(tmp, Sy_inv_tiled)
        mu = tf.matmul(tmp, tf.expand_dims(tf.transpose(ybark), 2))

        tmp = tf.matmul(B_invs, kx_tiled)
        tmp = tf.matmul(tf.matrix_transpose(kx_tiled), tmp)
        var = tf.expand_dims(kxx, 0) - tmp

        return mu, var
