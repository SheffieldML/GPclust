# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from .collapsed_mixture import CollapsedMixture
import GPflow
import tensorflow as tf


class OMGP(CollapsedMixture):
    """
    Overlapping mixtures of Gaussian processes
    """
    def __init__(self, X, Y, num_clusters=2, kernels=None, variance=1., alpha=1., prior_Z='symmetric', name='OMGP'):
        num_data, self.D = Y.shape
        self.Y = GPflow.param.DataHolder(Y, on_shape_change='raise')
        self.X = GPflow.param.DataHolder(X, on_shape_change='pass')
        #assert X.shape[0] == self.D, "input data don't match observations"

        self.TWOPI = 2.0*np.pi
        self.variance = tf.Variable(variance,dtype=tf.float64,name='variance')

        CollapsedMixture.__init__(self, num_data, num_clusters, prior_Z, alpha)

        if kernels is None:
            kernels = []
            for i in range(self.num_clusters):
            #for i in range(10):
                kernels.append(GPflow.kernels.RBF(input_dim=1))
        self.kern = GPflow.param.ParamList(kernels)

        self.YYT = GPflow.param.DataHolder(np.dot(Y, Y.T))

    def build_likelihood(self):
        """
        Compute the lower bound on the marginal likelihood (conditioned on the
        GP hyper parameters).
        """
        GP_bound = 0.0
        phi = tf.nn.softmax(self.logphi)

        # if len(self.kern._list) < self.num_clusters:
            # self.kern.append(self.kern[-1].copy())

        # if len(self.kern) > self.num_clusters:
            # self.kern = self.kern[:self.num_clusters]

        #for i, kern in enumerate(self.kern):
        for i in range(self.num_clusters):
            K = self.kern[i].K(self.X)
            B_inv = tf.diag(1. / ((phi[:, i] + 1e-6) / self.variance))

            # Make more stable using cholesky factorization:
            # Bi, LB, LBi, Blogdet = pdinv(K+B_inv)
            LB = tf.cholesky(K + B_inv + GPflow.tf_wraps.eye(self.num_data) * 1e-6)
            Blogdet = 2.*tf.reduce_sum(tf.log(tf.diag_part(LB)))
            # Data fit
            # GP_bound -= .5 * dpotrs(LB, self.YYT)[0].trace()
            GP_bound -= 0.5 * tf.trace(tf.matrix_triangular_solve(LB, self.YYT))
            # Penalty
            # GP_bound += -0.5 * np.linalg.slogdet(K + B_inv)[1]
            GP_bound -= 0.5 * Blogdet

            # Constant, weighted by  model assignment per point
            # GP_bound += -0.5 * (self.phi[:, i] * np.log(2 * np.pi * self.variance)).sum()
            # GP_bound -= .5*self.D * np.einsum('j,j->',self.phi[:, i], np.log(2*np.pi*self.variance))
            GP_bound -= 0.5*self.D*tf.reduce_sum(tf.mul(phi[:, i],tf.log(self.TWOPI*self.variance)))

        return GP_bound - self.build_KL_Z()


    def predict(self, Xnew, i, phi=None):
        """ Predictive mean for a given component
        """
        if phi is None:
            phi = tf.nn.softmax(self.logphi)
        kern = self.kern[i]
        K = kern.K(self.X)
        kx = kern.K(self.X, Xnew)

        # Notation is from pages 3-5 of M. Lazaro-Gredilla et al. / Overlapping Mixtures of 
        #Gaussian Processes for the data association problem (used original equations - not
        # "stable" ones)
        # Predict mean
        B_inv = tf.diag(1. / ((phi[:, i] + 1e-6) / self.variance))
        LB = tf.cholesky(K + B_inv + GPflow.tf_wraps.eye(self.num_data)*1e-6)
        mu = tf.matmul(tf.transpose(kx), tf.cholesky_solve(LB, self.Y))

        # Predict variance
        kxx = kern.K(Xnew, Xnew)
        va = self.variance + kxx - tf.matmul(tf.transpose(kx), tf.cholesky_solve(LB, kx))

        return mu, tf.diag_part(va)


    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_components(self, Xnew):
        """The predictive density under each component"""
        phi = tf.nn.softmax(self.logphi)
        mus = []
        vas = []
        # For now, the number of kernels is the number of clusters
        #for i in range(len(self.kern)):
        for i in range(self.num_clusters):
            mu, va = self.predict(Xnew, i, phi)
            mus.append(mu)
            vas.append(va)

        return tf.transpose(tf.squeeze(tf.pack(mus),[2])), tf.transpose(tf.pack(vas))


    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def sample(self, Xnew, gp=0, size=10, full_cov=True):
        ''' Sample the posterior of a component
        '''
        mu, va = self.predict(Xnew, gp)

        samples = []
        for i in range(mu.shape[1]):
            if full_cov:
                smp = np.random.multivariate_normal(mean=mu[:, i], cov=va, size=size)
            else:
                smp = np.random.multivariate_normal(mean=mu[:, i], cov=np.diag(np.diag(va)), size=size)

            samples.append(smp)

        return np.stack(samples, -1)
    
    @GPflow.param.AutoFlow()
    def get_variance(self):
        return self.variance

    @GPflow.param.AutoFlow()
    def vb_bound_grad_natgrad(self):
        """
        Natural Gradients of the bound with respect to the variational
        parameters controlling assignment of the data to clusters
        """
        bound = self.build_likelihood()
        grad, = tf.gradients(bound, tf.concat(0,[self.logphi,tf.expand_dims(self.variance,0)]))
        natgrad = grad[:-1] / tf.nn.softmax(self.logphi)
        grad, natgrad = tf.clip_by_value(grad, -100, 100), tf.clip_by_value(natgrad, -100, 100)
        return bound, tf.reshape(grad, [-1]), tf.reshape(natgrad, [-1])
