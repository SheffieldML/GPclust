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
        assert X.shape[0] == self.D, "input data don't match observations"

        self.variance = variance

        CollapsedMixture.__init__(self, num_data, num_clusters, prior_Z, alpha)

        if kernels == None:
            self.kern = []
            for i in range(num_clusters):
                self.kern.append(GPflow.kernels.RBF(input_dim=1))
        else:
            self.kern = kernels

        self.YYT = GPflow.param.DataHolder(np.dot(Y, Y.T))

    def build_likelihood(self):
        """
        Compute the lower bound on the marginal likelihood (conditioned on the
        GP hyper parameters).
        """
        GP_bound = 0.0
        phi = tf.nn.softmax(self.logphi)

        if len(self.kern) < self.num_clusters:
            self.kern.append(self.kern[-1].copy())

        if len(self.kern) > self.num_clusters:
            self.kern = self.kern[:self.num_clusters]

        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            B_inv = tf.diag(1. / ((phi[:, i] + 1e-6) / self.variance))

            # Make more stable using cholesky factorization:
            #Bi, LB, LBi, Blogdet = pdinv(K+B_inv)
            LB = tf.cholesky(K + B_inv + GPflow.tf_wraps.eye(self.D) * 1e-6)
            Blogdet = 2.*tf.reduce_sum(tf.log(tf.diag_part(LB)))
            # Data fit
            #GP_bound -= .5 * dpotrs(LB, self.YYT)[0].trace()
            GP_bound -= 0.5 * tf.trace(tf.matrix_triangular_solve(LB,self.YYT))
            # Penalty
            # GP_bound += -0.5 * np.linalg.slogdet(K + B_inv)[1]
            GP_bound -= 0.5 * Blogdet

            # Constant, weighted by  model assignment per point
            #GP_bound += -0.5 * (self.phi[:, i] * np.log(2 * np.pi * self.variance)).sum()
            #GP_bound -= .5*self.D * np.einsum('j,j->',self.phi[:, i], np.log(2*np.pi*self.variance))
            GP_bound -= 0.5*self.D*tf.reduce_sum(tf.mul(phi[:,i],np.log(2.*np.pi*self.variance)))

        return  GP_bound - self.build_KL_Z()

    def predict(self, Xnew, i):
        """ Predictive mean for a given component
        """
        phi = tf.nn.softmax(self.logphi)
        kern = self.kern[i]
        K = kern.K(self.X)
        kx = kern.K(self.X, Xnew)

        # Predict mean
        # This works but should Cholesky for stability
        #B_inv = np.diag(1. / (self.phi[:, i] / self.variance))
        B_inv = tf.diag(1. / ((phi[:, i] + 1e-6) / self.variance))
        LB = tf.cholesky(K + B_inv + GPflow.tf_wraps.eye(self.D) * 1e-6)
        #K_B_inv = pdinv(K + B_inv)[0]
        K_B_inv = tf.matrix_triangular_solve(LB, GPflow.tf_wraps.eye(self.D), lower=True)
        mu = tf.matmul(tf.transpose(kx),tf.matmul(K_B_inv, self.Y))

        # Predict variance
        kxx = kern.K(Xnew, Xnew)
        va = self.variance + kxx - tf.matmul(tf.transpose(kx),tf.matmul(K_B_inv, kx))

        return mu, va

    def predict_components(self, Xnew):
        """The predictive density under each component"""
        mus = []
        vas = []
        for i in range(len(self.kern)):
            mu, va = self.predict(Xnew, i)
            mus.append(mu)
            vas.append(va)

        return np.array(mus)[:, :, 0].T, np.array(vas)[:, :, 0].T
    
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

