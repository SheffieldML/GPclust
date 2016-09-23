# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import GPflow
import tensorflow as tf
from .collapsed_mixture import CollapsedMixture

from tf_utilities import tf_multiple_pdinv, lngammad, tensor_lngammad
    
class MOG(CollapsedMixture):
    """
    A Mixture of Gaussians, using the fast variational framework

    Arguments
    =========
    X - a np.array of the observed data: each row contains one datum.
    num_clusters - the number of clusters (or initial number of clusters in the Dirichlet Process case)
    alpha - the A priori dirichlet concentrationn parameter (default 1.)
    prior_Z  - either 'symmetric' or 'DP', specifies whether to use a symmetric Dirichlet prior for the clusters, or a (truncated) Dirichlet process.
    name - a convenient string for printing the model (default MOG)

    Optional arguments for the parameters of the Gaussian-Wishart priors on the clusters
    prior_m - prior mean (defaults to mean of the data)
    prior_kappa - prior connectivity (default 1e-6)
    prior_S - prior Wishart covariance (defaults to 1e-3 * I)
    prior_v - prior Wishart degrees of freedom (defaults to dimension of the problem +1.)

    """
    def __init__(self, X, num_clusters=2, prior_Z='symmetric', alpha=1., prior_m=None, prior_kappa=1e-6, prior_S=None, prior_v=None, name='MOG'):
        self.num_data, self.D = X.shape
        self.X = GPflow.param.DataHolder(X, on_shape_change='pass')
        self.LOGPI = np.log(np.pi)

        # store the prior cluster parameters
        self.m0 = X.mean(0) if prior_m is None else prior_m
        self.k0 = prior_kappa
        self.S0 = GPflow.tf_hacks.eye(self.D)*1.e-3 if prior_S is None else prior_S
        self.v0 = prior_v or tf.to_double(self.D + 1.)

        #precomputed stuff
        self.k0m0m0T = GPflow.param.DataHolder(self.k0*self.m0[:,np.newaxis]*self.m0[np.newaxis,:])
        XXT = X[:,:,np.newaxis]*X[:,np.newaxis,:]
        self.reshapeXXT = GPflow.param.DataHolder(np.reshape(XXT,(self.num_data,self.D*self.D)).T)
        self.S0_halflogdet = tf.reduce_sum(tf.log(tf.sqrt(tf.diag(tf.cholesky(self.S0)))))
  
        CollapsedMixture.__init__(self, self.num_data, num_clusters, prior_Z, alpha)

    def build_likelihood(self):
        """Compute the lower bound on the model evidence.  """
        phi = tf.nn.softmax(self.logphi)
        phi_hat = tf.reduce_sum(phi, 0)

        self.kNs = tf.add(phi_hat,self.k0)
        self.vNs = tf.add(phi_hat,self.v0)

        # ---------------------------------------------------------------------------
        #Xsumk = np.tensordot(self.X,phi,((0),(0))) #D x num_clusters
        # X is num_data x D and phi is num_data x num_clusters, so Xsumk is X.T*phi
        Xsumk = tf.matmul(tf.transpose(self.X),phi)
        #Ck = np.tensordot(phi, self.XXT,((0),(0))).T# D x D x num_clusters
        # Ck is reshapeXXT*phi
        Ck = tf.reshape(tf.matmul(self.reshapeXXT,phi),tf.pack([self.D,self.D,self.num_clusters]))
        # ---------------------------------------------------------------------------

        self.mun = tf.div((self.k0*tf.expand_dims(self.m0,1) + Xsumk),tf.expand_dims(self.kNs,0)) # D x K
        munmunT = tf.mul(tf.expand_dims(self.mun,1),tf.expand_dims(self.mun,0))
        Sns = tf.expand_dims(self.S0,2) + Ck + tf.expand_dims(self.k0m0m0T,2) -\
            tf.mul(tf.expand_dims(tf.expand_dims(self.kNs,0),0),munmunT)

        self.Sns_inv , self.Sns_halflogdet = tf_multiple_pdinv(Sns)
        print self.Sns_inv.get_shape(), self.Sns_halflogdet.get_shape(), phi.get_shape(), self.vNs.get_shape()

        return -0.5*self.D*tf.reduce_sum(tf.log(tf.div(self.kNs,self.k0)))\
            +self.num_clusters*self.v0*self.S0_halflogdet - tf.reduce_sum(tf.mul(self.vNs,self.Sns_halflogdet))\
            +tf.reduce_sum(tensor_lngammad(self.vNs, self.D)) - self.num_clusters*lngammad(self.v0, self.D)\
            +self.build_KL_Z() - 0.5*self.num_data*self.D*self.LOGPI


    def predict_components(self, Xnew):
        """
        The predictive density under each component at Xnew
        """
        phi = tf.nn.softmax(self.logphi)
        phi_hat = tf.reduce_sum(phi, 0)

        self.kNs = tf.add(phi_hat,self.k0)
        self.vNs = tf.add(phi_hat,self.v0)

        Xsumk = tf.matmul(tf.transpose(self.X),phi) # D x num_clusters
        Ck = tf.reshape(tf.matmul(self.reshapeXXT,phi),tf.pack([self.D,self.D,self.num_clusters]))

        self.mun = tf.div((self.k0*tf.expand_dims(self.m0,1) + Xsumk),tf.expand_dims(self.kNs,0)) # D x num _clusters
        munmunT = tf.mul(tf.expand_dims(self.mun,1),tf.expand_dims(self.mun,0))
        Sns = tf.expand_dims(self.S0,2) + Ck + tf.expand_dims(self.k0m0m0T,2) -\
            tf.mul(tf.expand_dims(tf.expand_dims(self.kNs,0),0),munmunT)

        self.Sns_inv , self.Sns_halflogdet = tf_multiple_pdinv(Sns)

        Dist = tf.sub(tf.expand_dims(Xnew,2),tf.expand_dims(self.mun,0)) # Nnew x D x num_clusters
        # Tensorflow does not support the following broadcast (Nnew,D,1,num_cluster) * (1,D,D,num_clusters)
        #tmp = tf.reduce_sum(tf.mul(tf.expand_dims(Dist,2),tf.expand_dims(self.Sns_inv,0)),1)
        # So we will tile self.Sns_inv Nnew times to do the multiplication
        h = tf.tile(tf.expand_dims(self.Sns_inv,0),tf.pack([tf.get_shape(Dist)[0],1,1,1]))
        tmp = tf.reduce_sum(tf.mul(tf.expand_dims(Dist,2),h),1)
        mahalanobis = tf.reduce_sum(tf.mul(tmp,Dist), 1)/(self.kNs+1.)*self.kNs*(self.vNs-self.D+1.)
        halflndetSigma = self.Sns_halflogdet + 0.5*self.D*tf.log((self.kNs+1.)/(self.kNs*(self.vNs-self.D+1.)))

        Z  = tf.lgamma(0.5*(tf.expand_dims(self.vNs,0)+1.))-tf.lgamma(0.5*(tf.expand_dims(self.vNs,0)-self.D+1.))\
            -(0.5*self.D)*(tf.log(tf.expand_dims(self.vNs,0)-self.D+1.) + self.LOGPI)\
            - halflndetSigma#\
            - (0.5*(tf.expand_dims(self.vNs,0)+1.))*tf.log(1.+mahalanobis/(tf.expand_dims(self.vNs,0)-self.D+1.))

        return tf.exp(Z)

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict(self, Xnew):
        """The predictive density of the model at Xnew"""
        Z = self.predict_components(Xnew)
        #calculate the weights for each component
        phi = tf.nn.softmax(self.logphi)
        phi_hat = tf.reduce_sum(phi, 0)
        pie = tf.add(phi_hat,self.alpha)
        pie = tf.div(pie,tf.reduce_sum(pie))
        return tf.reduce_sum(tf.mul(Z,tf.expand_dims(pie,0)),1)
