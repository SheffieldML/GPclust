# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from collapsed_mixture import CollapsedMixture
import GPflow
#from GPy.util.linalg import mdot, pdinv, backsub_both_sides, dpotrs, jitchol, dtrtrs
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
    Y        - A np.array of the observed time-course values: each row contains a time series, each column represents a unique time point
    kernF    - A GPflow kernel to model the mean function of each cluster
    kernY    - A GPflow kernel to model the deviation of each of the time courses from the mean of the cluster
    alpha    - The a priori Dirichlet concentrationn parameter (default 1.)
    prior_Z  - Either 'symmetric' or 'dp', specifies whether to use a symmetric dirichelt prior for the clusters, or a (truncated) Dirichlet Process.
    name     - A convenient string for printing the model (default MOHGP)

    """
    def __init__(self, X, kernF, kernY, Y, num_clusters=2, alpha=1., prior_Z='symmetric', name='MOHGP'):

        num_data, self.D = Y.shape
        self.Y = GPflow.param.DataHolder(Y, on_shape_change='pass')
        self.Y.name='Y'
        self.X = GPflow.param.DataHolder(X, on_shape_change='pass')
        assert X.shape[0]==self.D, "input data don't match observations"

        CollapsedMixture.__init__(self, num_data, num_clusters, prior_Z, alpha, name)
        # CollapsedMixture returns the initialization of the variational parameters, so let's 
        # convert them to a tensorflow DataHolder
        self.tfphi = GPflow.param.DataHolder(self.phi)
        self.tfphi_hat = GPflow.param.DataHolder(self.phi_hat)

        self.kernF = kernF
        self.kernY = kernY
        self.LOG2PI = np.log(2.*np.pi)
        #initialize kernels
        with self.tf_mode():
            self.Sf = self.kernF.K(self.X)
            self.Sy = self.kernY.K(self.X)

            self.Sy_chol = tf.cholesky(self.Sy + GPflow.tf_hacks.eye(self.D)*1e-6)
            self.Sy_logdet = 2.*tf.reduce_sum(tf.log(tf.diag_part(self.Sy_chol)))
            self.Sy_inv = tf.matrix_triangular_solve(self.Sy_chol,GPflow.tf_hacks.eye(self.D), lower=True)

        #Computations that can be done outside the optimisation loop
        self.YYT = GPflow.param.DataHolder(Y[:,:,np.newaxis]*Y[:,np.newaxis,:])
        self.YTY = GPflow.param.DataHolder(np.dot(Y.T,Y))

        self.do_computations()


    def parameters_changed(self):
        """ Set the kernel parameters. Note that the variational parameters are handled separately."""
        with self.tf_mode():
            self.Sf = self.kernF.K(self.X)
            self.Sy = self.kernY.K(self.X)

            self.Sy_chol = tf.cholesky(self.Sy + GPflow.tf_hacks.eye(self.D)*1e-6)
            self.Sy_logdet = 2.*tf.reduce_sum(tf.log(tf.diag_part(self.Sy_chol)))
            self.Sy_inv = tf.matrix_triangular_solve(self.Sy_chol,GPflow.tf_hacks.eye(self.D), lower=True)

        #update everything
        self.do_computations()


    def do_computations(self):
        """
        Here we do all the computations that are required whenever the kernels
        or the variational parameters are changed.
        """
        #sufficient stats.
        with self.tf_mode():
            self.tfphi = GPflow.param.DataHolder(self.phi)
            self.tfphi_hat = GPflow.param.DataHolder(self.phi_hat)
            
            self.ybark = tf.transpose(tf.matmul(tf.transpose(self.tfphi),self.Y))

            # compute posterior variances of each cluster (lambda_inv)
            #tmp = backsub_both_sides(self.Sy_chol, self.Sf, transpose='right')
            tmp1 = tf.matrix_triangular_solve(self.Sy_chol, self.Sf, lower=True)
            tmp = tf.transpose(tf.matrix_triangular_solve(self.Sy_chol,tf.transpose(tmp1),lower=True))

            # a num_clustersxDxD series of identity matrices
            Id = tf.tile(tf.expand_dims(GPflow.tf_hacks.eye(self.D), 0), [self.num_clusters, 1, 1])  
            self.Cs =  Id * tf.reshape(self.tfphi_hat, [self.num_clusters, 1, 1])
            self.C_chols = tf.batch_cholesky(self.Cs)
    
            self.log_det_diff_sum = 0. 

            # ----- Need to fix below 2 lines
            #for i in range(self.num_clusters):
            #    self.log_det_diff_sum += 2.*tf.reduce_sum(tf.log(tf.diag_part(self.C_chols[i])))

            tmp = tf.batch_matrix_triangular_solve(self.C_chols,tf.tile(tf.expand_dims(self.Sy_chol,0),[self.num_clusters,1,1]),lower=True)
            tmp2 = tf.batch_matmul(tf.batch_matrix_transpose(tmp),tmp)
            self.Lambda_inv = (tf.tile(tf.expand_dims(self.Sy, 0),[self.num_clusters,1,1]) - tmp2) / tf.reshape(self.tfphi_hat, [-1, 1, 1]) 
                
            # It appears we only need self.muk for the vb_grad_natgrad function.  Since tf will take care of the derivatives, 
            #do we need this anymore? -- NO
            #self.muk[i] = tf.mul(self.Lambda_inv,tf.transpose(self.Syi_ybark)[:,:,None]) # ???

            #posterior mean and other useful quantities
            self.Syi_ybark = tf.matrix_triangular_solve(self.Sy_chol,self.ybark, lower=True)
            self.Syi_ybarkybarkT_Syi = tf.mul(tf.expand_dims(tf.transpose(self.Syi_ybark),1),tf.expand_dims(tf.transpose(self.Syi_ybark),2))

    def log_likelihood(self):
        # tensorflow edition of bound()
        # gets picked up automatically by tensorflow due to naming convention
        with self.tf_mode():
            return -0.5 * ( self.num_data * self.D * self.LOG2PI \
               + self.log_det_diff_sum + self.num_data* self.Sy_logdet )#\
               #+ tf.reduce_sum(tf.mul(self.YTY,self.Sy_inv)) ) #\
               #+ 0.5 * tf.reduce_sum(tf.mul(self.Syi_ybarkybarkT_Syi,self.Lambda_inv)) #\
               #+ self.mixing_prop_bound() + self.H

    def bound(self):
        """
        Compute the lower bound on the marginal likelihood (conditioned on the
        GP hyper parameters).
        """
        ll = self.log_likelihood()
        feed_dict = self.get_feed_dict()
        for w in feed_dict.keys():
            print w.name
        return self._session.run(ll,feed_dict=feed_dict)

    #def optimize_vb(self,args):
    #    compile_vb()
    #    self.optimize(args)

    def vb_grad_natgrad(self):
        """
        Natural Gradients of the bound with respect to phi, the variational
        parameters controlling assignment of the data to clusters
        """
        with self.tf_mode():
            bound = self.log_likelihood()
            grad_phi = tf.gradients(bound,self.tfphi)
            # replicate np.sum(self.phi*grad_phi,1)
            A = tf.matmul( tf.mul(self.tfphi,grad_phi),tf.ones(tf.Tensor.get_shape(self.tfphi)[0]) )
            natgrad = tf.sub(grad_phi,tf.expand_dims(A,1))
            grad = tf.mul(natgrad,self.tfphi) 
            return self._session.run([bound,grad,natgrad])
