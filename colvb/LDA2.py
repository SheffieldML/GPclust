# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
#import pylab as pb
import matplotlib as mlp
#mlp.use('cairo.pdf')
from scipy.special import gammaln, digamma
from scipy import sparse
from col_vb import col_vb
from weave_fns import LDA_mult

class LDA2(col_vb):
    def __init__(self, countdata, K, alpha_0=1., lambda_0=1., alpha_init=None, lambda_init=None):
        self.countdata = countdata # D x V
        self.K = K
        self.alpha_0 = alpha_0
        self.lambda_0 = lambda_0
        self.D,self.V = countdata.shape

        #sensible inits if none are passed
        if alpha_init is None:
            alpha_init = np.random.rand(self.D,self.K)
        else:
            assert alpha_init.shape == (self.D, self.K)

        if lambda_init is None:
            lambda_init = np.random.rand(self.K,self.V)
        else:
            assert lambda_init.shape == (self.K, self.V)

        self.set_vb_param(np.hstack((alpha_init.flatten(),lambda_init.flatten())))
        col_vb.__init__(self)

    def set_vb_param(self,p):
        self.vb_param = p.copy()
        self.alphas = p[:self.D*self.K].reshape(self.D,self.K)
        logtheta = digamma(self.alphas) - digamma(self.alphas.sum(1))[:,None]
        self.theta_hat = np.exp(logtheta) # D x K

        self.lambdas = p[self.D*self.K:].reshape(self.K,self.V)
        logbeta = digamma(self.lambdas) - digamma(self.lambdas.sum(1))[:,None]
        self.beta_hat = np.exp(logbeta) # K x V

        self.thetabeta = self.theta_hat[:,:,None]*self.beta_hat[None,:,:] # D x K x V (!)
        self.thetabeta_sumk = self.thetabeta.sum(1) # D x V

        #TODO: this aint right
        #KL_alpha = np.sum((self.alphas-self.alpha_0)*logtheta) - ?
        #KL_lambda = np.sum((self.lambdas-self.lambda_0)*logbeta)

        self.KL = KL_alpha + KL_lambda

    def get_vb_param(self):
        return self.vb_param


    def bound(self):
        return np.sum(self.countdata*np.log(self.thetabeta.sum(1))) - self.KL

    def vb_grad_natgrad(self):
        #natgrad in alpha
        d_logtheta = np.sum((self.countdata/self.thetabeta_sumk)[:,None,:]*self.thetabeta,2) + self.alpha_0 - self.alphas

        #natgrad in lambda
        d_logbeta = np.sum((self.countdata/self.thetabeta_sumk)[:,None,:]*self.thetabeta,0) + self.lambda_0 - self.lambdas

        #grad in lambda TODO
        #d_alphas = np.ones_like(self.alphas)
        #d_lambdas = np.ones_like(self.lambdas)
        d_alphas = d_logtheta.copy()
        d_lambdas = d_logbeta.copy()

        if hasattr(self,'debug'):
            d_logbeta *= 0

        return np.hstack((d_alphas.flatten(), d_lambdas.flatten())), np.hstack((d_logtheta.flatten(), d_logbeta.flatten())),





