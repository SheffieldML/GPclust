# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import pylab as pb
from scipy import optimize, linalg
from utilities import pdinv, softmax, multiple_pdinv, blockdiag, lngammad, ln_dirichlet_C, safe_GP_inv
from scipy.special import gammaln, digamma
from scipy import stats
from col_vb import col_vb

class GPRN(col_vb):
    def __init__(self, X, Y, K=2, kerns_W=None,kern_F=None):
        self.X = X
        self.Y = Y
        self.N, self.Q = X.shape
        N,self.D = self.Y.shape
        assert N==self.N, "data don't match"
        self.K = K

        #sort out the kernels. TODO: better defaults
        if kerns_W is None:
            kerns_W = [GPy.kern.rbf(self.Q) for k in xrange(self.K)]
        else:
            assert len(kerns_W) == self.K
        self.kerns_W = kerns_W
        if kerns_F is None:
            kerns_F = [[GPy.kern.rbf(self.Q) for k in range(self.K)] for d in xrange(self.D)]
        else:
            assert len(kerns_F) == self.D
            for d in range(self.D):
                assert len(kerns_F[d]) == self.K
        self.kerns_F = kerns_F

        #initialize vb params
        #TODO

        #initialize big memory things used in computation
        self.Atilde = np.zeros((self.N*self.K, self.N*self.K))
        self.Ystacked = np.vstack([Y for k in range(self.K])


    def _get_params(self)
        return np.hstack([k._get_params_transformed() for k in self.kerns_W + self.kerns_F])

    def _set_params(self,param):
        count = 0
        for k in self.kerns_W+self.kerns_F:
            k._set_params_transformed(param[count:count + k.Nparam_transformed])
            count += k.Nparam_transformed
        self._computations

    def set_vb_param(self,param):
        #TODO: could we use the Opper-Archabeau parameterisation? If so, what are the canonical/expectation parameters?
        self.q_w_canonical_flat = param.copy()
        self.q_w_precisions = -2.*self.q_w_canonical_flat[self.N*self.K:].reshape(self.K,self.N,self.N)
        self.q_w_covariances, self.q_w_cholinvs, self.q_w_chols, self.q_w_logdets = zip(pdinv(P) for P in self.q_w_precisions)
        self.q_w_logdets = -np.array(self.q_w_logdets) # make these the log det of the covariance, not precision
        self.q_w_means = [linalg.laplack.flapack.dpotrs(L,np.asfortranarray(X),lower=1)[0].flatten() for L,X in zip(self.q_w_cholinvs,self.q_w_canonical_flat[:self.N*self.K].reshape(self.K,self.N,1))]
        self.q_w_expectations = [self.q_w_means, [np.dot(m,m.T) + S for m,S in zip(self.q_w_means,self.q_w_covarainces)] ]

    def get_vb_param(self):
        return self.q_w_canonical_flat

    def do_computations(self):
        #compute and decompose all covariance matrices#
        self.K_w = [k(self.X) for k in self.kerns_W]
        self.K_f = [[k(self.X) for k in tmp] for tmp in self.kerns_F]
        self.Ki_w, self.L_w, self.Li_w, self.logdet_w = zip(*[pdinv(K) for K in self.K_w])

        #TODO: do we need full pdinv of the F covariances?

        #build the giant matrix A_tilde
        for k in range(self.K):
            for kk in range(self.K):
                i = np.arange(self.N)*k
                j = np.arange(self.N)*kk
                self.Atilde[i,j] += self.q_w_means[k]*self.q_w_means[kk]*self.beta
                if k==kk:
                    self.Atilde[i,j] += np.diag(self.q_w_covariances[k])*self.beta

        #construct the effective target data ytilde
        self.Ytilde = self.Ystacked*np.vstack(self.q_w_means)

        #compute and invert posterior precision matrices (!)
        self.posterior_precisons = [self.Atilde.copy() for d in range(self.D)]
        for d in range(self.D):
            for k in range(self.K):
                self.posterior_precisons[d][k*self.N:(k+1)*self.N.k(self.N:(k+1)*self.N)] += self.kerns_F[d][k].K(self.X)
        self.posterior_choleskies = [linalg.lapack.flapack.dpotrf(np.asfortranarray(PP.T),lower=1)[0] for PP in self.posterior_precisons]
        self.posterior_covariances = [linalg.laplack.flapack.dpotri(L,lower=1)[0] for L in self.posterior_choleskies]
        self.posterior_logdets = [2.*np.sum(np.log(np.diag(L))) for L in self.posterior_choleskies]

        #compute posterior means
        self.posterior_means = [self.beta*linalg.lapack.flapack.dpotrs(L,np.asfortranarray(y_tilde_d.reshape(self.N,1)),lower=1) for L,y_tilde_d in zip(self.posterior_choleskies,self.Ytilde.T)]

        #compute intermediary derivatives
        self.dL_dAtilde = np.zeros_like(self.A_tilde)
        for d in range(self.D):
            self.dL_dAtilde + -0.5*(self.posterior_covariances[d])
            tmp = linalg.lapack.flapack.dpotrs(self.posterior_choleskies[d], self.Ytilde[:,d:d+1])
            self.dL_dAtilde += -0.5*np.dot(tmp.tmp.T)

    def bound(self):
        A = -0.5*self.N*self.D*(np.log(2*np.pi) - np.log(self.beta)) + 0.5*self.N*self.Q
        B = -0.5*self.beta*np.sum(np.square(self.Y))
        C = -0.5*np.sum(self.K_w_logdets) - 0.5*np.sum(self.K_f_logdets) + 0.5*np.sum(self.q_w_logdets) - 0.5*np.sum(self.posterior_logdets)
        D = -0.5*np.sum([np.sum(Ki_k*q_w_expectation_k) for Ki_k,q_w_expectation_k in zip(self.K_w_invs,self.q_w_expectations[1])])
        tmp = [linalg.lapack.flapack.dtrtrs(L,np.asfortranarray(y_tilde_d.reshape(self.N,1)),lower=1) for L,y_tilde_d in zip(self.posterior_choleskies,self.Ytilde.T)]
        E = 0.5*self.beta**2*np.sum([np.sum(np.square(tmp_i)) for tmp_i in tmp])
        return A+B+C+D+E

    def vb_grad_natgrad(self):
        raise NotImplementedError

    def predict(self, Xnew):
        raise NotImplementedError

    def plot(self, newfig=True):
        raise NotImplementedError
