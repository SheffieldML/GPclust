# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import GPy
import time
import sys #for flushing

class CollapsedVB(GPy.core.Model):
    """
    A base class for collapsed variational models, using the GPy framework for
    non-variational parameters.

    Optimisation of the (collapsed) variational paremters is performed by
    Riemannian conjugate-gradient ascent, interleaved with optimisation of the
    non-variational parameters

    This class specifies a scheme for implementing the model, as well as
    providing the optimisation routine.
    """

    def __init__(self, name):
        """"""
        GPy.core.Model.__init__(self, name)

        self.hyperparam_interval=50

        self.default_method = 'HS'
        self.hyperparam_opt_args = {
                'max_iters':20,
                'messages':1}

    def randomize(self):
        GPy.core.Model.randomize(self)
        self.set_vb_param(np.random.randn(self.get_vb_param().size))

    def get_vb_param(self):
        """Return a vector of variational parameters"""
        raise NotImplementedError

    def set_vb_param(self,x):
        """Expand a vector of variational parameters into the model"""
        raise NotImplementedError

    def bound(self):
        """Returns the lower bound on the marginal likelihood"""
        raise NotImplementedError

    def vb_grad_natgrad(self):
        """Returns the gradient and natural gradient of the variational parameters"""

    def log_likelihood(self):
        """
        In optimising the non variational (e.g. kernel) parameters, use the
        bound as a proxy for the likelihood
        """
        return self.bound()

    def optimize(self, method='HS', maxiter=500, ftol=1e-6, gtol=1e-6, step_length=1., callback=None, verbose=True):
        """
        Optimize the model.

        The strategy is to run conjugate natural gradients on the variational
        parameters, interleaved with gradient based optimization of any
        non-variational parameters. self.hyperparam_interval dictates how
        often this happens.

        Arguments
        ---------
        :method: ['FR', 'PR','HS','steepest'] -- conjugate gradient method
        :maxiter: int
        :ftol: float
        :gtol: float
        :step_length: float

        """

        assert method in ['FR', 'PR','HS','steepest'], 'invalid conjugate gradient method specified.'

        iteration = 0
        bound_old = self.bound()
        searchDir_old = 0.
        while True:

            if callback is not None:
                callback()

            grad,natgrad = self.vb_grad_natgrad()
            grad,natgrad = -grad,-natgrad
            squareNorm = np.dot(natgrad,grad) # used to monitor convergence

            #find search direction
            if (method=='steepest') or not iteration:
                beta = 0
            elif (method=='PR'):
                beta = np.dot((natgrad-natgrad_old),grad)/squareNorm_old
            elif (method=='FR'):
                beta = squareNorm/squareNorm_old
            elif (method=='HS'):
                beta = np.dot((natgrad-natgrad_old),grad)/np.dot((natgrad-natgrad_old),grad_old)
            if np.isnan(beta) or (beta < 0.):
                beta = 0.
            searchDir = -natgrad + beta*searchDir_old

            #try a conjugate step
            phi_old = self.get_vb_param().copy()
            self.set_vb_param(phi_old + step_length*searchDir)
            bound = self.bound()
            iteration += 1

            #make sure there's an increase in the bound, else revert to steepest, which is guaranteed to increase the bound (it's the same as VBEM)
            if bound<bound_old:
                searchDir = -natgrad
                self.set_vb_param(phi_old + step_length*searchDir)
                bound = self.bound()
                iteration += 1


            if verbose:
                print '\riteration '+str(iteration)+' bound='+str(bound) + ' grad='+str(squareNorm) + ', beta='+str(beta),
                sys.stdout.flush()

            # converged yet? try the parameters if so
            if np.abs(bound-bound_old)<=ftol:
                if verbose:print 'vb converged (ftol)'
                if self.optimize_parameters()<1e-1:
                    break
            if squareNorm<=gtol:
                if verbose:print 'vb converged (gtol)'
                if self.optimize_parameters()<1e-1:
                    break
            if iteration>=maxiter:
                if verbose:print 'maxiter exceeded'
                break

            #store essentials of previous iteration
            natgrad_old = natgrad.copy() # copy: better safe than sorry.
            grad_old = grad.copy()
            searchDir_old = searchDir.copy()
            squareNorm_old = squareNorm

            # hyper param_optimisation
            if (iteration >1) and not (iteration%self.hyperparam_interval):
                self.optimize_parameters()

            bound_old = bound



    def optimize_parameters(self):
        """
        Optimises the model parameters (non variational parameters)
        Returns the increment in the bound acheived
        """
        if self.size:
            start = self.bound()
            GPy.core.model.Model.optimize(self,**self.hyperparam_opt_args)
            return self.bound()-start






