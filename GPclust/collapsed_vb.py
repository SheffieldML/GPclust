# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import gpflow
import sys  # for flushing
from numpy.linalg.linalg import LinAlgError


class LinAlgWarning(Warning):
    pass


class CollapsedVB(gpflow.model.Model):
    """
    A base class for collapsed variational models, using the gpflow framework for
    non-variational parameters.

    Optimisation of the (collapsed) variational parameters is performed by
    Riemannian conjugate-gradient ascent, interleaved with optimization (in tensorfow)
    of the non-variational parameters

    This class specifies a scheme for implementing the model, as well as
    providing the optimisation routine.
    """

    def __init__(self):
        """"""
        gpflow.model.Model.__init__(self)

        # settings for optimizing hyper parameters
        self.hyperparam_interval = 50
        self.default_method = 'HS'
        self.hyperparam_opt_args = dict(maxiter=20, disp=True)

    def randomize(self):
        self.set_vb_param(np.random.randn(self.get_vb_param().size))

    def get_vb_param(self):
        """Return a vector of variational parameters"""
        raise NotImplementedError

    def set_vb_param(self, x):
        """Expand a vector of variational parameters into the model"""
        raise NotImplementedError

    def vb_bound_grad_natgrad(self):
        """Returns the bound, gradient and natural gradient of the variational parameters"""
        raise NotImplementedError

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

        assert method in ['FR', 'PR', 'HS', 'steepest'], 'invalid conjugate gradient method specified.'

        iteration = 0
        bound_old, grad_old, natgrad_old = self.vb_bound_grad_natgrad()
        squareNorm_old = np.dot(natgrad_old, grad_old)  # used to monitor convergence
        searchDir_old = 0.
        iteration_failed = False
        while True:

            if callback is not None:
                callback()

            bound, grad, natgrad = self.vb_bound_grad_natgrad()
            grad, natgrad = -grad, -natgrad
            if np.any(np.isnan(natgrad)):
                stop
            if np.any(np.isnan(grad)):
                stop
            squareNorm = np.dot(natgrad, grad)  # used to monitor convergence

            # find search direction
            if (method == 'steepest') or not iteration:
                beta = 0
            elif (method == 'PR'):
                beta = np.dot((natgrad-natgrad_old), grad) / squareNorm_old
            elif (method == 'FR'):
                beta = squareNorm/squareNorm_old
            elif (method == 'HS'):
                beta = np.dot((natgrad-natgrad_old), grad) / (np.dot((natgrad-natgrad_old), grad_old) + 1e-6 )
            if np.isnan(beta) or (beta < 0.):
                beta = 0.
            searchDir = -natgrad + beta * searchDir_old

            # Try a conjugate step
            x_old = self.get_vb_param()
            try:
                self.set_vb_param(x_old + step_length * searchDir)
                bound, _, _ = self.vb_bound_grad_natgrad()
            except LinAlgError:  # What is the exception in tensorflow?
                self.set_vb_param(x_old)
                bound = bound_old - 1

            iteration += 1

            # Make sure there's an increase in the bound, else revert to steepest,
            # which is guaranteed to increase the bound.
            # (It's the same as VBEM.)
            if bound < bound_old:
                searchDir = -natgrad
                try:
                    self.set_vb_param(x_old + step_length*searchDir)
                    bound = self.bound()
                except LinAlgError:  # What is the tensorflow exception??
                    import warnings
                    warnings.warn("Caught LinalgError in setting variational parameters,\
                                  trying to continue with old parameter settings", LinAlgWarning)
                    self.set_vb_param(x_old)
                    bound, _, _ = self.vb_bound_grad_natgrad()
                    iteration_failed = False
                iteration += 1

            if verbose:
                print('\riteration '+str(iteration) +
                      ' bound='+str(bound) +
                      ' grad='+str(squareNorm) +
                      ', beta='+str(beta))
                sys.stdout.flush()

            # Converged yet? try the parameters if so
            if np.abs(bound-bound_old) <= ftol:
                if verbose:
                    print('vb converged (ftol)')

                if self.optimize_parameters() < 1e-1:
                    break

            if squareNorm <= gtol:
                if verbose:
                    print('vb converged (gtol)')

                if self.optimize_parameters() < 1e-1:
                    break

            if iteration >= maxiter:
                if verbose:
                    print('maxiter exceeded')
                break

            # store essentials of previous iteration
            natgrad_old = natgrad
            grad_old = grad
            searchDir_old = searchDir
            squareNorm_old = squareNorm

            # hyper param_optimisation
            if ((iteration > 1) and not (iteration % self.hyperparam_interval)) or iteration_failed:
                self.optimize_parameters()

            bound_old = bound

    def optimize_parameters(self):
        """
        Optimises the model parameters (non variational parameters)
        Returns the increment in the bound acheived
        """
        if self.get_free_state().size > 0:
            start = self.compute_log_likelihood()
            gpflow.model.Model.optimize(self, **self.hyperparam_opt_args)
            return self.compute_log_likelihood() - start
        else:
            return 0.
