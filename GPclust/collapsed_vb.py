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
    Reimannian conjugate-gradient ascent, interleaved with optimisation of the
    non-variational parameters

    This class specifies a scheme for implementing the model, as well as
    providing the optimisation routine, and methods for tracking the optimisation
    """

    def __init__(self, name):
        """"""
        GPy.core.Model.__init__(self, name)

        #stuff for monitoring the different methods
        self.tracks = []
        self.tracktypes = []

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
        """In optimising the non variational (e.g. kernel) parameters, use the
        bound as a proxy for the likelihood"""
        return self.bound()

    def log_likelihood_gradients(self):
        """ Returns the gradient of the bound w.r.t. the non-variational parameters"""
        raise NotImplementedError

    def newtrack(self, method):
        """A simple method for keeping track of the optimisation"""
        self.tracktypes.append(method)
        self.tracks.append([])

    def track(self,stuff):
        self.tracks[-1].append(np.hstack([time.time(),stuff]))

    def closetrack(self):
        self.tracks[-1] = np.array(self.tracks[-1])
        self.tracks[-1][:,0] -= self.tracks[-1][0,0] # start timer from 0

    def plot_tracks(self,bytime=True):
        from matplotlib import pyplot as plt
        #first plot the bound as a function of iterations (or time)
        plt.figure()
        colours = {'steepest':'r','PR':'k','FR':'b', 'HS':'g', 'cg':'m','tnc':'c'}
        labels = {'steepest':'steepest (=VBEM)', 'PR':'Polack-Ribiere', 'FR':'Fletcher-Reeves', 'HS':'Hestenes-Stiefel','cg':'cg in gamma', 'tnc':'tnc in gamma'}
        for ty,col in colours.items():
            if not ty in self.tracktypes:
                continue
            if bytime:
                t = [np.vstack((x[:,:2],np.nan*np.ones(2))) for x,tt in zip(self.tracks, self.tracktypes) if tt==ty]
                if not len(t):
                    continue
                t = np.vstack(t)
                plt.plot(t[:,0],t[:,1],col,linewidth=1.7,label=labels[ty])
                plt.xlabel('time (seconds)')
            else:
                t = [np.vstack((np.vstack((np.arange(x.shape[0]),x[:,1])).T,np.nan*np.ones(2))) for x,tt in zip(self.tracks, self.tracktypes) if tt==ty]
                if not len(t):
                    continue
                t = np.vstack(t)
                plt.plot(t[:,0],t[:,1],col,linewidth=1.7,label=labels[ty])# just plot by iteration
                plt.xlabel('iterations')
            #now plot crosses on the ends
            if bytime:
                x = np.vstack([t[-1] for t,tt in zip(self.tracks, self.tracktypes) if tt==ty])
            else:
                x = np.array([[len(t),t[-1,1]] for t,tt in zip(self.tracks, self.tracktypes) if tt==ty])
            plt.plot(x[:,0],x[:,1],col+'x',mew=1.5)
        plt.ylabel('bound')
        plt.legend(loc=4)

    def optimize(self,method=None, maxiter=500, ftol=1e-6, gtol=1e-6, step_length=1., line_search=False, callback=None):
        """
        Optimize the model

        Arguments
        ---------
        :method: ['FR', 'PR','HS','steepest'] -- conjugate gradient method
        :maxiter: int
        :ftol: float
        :gtol: float
        :step_length: float (ignored if line-search is used)
        :line_search: bool -- whether to perform line searches

        Notes
        -----
        OPtimisation of the hyperparameters is interleaved with
        vb optimisation. The parameter self.hyperparam_interval
        dictates how often.
        """

        if method is None:
            method = self.default_method
        assert method in ['FR', 'PR','HS','steepest']
        # track:
        self.newtrack(method)

        iteration = 0
        bound_old = self.bound()
        while True:

            if not callback is None:
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
            if np.isnan(beta):
                beta = 0.
            if beta > 0:
                searchDir = -natgrad + beta*searchDir_old
            else:
                searchDir = -natgrad

            if line_search:
                xk = self.get_vb_param().copy()
                alpha = LS.line_search(self._ls_ffp,xk.copy(),searchDir)
                self.set_vb_param(xk + alpha*searchDir)
                bound = self.bound()
                print alpha, bound
                if bound < bound_old:
                    pdb.set_trace()
                iteration += 1

            else:
                #try a conjugate step
                phi_old = self.get_vb_param().copy()
                self.set_vb_param(phi_old + step_length*searchDir)
                bound = self.bound()
                iteration += 1

                #make sure there's an increase in L, else revert to steepest
                if bound<bound_old:
                    searchDir = -natgrad
                    self.set_vb_param(phi_old + step_length*searchDir)
                    bound = self.bound()
                    iteration += 1

            # track:
            self.track(np.hstack((bound, beta)))

            print '\riteration '+str(iteration)+' bound='+str(bound) + ' grad='+str(squareNorm) + ', beta='+str(beta),
            sys.stdout.flush()

            # converged yet? try the parameters if so
            if np.abs(bound-bound_old)<=ftol:
                print 'vb converged (ftol)'
                if self.optimize_parameters()<1e-1:
                    break
            if squareNorm<=gtol:
                print 'vb converged (gtol)'
                if self.optimize_parameters()<1e-1:
                    break
            if iteration>=maxiter:
                print 'maxiter exceeded'
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


        # track:
        self.closetrack()

    def optimize_parameters(self):
        """ optimises the model parameters (non variational parameters)
        Returns the increment in the bound acheived"""
        if self.size:
            start = self.bound()
            GPy.core.model.Model.optimize(self,**self.hyperparam_opt_args)
            return self.bound()-start






