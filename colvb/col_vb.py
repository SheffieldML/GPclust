# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import pylab as pb
import GPy
import time
import sys #for flushing

class col_vb(GPy.core.model.Model):
    """
    A base class for collapsed variational models, using the GPy framework for
    non-variational parameters.

    Optimisation of the (collapsed) variational paremters is performed by
    Reimannian conjugate-gradient ascent, interleaved with optimisation of the
    non-variational parameters

    This class specifies a scheme for implementing the model, as well as
    providing the optimisation routine, and methods for tracking the optimisation
    """

    def __init__(self):
        """"""
        GPy.core.model.Model.__init__(self)

        #stuff for monitoring the different methods
        self.tracks = []
        self.tracktypes = []

        self.hyperparam_interval=50

        self.default_method = 'HS'
        self.hyperparam_opt_args = {
                'max_f_eval':20,
                'messages':1}

    def randomize(self):
        GPy.core.model.Model.randomize(self)
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
    def _set_params(self,x):
        """
        Set the non-variational parameters.
        Here we assume that there are no parameters...
        """
        pass
    def _get_params(self):
        """Returns the non-variational parameters"""
        return np.zeros(0)

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
        #first plot the bound as a function of iterations (or time)
        #pb.figure()
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
                pb.plot(t[:,0],t[:,1],col,linewidth=1.7,label=labels[ty])
                pb.xlabel('time (seconds)')
            else:
                t = [np.vstack((np.vstack((np.arange(x.shape[0]),x[:,1])).T,np.nan*np.ones(2))) for x,tt in zip(self.tracks, self.tracktypes) if tt==ty]
                if not len(t):
                    continue
                t = np.vstack(t)
                pb.plot(t[:,0],t[:,1],col,linewidth=1.7,label=labels[ty])# just plot by iteration
                pb.xlabel('iterations')
            #now plot crosses on the ends
            if bytime:
                x = np.vstack([t[-1] for t,tt in zip(self.tracks, self.tracktypes) if tt==ty])
            else:
                x = np.array([[len(t),t[-1,1]] for t,tt in zip(self.tracks, self.tracktypes) if tt==ty])
            pb.plot(x[:,0],x[:,1],col+'x',mew=1.5)
        pb.ylabel('bound')
        #pb.semilogx()
        pb.legend(loc=4)

    #def _ls_ffp(self,x):
        #""" objective for line search routine TODO: unify this and the below with a better line-search"""
        #self.set_vb_param(x.copy())
        #return -self.bound(), -self.vb_grad_natgrad()[0]

    def optimize_restarts(self,Nrestarts=10,robust=False,**kwargs):
        """TODO: integrate properly with GPy. probably means making a VB_optimiser class"""
        params = [self.get_vb_param()]
        bounds = [self.bound()]

        for i in range(Nrestarts):
            try:
                self.randomize()
                self.optimize(**kwargs)
            except Exception as e:
                if robust:
                    print "Warning: optimization failed"
                    continue
                else:
                    raise e
            bounds.append(self.bound())
            params.append(self.get_vb_param().copy())

        self.set_vb_param(params[np.argmax(bounds)])


    def E_time_to_opt(self, tolerance=0.1):
        """
        For each optimisation method, compute the expected time to find the best known solution, over all restarts
        """
        #find optimal solution (bound)
        bmax = np.max([e[-1,1] for e in self.tracks]) - tolerance

        labels = {'steepest':'steepest (=VBEM)', 'PR':'Polack-Ribiere', 'FR':'Fletcher-Reeves', 'HS':'Hestenes-Stiefel'}
        for trty, l in labels.items():
            time = np.sum([x[-1,0] for x,tt in zip(self.tracks, self.tracktypes) if tt==trty])
            N_opt = np.sum([x[-1,1]>bmax for x,tt in zip(self.tracks, self.tracktypes) if tt==trty])
            iters = np.sum([x.shape[0] for x,tt in zip(self.tracks, self.tracktypes) if tt==trty])*1.0 # force float division
            print l, 'E[time]='+str(time/N_opt), 'E[iters]='+str(iters/N_opt)

    def checkgrad_vb(self,**kwargs):
        """subvert GPy's checkgrad routine to check the vb gradients"""
        eg = self._log_likelihood_gradients_transformed
        self._log_likelihood_gradients_transformed = lambda : self.vb_grad_natgrad()[0]

        ep = self._set_params_transformed
        self._set_params_transformed = lambda x: self.set_vb_param(x)

        lp = self.log_prior
        self.log_p = lambda x: 0.

        et = self._get_params_transformed
        self._get_params_transformed = lambda : self.get_vb_param()

        en = self._get_param_names_transformed
        pn = [str(i) for i in range(self.get_vb_param().size)]
        self._get_param_names_transformed = lambda : pn

        ret = self.checkgrad(**kwargs)

        self._log_likelihood_gradients_transformed = eg
        self._set_params_transformed = ep
        self.log_prior = lp
        self._get_params_transformed = et
        self._get_param_names_transformed = en

        return ret

    def optimize_free_form(self, maxiter=500,ftol=1e-6):
        """Optimize the variational part o the model using the TNC optimizer"""
        def f_fprime(x):
            self.set_vb_param(x)
            return self.bound(), self.vb_grad_natgrad()[0]
        start = self.get_vb_param()
        from scipy import optimize
        opt,nf,rc = optimize.fmin_tnc(f_fprime,start,ftol=ftol)
        self.set_vb_param(opt)
        print nf, 'iters'

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
        if self._get_params().size:
            start = self.bound()
            GPy.core.model.Model.optimize(self,**self.hyperparam_opt_args)
            return self.bound()-start






