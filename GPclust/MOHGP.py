# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from collapsed_mixture import CollapsedMixture
import GPy
from GPy.util.linalg import mdot, pdinv, backsub_both_sides, dpotrs, jitchol, dtrtrs
from utilities import multiple_mahalanobis

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
    kernF    - A GPy kernel to model the mean function of each cluster
    kernY    - A GPy kernel to model the deviation of each of the time courses from the mean fro teh cluster
    alpha    - The a priori Dirichlet concentrationn parameter (default 1.)
    prior_Z  - Either 'symmetric' or 'dp', specifies whether to use a symmetric dirichelt prior for the clusters, or a (truncated) Dirichlet Process.
    name     - A convenient string for printing the model (default MOHGP)

    """
    def __init__(self, X, kernF, kernY, Y, K=2, alpha=1., prior_Z='symmetric', name='MOHGP'):

        N, self.D = Y.shape
        self.Y = Y
        self.X = X
        assert X.shape[0]==self.D, "input data don't match observations"

        CollapsedMixture.__init__(self, N, K, prior_Z, alpha, name)

        self.kernF = kernF
        self.kernY = kernY
        self.link_parameters(self.kernF, self.kernY)

        #initialize kernels
        self.Sf = self.kernF.K(self.X)
        self.Sy = self.kernY.K(self.X)
        self.Sy_inv, self.Sy_chol, self.Sy_chol_inv, self.Sy_logdet = pdinv(self.Sy+np.eye(self.D)*1e-6)

        #Computations that can be done outside the optimisation loop
        self.YYT = self.Y[:,:,np.newaxis]*self.Y[:,np.newaxis,:]
        self.YTY = np.dot(self.Y.T,self.Y)

        self.do_computations()


    def parameters_changed(self):
        """ Set the kernel parameters. Note that the variational parameters are handled separately."""
        #get the latest kernel matrices, decompose
        self.Sf = self.kernF.K(self.X)
        self.Sy = self.kernY.K(self.X)
        self.Sy_inv, self.Sy_chol, self.Sy_chol_inv, self.Sy_logdet = pdinv(self.Sy+np.eye(self.D)*1e-6)

        #update everything
        self.do_computations()
        self.update_kern_grads()


    def do_computations(self):
        """
        Here we do all the computations that are required whenever the kernels
        or the variational parameters are changed.
        """
        #sufficient stats.
        self.ybark = np.dot(self.phi.T,self.Y).T

        # compute posterior variances of each cluster (lambda_inv)
        tmp = backsub_both_sides(self.Sy_chol, self.Sf, transpose='right')
        self.Cs = [np.eye(self.D) + tmp*phi_hat_i for phi_hat_i in self.phi_hat]

        self._C_chols = [jitchol(C) for C in self.Cs]
        self.log_det_diff = np.array([2.*np.sum(np.log(np.diag(L))) for L in self._C_chols])
        tmp = [dtrtrs(L, self.Sy_chol.T, lower=1)[0] for L in self._C_chols]
        self.Lambda_inv = np.array([( self.Sy - np.dot(tmp_i.T, tmp_i) )/phi_hat_i if (phi_hat_i>1e-6) else self.Sf for phi_hat_i, tmp_i in zip(self.phi_hat, tmp)])

        #posterior mean and other useful quantities
        self.Syi_ybark, _ = dpotrs(self.Sy_chol, self.ybark, lower=1)
        self.Syi_ybarkybarkT_Syi = self.Syi_ybark.T[:,None,:]*self.Syi_ybark.T[:,:,None]
        self.muk = (self.Lambda_inv*self.Syi_ybark.T[:,:,None]).sum(1).T

    def update_kern_grads(self):
        """
        Set the derivative of the lower bound wrt the (kernel) parameters
        """

        tmp = [dtrtrs(L, self.Sy_chol_inv, lower=1)[0] for L in self._C_chols]
        B_invs = [phi_hat_i*np.dot(tmp_i.T, tmp_i) for phi_hat_i, tmp_i in zip(self.phi_hat, tmp)]
        #B_invs = [phi_hat_i*mdot(self.Sy_chol_inv.T,Ci,self.Sy_chol_inv) for phi_hat_i, Ci in zip(self.phi_hat,self.C_invs)]

        #heres the mukmukT*Lambda term
        LiSfi = [np.eye(self.D)-np.dot(self.Sf,Bi) for Bi in B_invs]#seems okay
        tmp1 = [np.dot(LiSfik.T,Sy_inv_ybark_k) for LiSfik, Sy_inv_ybark_k in zip(LiSfi,self.Syi_ybark.T)]
        tmp = 0.5*sum([np.dot(tmpi[:,None],tmpi[None,:]) for tmpi in tmp1])

        #here's the difference in log determinants term
        tmp += -0.5*sum(B_invs)

        #kernF_grads = np.array([np.sum(tmp*g) for g in self.kernF.extract_gradients()]) # OKAY!
        self.kernF.update_gradients_full(dL_dK=tmp,X=self.X)

        #gradient wrt Sigma_Y
        ybarkybarkT = self.ybark.T[:,None,:]*self.ybark.T[:,:,None]
        Byks = [np.dot(Bi,yk) for Bi,yk in zip(B_invs,self.ybark.T)]
        tmp = sum([np.dot(Byk[:,None],Byk[None,:])/np.power(ph_k,3)\
                -Syi_ybarkybarkT_Syi/ph_k -Bi/ph_k for Bi, Byk, yyT, ph_k, Syi_ybarkybarkT_Syi in zip(B_invs, Byks, ybarkybarkT, self.phi_hat, self.Syi_ybarkybarkT_Syi) if ph_k >1e-6])
        tmp += (self.K - self.N) * self.Sy_inv
        tmp += mdot(self.Sy_inv,self.YTY,self.Sy_inv)
        tmp /= 2.

        #kernY_grads = np.array([np.sum(tmp*g) for g in self.kernY.extract_gradients()])
        self.kernY.update_gradients_full(dL_dK=tmp, X=self.X)

    def bound(self):
        """
        Compute the lower bound on the marginal likelihood (conditioned on the
        GP hyper parameters).
        """
        return -0.5 * ( self.N * self.D * np.log(2. * np.pi) \
                        + self.log_det_diff.sum() \
                        + self.N * self.Sy_logdet \
                        + np.sum(self.YTY * self.Sy_inv) ) \
               + 0.5 * np.sum(self.Syi_ybarkybarkT_Syi * self.Lambda_inv) \
               + self.mixing_prop_bound() \
               + self.H

    def vb_grad_natgrad(self):
        """
        Natural Gradients of the bound with respect to phi, the variational
        parameters controlling assignment of the data to clusters
        """
        #yn_mk = self.Y[:,:,None] - self.muk[None,:,:]
        #ynmk2 = np.sum(np.dot(self.Sy_inv, yn_mk) * np.rollaxis(yn_mk,0,2),0)
        ynmk2 = multiple_mahalanobis(self.Y, self.muk.T, self.Sy_chol)

        grad_phi = (self.mixing_prop_bound_grad() -
                    0.5 * np.sum(np.sum(self.Lambda_inv * self.Sy_inv[None, :, :], 1), 1)) + \
                   ( self.Hgrad - 0.5 * ynmk2 ) # parentheses are for operation ordering!

        natgrad = grad_phi - np.sum(self.phi*grad_phi,1)[:,None]
        grad = natgrad*self.phi

        return grad.flatten(), natgrad.flatten()


    def predict_components(self,Xnew):
        """The predictive density under each component"""

        tmp = [dtrtrs(L, self.Sy_chol_inv, lower=1)[0] for L in self._C_chols]
        B_invs = [phi_hat_i*np.dot(tmp_i.T, tmp_i) for phi_hat_i, tmp_i in zip(self.phi_hat, tmp)]
        kx= self.kernF.K(self.X,Xnew)
        try:
            kxx = self.kernF.K(Xnew) + self.kernY.K(Xnew)
        except TypeError:
            #kernY has a hierarchical structure that we should deal with
            con = np.ones((Xnew.shape[0],self.kernY.connections.shape[1]))
            kxx = self.kernF.K(Xnew) + self.kernY.K(Xnew,con)

        #prediction as per my notes
        tmp = [np.eye(self.D) - np.dot(Bi,self.Sf) for Bi in B_invs]
        mu = [mdot(kx.T,tmpi,self.Sy_inv,ybark) for tmpi,ybark in zip(tmp,self.ybark.T)]
        var = [kxx - mdot(kx.T,Bi,kx) for Bi in B_invs]

        return mu,var

    def plot_simple(self):
        from matplotlib import pyplot as plt
        assert self.X.shape[1]==1, "can only plot mixtures of 1D functions"
        #plt.figure()
        plt.plot(self.Y.T,'k',linewidth=0.5,alpha=0.4)
        plt.plot(self.muk[:,self.phi_hat>1e-3],'k',linewidth=2)

    def plot(self, on_subplots=True, colour=False, newfig=True, errorbars=False, in_a_row=False, joined=True, gpplot=True,min_in_cluster=1e-3, data_in_grey=False, numbered=True, data_in_replicate=False, fixed_inputs=[], ylim=None):
        """
        Plot the mixture of Gaussian processes. Some of these arguments are rather esoteric! The defaults should be okay for most cases.

        Arguments
        ---------

        on_subplots (bool) whether to plot all the clusters on separate subplots (True), or all on the same plot (False)
        colour      (bool) to cycle through colours (True) or plot in black nd white (False)
        newfig      (bool) whether to make a new matplotlib figure(True) or use the current figure (False)
        in_a_row    (bool) if true, plot the subplots (if using) in a single row. Else make the subplots approximately square.
        joined      (bool) if true, connect the data points with lines
        gpplot      (bool) if true, plto the posterior of the GP for each cluster.
        min_in_cluster (float) ignore clusterse with less total assignemnt than this
        data_in_grey (bool) whether the data should be plotted in black and white.
        numbered    (bool) whether to include numbers on the top-right of each subplot.
        data_in_replicate (bool) whether to assume the data are in replicate, and plot the mean of each replicate instead of each data point
        fixed_inputs (list of tuples, as GPy.GP.plot) for GPs defined on more that one input, we'll plot a slice of the GP. this list defines how to fix the remaining inputs.
        ylim        (tuple) the limits to set on the y-axes. 

        """

        from matplotlib import pyplot as plt

        #work out what input dimensions to plot
        fixed_dims = np.array([i for i,v in fixed_inputs])
        free_dims = np.setdiff1d(np.arange(self.X.shape[1]),fixed_dims)
        assert len(free_dims)==1, "can only plot mixtures of 1D functions"

        #figure, subplots
        if newfig:
            fig = plt.figure()
        else:
            fig = plt.gcf()
        GPy.plotting.matplot_dep.Tango.reset()

        if data_in_replicate:
            X_ = self.X[:, free_dims].flatten()
            X = np.unique(X_).reshape(-1,1)
            Y = np.vstack([self.Y[:,X_==x].mean(1) for x in X.flatten()]).T

        else:
            Y = self.Y
            X = self.X[:,free_dims]

        #find some sensible y-limits for the plotting
        if ylim is None:
            ymin,ymax = Y.min(), Y.max()
            ymin,ymax = ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)
        else:
            ymin, ymax = ylim

        #work out how many clusters we're going to plot.
        Ntotal = np.sum(self.phi_hat > min_in_cluster)
        if on_subplots:
            if in_a_row:
                Nx = 1
                Ny = Ntotal
            else:
                Nx = np.floor(np.sqrt(Ntotal))
                Ny = int(np.ceil(Ntotal/Nx))
                Nx = int(Nx)
        else:
            ax = plt.gca() # this seems to make new ax if needed

        #limits of GPs
        xmin,xmax = X.min(), X.max()
        xmin,xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)

        Xgrid = np.empty((300,self.X.shape[1]))
        Xgrid[:,free_dims] = np.linspace(xmin,xmax,300)[:,None]
        for i,v in fixed_inputs:
            Xgrid[:,i] = v


        subplot_count = 0
        for i, ph, mu, var in zip(range(self.K), self.phi_hat, *self.predict_components(Xgrid)):
            if ph>(min_in_cluster):
                ii = np.argmax(self.phi,1)==i
                num_in_clust = np.sum(ii)
                if not np.any(ii):
                    continue
                if on_subplots:
                    ax = fig.add_subplot(Nx,Ny,subplot_count+1)
                    subplot_count += 1
                if colour:
                    col = GPy.plotting.matplot_dep.Tango.nextMedium()
                else:
                    col='k'
                if joined:
                    if data_in_grey:
                        ax.plot(X,Y[ii].T,'k',marker=None, linewidth=0.2,alpha=0.4)
                    else:
                        ax.plot(X,Y[ii].T,col,marker=None, linewidth=0.2,alpha=1)
                else:
                    if data_in_grey:
                        ax.plot(X,Y[ii].T,'k',marker='.', linewidth=0.0,alpha=0.4)
                    else:
                        ax.plot(X,Y[ii].T,col,marker='.', linewidth=0.0,alpha=1)

                if gpplot: GPy.plotting.matplot_dep.base_plots.gpplot(Xgrid[:,free_dims].flatten(),mu.flatten(),mu- 2.*np.sqrt(np.diag(var)),mu+2.*np.sqrt(np.diag(var)),col,col,ax=ax,alpha=0.1)

                if numbered and on_subplots:
                    ax.text(1,1,str(int(num_in_clust)),transform=ax.transAxes,ha='right',va='top',bbox={'ec':'k','lw':1.3,'fc':'w'})

                err = 2*np.sqrt(np.diag(self.Lambda_inv[i,:,:]))
                if errorbars:ax.errorbar(self.X.flatten(), self.muk[:,i], yerr=err,ecolor=col, elinewidth=2, linewidth=0)

                ax.set_ylim(ymin, ymax)

        if on_subplots:
            GPy.plotting.matplot_dep.base_plots.align_subplots(Nx,Ny,xlim=(xmin,xmax), ylim=(ymin, ymax))
        else:
            ax.set_xlim(xmin,xmax)
