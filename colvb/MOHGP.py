import numpy as np
import pylab as pb
from scipy import optimize, linalg
from utilities import pdinv, softmax, multiple_pdinv, blockdiag, lngammad, ln_dirichlet_C, safe_GP_inv
from scipy.special import gammaln, digamma
import sys
from col_vb import col_vb
from col_mix import collapsed_mixture
import GPy
from GPy.util.linalg import mdot

class MOHGP(collapsed_mixture):
    """
    A Hierarchical Mixture of Gaussian Processes
    A hierarchy is formed by using a GP prior for the cluster function values and another for the likelihood
    """
    def __init__(self, X, kernF, kernY, Y, K=2, alpha=1., prior_Z='symmetric'):
        N,self.D = Y.shape
        self.X = X
        assert X.shape[0]==self.D, "input data don't match observations"
        self.kernF = kernF
        self.kernY = kernY
        self.Y = Y

        #Computations that can be done outside the optimisation loop
        self.YYT = self.Y[:,:,np.newaxis]*self.Y[:,np.newaxis,:]
        self.YTY = np.dot(self.Y.T,self.Y)

        collapsed_mixture.__init__(self, N, K, prior_Z, alpha)

    def _set_params(self,x):
        """ Set the kernel parameters """
        self.kernF._set_params_transformed(x[:self.kernF.Nparam])
        self.kernY._set_params_transformed(x[self.kernF.Nparam:])
        self.do_computations()

    def _get_params(self):
        """ returns the kernel parameters """
        return np.hstack([self.kernF._get_params_transformed(), self.kernY._get_params_transformed()])

    def _get_param_names(self):
        return ['kernF_'+ n for n in self.kernF._get_param_names_transformed()] + ['kernY_' + n for n in self.kernY._get_param_names_transformed()]

    def _log_likelihood_gradients(self):
        """
        The derivative of the lower bound wrt the (kernel) parameters
        """

        #heres the mukmukT*Lambda term
        LiSfi = [np.eye(self.D)-np.dot(self.Sf,Bi) for Bi in self.B_invs]#seems okay
        tmp1 = [mdot(LiSfik.T,self.Sy_inv,y) for LiSfik,y in zip(LiSfi,self.ybark.T)]
        tmp = 0.5*sum([np.dot(tmpi[:,None],tmpi[None,:]) for tmpi in tmp1])

        #here's the difference in log determinants term
        tmp += -0.5*sum(self.B_invs)

        #kernF_grads = np.array([np.sum(tmp*g) for g in self.kernF.extract_gradients()]) # OKAY!
        kernF_grads = self.kernF.dK_dtheta(tmp,self.X)

        #gradient wrt Sigma_Y
        Byks = [np.dot(Bi,yk) for Bi,yk in zip(self.B_invs,self.ybark.T)]
        tmp = sum([np.dot(Byk[:,None],Byk[None,:])/np.power(ph_k,3)\
                -self.Syi_ybarkybarkT_Syi[:,:,k]/ph_k -Bi/ph_k for k,(Bi,Byk,yyT,ph_k) in enumerate(zip(self.B_invs,Byks,np.rollaxis(self.ybarkybarkT,-1),self.phi_hat)) if ph_k >1e-6])
        tmp += (self.K-self.N)*self.Sy_inv
        #tmp += mdot(self.Sy_inv,self.Ck.sum(-1),self.Sy_inv)
        tmp += mdot(self.Sy_inv,self.YTY,self.Sy_inv)
        tmp /= 2.

        #kernY_grads = np.array([np.sum(tmp*g) for g in self.kernY.extract_gradients()])
        kernY_grads = self.kernY.dK_dtheta(tmp,self.X)

        return np.hstack((kernF_grads, kernY_grads))

    def do_computations(self):
        #get the latest kernel matrices
        self.Sf = self.kernF.K(self.X)
        self.Sy = self.kernY.K(self.X)

        #sufficient stats. speed bottleneck?
        self.ybark = np.dot(self.phi.T,self.Y).T
        #self.Ck = np.dstack([np.dot(self.Y.T,phi[:,None]*self.Y) for phi in self.phi.T])

        # compute posterior variances of each cluster (lambda_inv)
        self.Sy_chol = np.linalg.cholesky(self.Sy)
        self.Sy_chol_inv = linalg.flapack.dtrtri(self.Sy_chol.T)[0].T
        self.Sy_inv = np.dot(self.Sy_chol_inv.T, self.Sy_chol_inv)
        self.Sy_hld = np.sum(np.log(np.diag(self.Sy_chol)))
        tmp = mdot(self.Sy_chol_inv,self.Sf,self.Sy_chol_inv.T)
        self.Cs = [np.eye(self.D) + tmp*phi_hat_i for phi_hat_i in self.phi_hat]
        self.C_invs, self.hld_diff = zip(*[pdinv(C) for C in self.Cs])
        self.Lambda_inv = [self.Sy/phi_hat_i - mdot(self.Sy_chol,Ci,self.Sy_chol.T)/phi_hat_i if (phi_hat_i>1e-6) else self.Sf for phi_hat_i,Ci in zip(self.phi_hat,self.C_invs)]
        self.hld_diff = np.array(self.hld_diff)

        #compute posterior means
        self.muk = np.array([mdot(Li,self.Sy_inv,ybark) for Li,ybark in zip(self.Lambda_inv,self.ybark.T)]).T

        #useful quantities
        self.Lambda_inv, self.hld_diff = np.dstack(self.Lambda_inv), np.array(self.hld_diff)
        self.ybarkybarkT = self.ybark[:,None,:]*self.ybark[None,:,:]
        self.mukmukT = self.muk[:,None,:]*self.muk[None,:,:]
        self.Syi_ybark = np.dot(self.Sy_inv, self.ybark)
        self.Syi_ybarkybarkT_Syi = self.Syi_ybark[:,None,:]*self.Syi_ybark[None,:,:]

        self.B_invs = [phi_hat_i*mdot(self.Sy_chol_inv.T,Ci,self.Sy_chol_inv) for phi_hat_i, Ci in zip(self.phi_hat,self.C_invs)]

    def bound(self):
        """Compute the lower bound on the marginal likelihood (conditioned on the GP hyper parameters). """
        #return -0.5*self.N*self.D*np.log(2.*np.pi) -self.hld_diff.sum() - self.N*self.Sy_hld -0.5*np.sum(self.Ck*self.Sy_inv[:,:,None])\
        return -0.5*self.N*self.D*np.log(2.*np.pi) -self.hld_diff.sum() - self.N*self.Sy_hld -0.5*np.sum(self.YTY*self.Sy_inv)\
            + 0.5*np.sum(self.Syi_ybarkybarkT_Syi*self.Lambda_inv)\
            + self.mixing_prop_bound() + self.H

    def vb_grad_natgrad(self):
        """Gradients of the bound"""
        yn_mk = self.Y[:,:,None]-self.muk[None,:,:]
        ynmk2 = np.sum(np.dot(self.Sy_inv,yn_mk)*np.rollaxis(yn_mk,0,2),0)
        grad_phi = self.mixing_prop_bound_grad() - 0.5*np.sum(np.sum(self.Lambda_inv*self.Sy_inv[:,:,None],0),0) -0.5*ynmk2 +self.Hgrad
        natgrad = grad_phi - np.sum(self.phi*grad_phi,1)[:,None]
        grad = natgrad*self.phi

        return grad.flatten(), natgrad.flatten()


    def predict_components(self,Xnew):
        """The predictive density under each component"""
        kx= self.kernF.K(self.X,Xnew)
        try:
            kxx = self.kernF.K(Xnew) + self.kernY.K(Xnew)
        except TypeError:
            #kernY has a hierarchical structure that we should deal with
            con = np.ones((Xnew.shape[0],self.kernY.connections.shape[1]))
            kxx = self.kernF.K(Xnew) + self.kernY.K(Xnew,con)


        #prediction as per my notes
        tmp = [np.eye(self.D) - np.dot(Bi,self.Sf) for Bi in self.B_invs]
        mu = [mdot(kx.T,tmpi,self.Sy_inv,ybark) for tmpi,ybark in zip(tmp,self.ybark.T)]
        var = [kxx - mdot(kx.T,Bi,kx) for Bi in self.B_invs]

        return mu,var

    def plot_simple(self):
        assert self.X.shape[1]==1, "can only plot mixtures of 1D functions"
        #pb.figure()
        pb.plot(self.Y.T,'k',linewidth=0.5,alpha=0.4)
        pb.plot(self.muk[:,self.phi_hat>1e-3],'k',linewidth=2)

    def plot(self, on_subplots=False,colour=False,newfig=True,errorbars=False,in_a_row=False,joined=True):

        assert self.X.shape[1]==1, "can only plot mixtures of 1D functions"

        #figure, subplots
        if newfig:
            f = pb.figure()
        else:
            f = pb.gcf()
        GPy.util.plot.Tango.reset()
        if on_subplots:
            if in_a_row:
                Nx = 1
                Ny = self.K
            else:
                Nx = np.floor(np.sqrt(self.K))
                Ny = int(np.ceil(self.K/Nx))
                Nx = int(Nx)
        else:
            ax = pb.gca() # this seems to make new ax if needed

        #limits of GPs
        xmin,xmax = self.X.min(), self.X.max()
        ymin,ymax = self.Y.min(), self.Y.max()
        xmin,xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)
        ymin,ymax = ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)
        xgrid = np.linspace(xmin,xmax,300)[:,None]

        for i,ph, mu, var in zip(range(self.K),self.phi_hat, *self.predict_components(xgrid)):
            if ph>(1e-3):
                ii = np.argmax(self.phi,1)==i
                if not np.any(ii):
                    continue
                if on_subplots:
                    ax = pb.subplot(Nx,Ny,i+1)
                if colour:
                    col = GPy.util.plot.Tango.nextMedium()
                else:
                    col='k'
                if joined:
                    ax.plot(self.X,self.Y[ii].T,col,marker=None, linewidth=0.2,alpha=1)
                else:
                    ax.plot(self.X,self.Y[ii].T,col,marker='.', linewidth=0.0,alpha=1)
                GPy.util.plot.gpplot(xgrid.flatten(),mu.flatten(),mu- 2.*np.sqrt(np.diag(var)),mu+2.*np.sqrt(np.diag(var)),col,col,axes=ax,alpha=0.1)

                err = 2*np.sqrt(np.diag(self.Lambda_inv[:,:,i]))
                if errorbars:ax.errorbar(self.X.flatten(), self.muk[:,i], yerr=err,ecolor=col, elinewidth=2, linewidth=0)

        if on_subplots:
            GPy.util.plot.align_subplots(Nx,Ny,xlim=(xmin,xmax))
        else:
            ax.set_xlim(xmin,xmax)


