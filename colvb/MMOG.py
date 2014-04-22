import numpy as np
import pylab as pb
from scipy import optimize, linalg
from utilities import multiple_pdinv, lngammad, ln_dirichlet_C
from scipy.special import gammaln, digamma
from scipy import stats
from col_vb import col_vb
import sys
import GPy

def multi_softmax(x):
    ex = np.exp(x-x.max(-1)[:,:,None])
    return ex/ex.sum(-1)[:,:,None]


class MMOG(col_vb):
    """
    Many Mixtures of Gaussians (like LDA for continuous data)
    """
    def __init__(self,X,K=10,alpha=1.,prior_Z='symmetric'):
        self.M,self.N,self.D = X.shape
        self.X = X
        self.K = K

        #prior cluster parameters
        self.m0 = np.vstack(self.X).mean(0) # priors on the Gaussian components
        self.k0 = 1e-3
        self.S0 = np.eye(self.D)*1e-3
        self.S0_halflogdet = np.sum(np.log(np.sqrt(np.diag(self.S0))))
        self.v0 = self.D+1.

        #prior Z parameters
        if prior_Z=='symmetric':
            self.alpha = alpha
        else:
            raise NotImplementedError # TODO: DP priors

        #precomputed stuff
        self.k0m0m0T = self.k0*self.m0[:,np.newaxis]*self.m0[np.newaxis,:]
        self.XXT = self.X[:,:,:,np.newaxis]*self.X[:,:,np.newaxis,:]

        col_vb.__init__(self)
        self.set_vb_param(np.random.randn(self.M*self.N*self.K))

        #stuff for monitoring the different methods
        self.tracks = []
        self.tracktypes = []

    def systematic_splits(self):
        for kk in range(self.K):
            self.recursive_splits(kk)

    def recursive_splits(self,k=0):
        success = self.try_split(k)
        if success:
            self.recursive_splits(self.K-1)
            self.recursive_splits(k)

    def try_split(self,indexK=None,threshold=0.9):
        """
        Re-initialize one of the clusters as two clusters, optimize, and keep
        the solution if the bound is increased. Kernel hypers stay constant.

        Arguments
        ---------
        indexK: (int) the index of the cluster to split
        threshold: float [0,1], to assign data to the splitting cluster

        Returns
        -------
        Success: (bool)

        """
        if indexK is None:
            indexK = np.random.multinomial(1,self.phi_hat/(self.M*self.N)).argmax()
        if self.phi_hat[indexK]<1:
            return False # no data to split

        bound_old = self.bound()
        phi_old = self.get_vb_param().copy()

        #re-initalize
        self.K += 1
        self.phi_ = np.asarray([np.hstack((phi_,np.ones((self.N,1))*phi_[:,indexK].min())) for phi_ in self.phi_])
        for m in range(self.M):
            indexN = np.nonzero(self.phi[m,:,indexK] > threshold)[0]
            self.phi_[m,indexN,indexK] = np.random.randn(indexN.size)
            self.phi_[m,indexN,-1] = np.random.randn(indexN.size)
        self.set_vb_param(self.get_vb_param())

        self.optimize(method='HS')
        bound_new = self.bound()

        bound_increase = bound_new-bound_old
        if (bound_increase < 0.):
            self.K -= 1
            self.set_vb_param(phi_old)
            print "split failed, bound changed by: ",bound_increase
            return False
        else:
            print "split suceeded, bound changed by: ",bound_increase
            return True

    def get_vb_param(self):
        return self.phi_.flatten()

    def set_vb_param(self,phi_):
        #unflatten and softmax
        self.phi_ = phi_.reshape(self.M,self.N,self.K)
        self.phi = multi_softmax(self.phi_)

        #computations needed for bound, gradient and predictions
        self.phi_hat_mk = self.phi.sum(1) # M x K
        self.phi_hat = self.phi_hat_mk.sum(0) # K
        self.kNs = self.phi_hat + self.k0
        self.vNs = self.phi_hat + self.v0
        self.Xsumk = np.sum(np.sum(self.X[:,:,:,None]*self.phi[:,:,None,:],0),0) # D x K
        Ck = np.sum(np.sum(self.phi[:,:,None,None,:]*self.XXT[:,:,:,:,None],0),0) # D x D x K
        self.mun = (self.k0*self.m0[:,None] + self.Xsumk)/self.kNs[None,:] # D x K
        self.munmunT = self.mun[:,None,:]*self.mun[None,:,:] # D x D x K
        self.Sns = self.S0[:,:,None] + Ck + self.k0m0m0T[:,:,None] - self.kNs[None,None,:]*self.munmunT
        self.Sns_inv, self.Sns_halflogdet = multiple_pdinv(self.Sns)
        self.logphi = np.log(np.clip(self.phi,1e-6,1.))
        self.H = -np.sum(self.phi*self.logphi)

    def bound(self):
        """Compute the lower bound on the model evidence."""
        return -0.5*self.D*np.sum(np.log(self.kNs/self.k0))\
            +self.K*self.v0*self.S0_halflogdet - np.sum(self.vNs*self.Sns_halflogdet)\
            +np.sum(lngammad(self.vNs,self.D))- self.K*lngammad(self.v0,self.D)\
            +self.M*ln_dirichlet_C(np.ones(self.K)*self.alpha) - np.sum([ln_dirichlet_C(self.alpha + ph) for ph in self.phi_hat_mk])\
            + self.H\
            -0.5*self.N*self.M*self.D*np.log(np.pi)

    def vb_grad_natgrad(self):
        """Gradients of the bound"""
        x_m = self.X[:,:,:,None]-self.mun[None,None,:,:] # M, N, D, K
        dS = x_m[:,:,:,None,:]*x_m[:,:,None,:,:] # M, N, D, D, K
        SnidS = self.Sns_inv[None,None,:,:,:]*dS
        dlndtS_dphi = np.dot(np.ones(self.D),np.dot(np.ones(self.D),SnidS)) # sum out the D axes to get M,N,K

        #grad_phi =  (-0.5*self.D/self.kNs + 0.5*digamma((self.vNs-np.arange(self.D)[:,None])/2.).sum(0) + digamma(self.phi_hat+self.alpha) - self.Sns_halflogdet -1.) + (-np.log(np.clip(self.phi,1e-20,1))-0.5*dlndtS_dphi*self.vNs)
        grad_phi =  -0.5*self.D/self.kNs + 0.5*digamma((self.vNs-np.arange(self.D)[:,None])/2.).sum(0)-0.5*dlndtS_dphi*self.vNs - self.Sns_halflogdet -1. - self.logphi + digamma(self.phi_hat_mk+self.alpha)[:,None,:]

        natgrad = grad_phi - np.sum(self.phi*grad_phi,-1)[:,:,None]


        grad = natgrad*self.phi

        return grad.flatten(), natgrad.flatten()


    def predict_components_ln(self,Xnew):
        """The predictive density under each component (a T distribution)"""
        Dist =     Xnew[:,:,np.newaxis]-self.mun[np.newaxis,:,:] # Nnew x D x K
        tmp = np.sum(Dist[:,:,None,:]*self.Sns_inv[None,:,:,:],1)#*(kn+1.)/(kn*(vn-self.D+1.))
        mahalanobis = np.sum(tmp*Dist,1)/(self.kNs+1.)*self.kNs*(self.vNs-self.D+1.)
        halflndetSigma = self.Sns_halflogdet + 0.5*self.D*np.log((self.kNs+1.)/(self.kNs*(self.vNs-self.D+1.)))

        Z  = gammaln(0.5*(self.vNs[np.newaxis,:]+1.))\
            -gammaln(0.5*(self.vNs[np.newaxis,:]-self.D+1.))\
            -(0.5*self.D)*(np.log(self.vNs[np.newaxis,:]-self.D+1.) + np.log(np.pi))\
            - halflndetSigma \
            - (0.5*(self.vNs[np.newaxis,:]+1.))*np.log(1.+mahalanobis/(self.vNs[np.newaxis,:]-self.D+1.))
        return Z

    def predict_components(self,Xnew):
        return np.exp(self.predict_components_ln(Xnew))

    def predict(self,Xnew):
        Z = self.predict_components(Xnew)
        #calculate the weights for each component
        pi = self.phi_hat_mk+self.alpha # M x K
        pi /= pi.sum(-1)[:,None]
        Z = Z[:,None,:]*pi[None,:,:]
        return Z.sum(-1)


    def plot(self):
        rows = np.sqrt(self.M).round()
        cols = np.ceil(self.M/rows)

        if self.D==1:
            xmin = self.X.min()
            xmax = self.X.max()
            xmin,xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)
            Xgrid = np.linspace(xmin,xmax,100)[:,None]
            zz = self.predict(Xgrid)

        if self.D==2:
            xmin,ymin = np.vstack(self.X).min(0)
            xmax,ymax = np.vstack(self.X).max(0)
            xmin,xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)
            ymin,ymax = ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)
            xx,yy = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
            Xgrid = np.vstack((xx.flatten(),yy.flatten())).T
            zz = self.predict(Xgrid).reshape(100,100,self.M)

        for m in range(self.M):
            pb.subplot(rows,cols,m+1)
            if self.D==1:
                pb.hist(self.X[m,:,0],self.N/10.,normed=True)
                pb.plot(Xgrid,zz[:,m],'r',linewidth=2)
            elif self.D==2:
                pb.plot(self.X[m,:,0],self.X[m,:,1],'rx',mew=2)
                zz_data = self.predict(self.X[m])[:,m]
                pb.contour(xx,yy,zz[:,:,m],[stats.scoreatpercentile(zz_data,5)],colors='r',linewidths=1.5)
                pb.imshow(zz[:,:,m].T,extent=[xmin,xmax,ymin,ymax],origin='lower',cmap=pb.cm.binary,vmin=0.,vmax=zz_data.max())
                #pb.pcolor(xx,yy,zz[:,:,m],cmap=pb.cm.binary,vmin=0.,vmax=zz_data.max())






