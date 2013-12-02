import numpy as np
import pylab as pb
import sys
from GPy import kern
from colvb import MOHGP
np.random.seed(1)
pb.close('all')

#cool structed GP demo
Nclust = 50
Nx = 20
Nobs = [np.random.randint(20,21) for i in range(Nclust)]
X = np.random.rand(Nx,1)*5
X.sort(0)

Kf = kern.rbf(1) + kern.white(1, 1e-6)
S = Kf.K(X)
means = np.vstack([np.tile(np.random.multivariate_normal(np.zeros(Nx),S,1),(N,1)) for N in Nobs]) # GP draws for mean of each cluster

#add GP draw for noise
Ky = kern.rbf(1,0.3,1) +  kern.white(1,0.001)
Y = means + np.random.multivariate_normal(np.zeros(Nx),Ky.K(X),means.shape[0])

#construct model
m = MOHGP(X, Kf.copy(), Ky.copy(), Y, K=Nclust)
m.constrain_positive('')

m.optimize()
m.preferred_optimizer='bfgs'
m.systematic_splits()
m.remove_empty_clusters(1e-3)
m.plot(1,1,1,0,0,1)
raw_input('press enter to continue ...')

#and again without structure
Y -= Y.mean(1)[:,None]
Y /= Y.std(1)[:,None]
m2 = MOHGP(X, Kf, kern.white(1), Y, K=Nclust)
m2.constrain_positive('')
m2.preferred_optimizer='bfgs'
m2.optimize()
m2.systematic_splits()
m2.remove_empty_clusters(1e-3)
m2.plot(1,1,1,0,0,1)
