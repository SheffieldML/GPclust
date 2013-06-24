import numpy as np
import pylab as pb
import sys
from GPy import kern
from colvb import MOHGP

#cool structed GP demo
Nclust = 4
Nx = 20
Nobs = [np.random.randint(30,41) for i in range(Nclust)]
X = np.random.rand(Nx,1)*5
X.sort(0)

Kf = kern.rbf(1) + kern.white(1, 1e-6)
S = Kf.K(X)
means = np.vstack([np.tile(np.random.multivariate_normal(np.zeros(Nx),S,1),(N,1)) for N in Nobs]) # GP draws for mean of each cluster

#add GP draw for noise
Ky = kern.rbf(1,0.4,0.001) +  kern.white(1,0.01)
Y = means + np.random.multivariate_normal(np.zeros(Nx),Ky.K(X),means.shape[0])

#construct model
#m = MOHGP(X,Kf,Ky,Y, K=Nclust)
m = MOHGP(X,Kf,kern.white(1),Y, K=Nclust)
m.constrain_positive('')
m.checkgrad()
#m.checkgrad_vb()

m.optimize()
#m.systematic_splits()
#pb.ion()
#m.plot()






