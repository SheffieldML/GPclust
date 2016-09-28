import numpy as np
import GPflow
from GPclust import MOHGP
from plotter import plot
np.random.seed(1)

#cool structed GP demo
Nclust = 20
Nx = 12
Nobs = [np.random.randint(20,21) for i in range(Nclust)]
X = np.random.rand(Nx,1)*5
X.sort(0)

Kf = GPflow.kernels.RBF(1) + GPflow.kernels.White(1, variance=1e-6)

S = Kf.compute_K(X,X)
means = np.vstack([np.tile(np.random.multivariate_normal(np.zeros(Nx),S,1),(N,1)) for N in Nobs]) # GP draws for mean of each cluster

#add GP draw for noise
Ky = GPflow.kernels.RBF(1,0.3,1) +  GPflow.kernels.White(1,0.001)
Y = means + np.random.multivariate_normal(np.zeros(Nx),Ky.compute_K(X,X),means.shape[0])
#construct model
m = MOHGP(X, Kf, Ky, Y, num_clusters=Nclust)

m.optimize()
m.preferred_optimizer='bfgs'
#m.systematic_splits()
#m.remove_empty_clusters(1e-3)
plot(m,X)
raw_input('press enter to continue ...')

#and again without structure
Y -= Y.mean(1)[:,None]
Y /= Y.std(1)[:,None]
m2 = MOHGP(X, Kf, GPy.kern.White(1), Y, K=Nclust)
#m2.preferred_optimizer='bfgs'
m2.optimize()
m2.systematic_splits()
#m2.remove_empty_clusters(1e-3)
plot(m2,X)
