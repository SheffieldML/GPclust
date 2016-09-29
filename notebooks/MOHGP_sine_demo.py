import numpy as np
import scipy as sp
import GPflow
import GPclust
#from GPclust import MOHGP, MOG, utilities
from plotter import plot
np.random.seed(0)

#cool structed GP demo
Nclust = 10
Nx = 12
Nobs = [np.random.randint(20,31) for i in range(Nclust)]
X = np.random.rand(Nx,1)
X.sort(0)
#ground_truth_phi = sp.linalg.block_diag([np.ones((n,1)) for n in Nobs])
alpha = 2. #(should give approximately 10 clusters)

freqs = 2*np.pi + 0.3*(np.random.rand(Nclust)-.5)
phases = 2*np.pi*np.random.rand(Nclust)
means = np.vstack([np.tile(np.sin(f*X+p).T,(Ni,1)) for f,p,Ni in zip(freqs,phases,Nobs)])

#add a lower freq sin for the noise
freqs = .4*np.pi + 0.01*(np.random.rand(means.shape[0])-.5)
phases = 2*np.pi*np.random.rand(means.shape[0])
offsets = 0.3*np.vstack([np.sin(f*X+p).T for f,p in zip(freqs,phases)])
Y = means + offsets + np.random.randn(*means.shape)*0.05

#construct full model
Kf = GPflow.kernels.RBF(1,0.01,0.001) 
Ky1 = GPflow.kernels.RBF(1,0.01,0.001) 
Ky2 = GPflow.kernels.White(1, variance=1e-6)
Ky = Ky1 + Ky2
m = GPclust.MOHGP(X, Kf, Ky, Y, num_clusters=Nclust, prior_Z='DP', alpha=alpha)

m.optimize()
m.systematic_splits()
plot(m,X)

#construct model without structure
#give it a fighting chance by normalising signals first
Y = Y.copy()
Y -= Y.mean(1)[:,None]
Y /= Y.std(1)[:,None]

Kf = GPflow.kernels.RBF(1,0.01,0.001) 
Ky = GPflow.kernels.White(1,0.01)
m2 = MOHGP(X, Kf, Ky, Y, num_clusters=Nclust, prior_Z = 'DP', alpha=alpha)

m2.optimize()
m2.systematic_splits()
m2.systematic_splits()
plot(m2,X)

#construct a MOG model (can't recover the clusters)
Y_ = Y.copy()
Y_ -= Y_.mean(0)
Y_ /= Y_.std(0)
m3 = MOG(Y_, prior_Z='DP', alpha=alpha)
m3.optimize()
m3.systematic_splits()
m3.systematic_splits()

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,2,1)
#plt.imshow(ground_truth_phi,aspect='auto',cmap=plt.cm.gray)
#plt.title('ground truth')
#plt.subplot(2,2,2)
plt.imshow(m.get_phi(),aspect='auto',cmap=plt.cm.gray)
plt.title('structured GP-DP')
plt.subplot(2,2,2)
plt.imshow(m2.get_phi(),aspect='auto',cmap=plt.cm.gray)
plt.title('unstructured GP-DP')
plt.subplot(2,2,3)
plt.imshow(m3.get_phi(),aspect='auto',cmap=plt.cm.gray)
plt.title('DP mixture model')






