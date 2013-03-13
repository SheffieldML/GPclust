import numpy as np
import pylab as pb
pb.close('all')
pb.ion()
from GPy import kern
import sys
sys.path.append('..')
from colvb import MOHGP, MOG, utilities
np.random.seed(0)

#cool structed GP demo
Nclust = 10
Nx = 12
Nobs = [np.random.randint(20,31) for i in range(Nclust)]
X = np.random.rand(Nx,1)
X.sort(0)
ground_truth_phi = utilities.blockdiag([np.ones((n,1)) for n in Nobs])
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
Kf = kern.rbf(1,0.01,0.001)
Ky1 = kern.rbf(1,0.1,0.001)
Ky2 = kern.white(1,0.01)
Ky = Ky1 + Ky2
m = MOHGP(X,Kf,Ky,Y, K=Nclust, prior_Z = 'DP', alpha=alpha)
m.ensure_default_constraints()
m.checkgrad()
#m.checkgrad_vb()

m.randomize()
m.optimize()
m.systematic_splits()
m.systematic_splits()
m.plot(1,1,1,0,1)

#construct model without structure
#give it a fighting chance by normalising signals first
Y = Y.copy()
Y -= Y.mean(1)[:,None]
Y /= Y.std(1)[:,None]
Kf = kern.rbf(1,0.01,0.001)
Ky = kern.white(1,0.01)
m2 = MOHGP(X,Kf,Ky,Y, K=Nclust, prior_Z = 'DP', alpha=alpha)
m2.ensure_default_constraints()
m2.checkgrad()

m2.randomize()
m2.optimize()
m2.systematic_splits()
m2.systematic_splits()
m2.plot(1,1,1,0,1)

#construct a MOG model (can't recover the clusters)
Y_ = Y.copy()
Y_ -= Y_.mean(0)
Y_ /= Y_.std(0)
m3 = MOG(Y_, prior_Z='DP', alpha=alpha)
m3.randomize()
m3.optimize()
m3.systematic_splits()
m3.systematic_splits()
pb.figure()
pb.subplot(2,2,1)
pb.imshow(ground_truth_phi,aspect='auto',cmap=pb.cm.gray)
pb.title('ground truth')
pb.subplot(2,2,2)
pb.imshow(m.phi,aspect='auto',cmap=pb.cm.gray)
pb.title('structured GP-DP')
pb.subplot(2,2,3)
pb.imshow(m2.phi,aspect='auto',cmap=pb.cm.gray)
pb.title('unstructured GP-DP')
pb.subplot(2,2,4)
pb.imshow(m3.phi,aspect='auto',cmap=pb.cm.gray)
pb.title('DP mixture model')






