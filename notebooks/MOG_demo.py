# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys
sys.path.append('..')
from colvb import MOG
pb.ion()

np.random.seed(0)

#make some Data which appears in clusters:
Nclust = 15
dim = 2
Nmin = 25
Nmax = 50
Ndata = np.random.randint(Nmin, Nmax, Nclust)
means = np.random.randn(Nclust, dim)*5
aa = [np.random.randn(dim, dim+1) for i in range(Nclust)]
Sigmas = [np.dot(a, a.T) for a in aa]
X = np.vstack([np.random.multivariate_normal(mu, cov, (n,)) for mu, cov, n in zip(means, Sigmas, Ndata)])/100
Nrestarts=3
Nclust = 15

m = MOG(X, Nclust, prior_Z='DP')

#starts = [np.random.randn(m.N*m.K) for i in range(Nrestarts)]
from scipy.cluster import vq
starts = []
for i in range(Nrestarts):
    means = X[np.random.permutation(X.shape[0])[:Nclust]]
    dists = np.square(X[:,:,None]-means.T[None,:,:]).sum(1)
    starts.append(dists)

for method in ['steepest', 'PR', 'FR', 'HS']:
    for st in starts:
        m.set_vb_param(st)
        m.optimize(method=method, maxiter=1e4)

m.plot_tracks()
m.plot()





