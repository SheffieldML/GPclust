import numpy as np
import pylab as pb
import sys
sys.path.append('..')
from colvb import MMOG

M = 20
N = 50
K = 8
D = 2

means = np.random.rand(K,D)*16 -8
pis = np.random.dirichlet(np.ones(K),M)
latents = np.asarray([np.random.multinomial(1,p,N).argmax(1) for p in pis])
data = np.random.randn(M,N,D) + means[latents,:]

m = MMOG(data,1)

m.optimize()
m.systematic_splits()
m.plot()
pb.show()
