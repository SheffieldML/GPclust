import numpy as np

N = 10; K = 2

phi = np.random.randn(N,K)
variance = 1.1
i = 0
B_inv = np.diag(1. / ((phi[:, i] + 1e-6) / variance))


