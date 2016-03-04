import numpy as np
from scipy import linalg, special, sparse
import sys # for flushing
import GPy

def multiple_pdinv(A):
    """
    Arguments
    ---------
    A : A DxDxN numpy array (each A[:,:,i] is pd)

    Returns
    -------
    invs : the inverses of A
    hld: 0.5* the log of the determinants of A
    """
    N = A.shape[-1]
    chols = [GPy.util.linalg.jitchol(A[:,:,i]) for i in range(N)]
    halflogdets = [np.sum(np.log(np.diag(L))) for L in chols]
    invs = [GPy.util.linalg.dpotri(L,True)[0] for L in chols]
    invs = [np.triu(I)+np.triu(I,1).T for I in invs]
    return np.dstack(invs),np.array(halflogdets)


def softmax_numpy(X):
    log_phi = X.copy()
    max_x_i = X.max(1)

    log_phi -= max_x_i[:, None]
    phi = np.exp(log_phi)
    norm_i = phi.sum(1)

    log_norm_i = np.log(norm_i)
    phi /= norm_i[:, None]
    log_phi -= log_norm_i[:, None]
    entropy_i = -(phi * log_phi).sum(1)

    entropy = entropy_i.sum()

    return phi, log_phi, entropy


def multiple_mahalanobis_numpy_loops(X1, X2, L):
    N1, D = X1.shape
    N2, D = X2.shape
    
    LLT = L.dot(L.T)
    result = np.zeros(shape=(N1, N2), dtype=np.float64)
    n = 0
    while n < N1:
        m = 0
        while m < N2:
            x1x2 = X1[n] - X2[m]
            result[n, m] = x1x2.dot(np.linalg.solve(LLT, x1x2))
            m += 1

        n += 1

    return result


def lngammad(v,D):
    """sum of log gamma functions, as appears in a Wishart Distribution"""
    return np.sum([special.gammaln((v+1.-d)/2.) for d in range(1,D+1)],0)


def ln_dirichlet_C(a):
    """the log-normalizer of a Dirichlet distribution"""
    return special.gammaln(a.sum())-np.sum(special.gammaln(a))
