# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from scipy import linalg, special, sparse, weave
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

def softmax_weave(X):
    """
    Use weave to compute the softmax of the (rows of) matrix X.

    Uses omp to parallelise the loop

    Also returns the log of the softmaxed result
    """

    #configure weave for parallel (or not)
    weave_options_openmp = {'headers'           : ['<omp.h>'],
                            'extra_compile_args': ['-fopenmp -O3'],
                            'extra_link_args'   : ['-lgomp'],
                            'libraries': ['gomp']}
    weave_options_noopenmp = {'extra_compile_args': ['-O3']}

    if GPy.util.config.config.getboolean('parallel', 'openmp'):
        weave_options = weave_options_openmp
        weave_support_code =  """
        #include <omp.h>
        #include <math.h>
        """
    else:
        weave_options = weave_options_noopenmp
        weave_support_code = "#include <math.h>"

    if GPy.util.config.config.getboolean('parallel', 'openmp'):
        pragma_string = '#pragma omp parallel for private(i, j, max_x, norm, log_norm, entropy)'
    else:
        pragma_string = ''

    N,D = X.shape
    phi = np.zeros_like(X)
    log_phi = X.copy()
    code = """
    int i, j;
    double max_x, norm, log_norm, entropy;
    double entropy_i [N];
    {pragma}
    for(i=0;i<N;i++){{

      //find the maximum element
      max_x = X(i,0);
      for(j=1;j<D;j++){{
        if (X(i,j)>max_x){{
          max_x = X(i,j);
        }}
      }}

      //compute un-normalised phi, normaliser
      norm = 0.0;
      for(j=0;j<D;j++){{
        log_phi(i,j) -= max_x;
        phi(i,j) = exp(log_phi(i,j));
        norm += phi(i,j);
      }}
      log_norm = log(norm);

      //normalise, compute entropy
      entropy_i[i] = 0.0;
      for(j=0;j<D;j++){{
        phi(i,j) /= norm;
        log_phi(i,j) -= log_norm;
        entropy_i[i] -= phi(i,j)*log_phi(i,j);
      }}
    }}

    //sum entropies for each variable
    entropy = 0.0;
    for(i=0;i<N;i++){{
      entropy += entropy_i[i];
    }}

    return_val = entropy;

    """.format(pragma=pragma_string)

    H = weave.inline(code, arg_names=["X", "phi", "log_phi", "N", "D"], type_converters=weave.converters.blitz, support_code=weave_support_code, **weave_options)
    return phi, log_phi, H


def multiple_mahalanobis(X1, X2, L):
    """
    X1 is a N1 x D array
    X2 is a N2 x D array
    L is a D x D array, lower triangular

    compute (x1_n - x2_m).T * (L * L.T)^-1 * (x1_n - x2_m)

    for each pair of the D_vectors x1_n, x2_m

    Returns: a N1 x N2 array of each distance
    """
    N1,D = X1.shape
    N2,D = X2.shape
    assert L.shape == (D,D)
    result = np.zeros(shape=(N1,N2), dtype=np.float64)

    #configure weave for parallel (or not)
    weave_options_openmp = {'headers'           : ['<omp.h>'],
                            'extra_compile_args': ['-fopenmp -O3'],
                            'extra_link_args'   : ['-lgomp'],
                            'libraries': ['gomp']}
    weave_options_noopenmp = {'extra_compile_args': ['-O3']}

    if GPy.util.config.config.getboolean('parallel', 'openmp'):
        weave_options = weave_options_openmp
        weave_support_code =  """
        #include <omp.h>
        #include <math.h>
        """
    else:
        weave_options = weave_options_noopenmp
        weave_support_code = "#include <math.h>"

    if GPy.util.config.config.getboolean('parallel', 'openmp'):
        pragma_string = '#pragma omp parallel for private(n,m,i,j,tmp)'
    else:
        pragma_string = ''

    code = """
    double tmp [D];
    //two loops over the N1 x N2 vectors
    int n, m, i, j;
    {pragma}
    for(n=0; n<N1; n++){{
      for(m=0; m<N2; m++){{

        //a double loop to solve the cholesky problem into tmp (should really use blas?)
        for(i=0; i<D; i++){{
          tmp[i] = X1(n,i) - X2(m,i);
          for(j=0; j<i; j++){{
            tmp[i] -= L(i,j)*tmp[j];
          }}
          tmp[i] /= L(i,i);
        }}

        //loop over tmp to get the result: tmp.T * tmp (should really use blas again)
        for(i=0; i<D; i++){{
          result(n,m) += tmp[i]*tmp[i];
        }}
      }}
    }}
    """.format(pragma=pragma_string)
    weave.inline(code, arg_names=["X1", "X2", "L", "N1", "N2", "D", "result"], type_converters=weave.converters.blitz, support_code=weave_support_code, **weave_options)
    return result



def lngammad(v,D):
    """sum of log gamma functions, as appears in a Wishart Distribution"""
    return np.sum([special.gammaln((v+1.-d)/2.) for d in range(1,D+1)],0)

def ln_dirichlet_C(a):
    """the log-normalizer of a Dirichlet distribution"""
    return special.gammaln(a.sum())-np.sum(special.gammaln(a))


