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
    chols = [jitchol(A[:,:,i]) for i in range(N)]
    halflogdets = [np.sum(np.log(np.diag(L[0]))) for L in chols]
    invs = [GPy.util.linalg.dpotri(L[0],True)[0] for L in chols]
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

def lngammad(v,D):
    return np.sum([special.gammaln((v+1.-d)/2.) for d in range(1,D+1)],0)

def ln_dirichlet_C(a):
    return special.gammaln(a.sum())-np.sum(special.gammaln(a))


