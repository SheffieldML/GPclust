# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from scipy import linalg, special, sparse, weave
import sys # for flushing
import GPy

def safe_GP_inv(K,w):
    """
    Arguments
    ---------
    K, a NxN pd matrix
    w, a N-vector

    Returns
    -------
    (K^-1 + diag(w))^-1
    and
    (1/2)*(ln|K^-1 + diag(w)| + ln|K|)
    and
    chol(K + diag(1./w))
    """
    N = w.size
    assert K.shape==(N,N)
    w_sqrt = np.sqrt(w)
    W_inv = np.diag(1./w)
    W_sqrt_inv = np.diag(1./w_sqrt)

    B = np.eye(N) + np.dot(w_sqrt[:,None],w_sqrt[None,:])*K
    cho = linalg.cho_factor(B)
    T = linalg.cho_solve(cho,W_sqrt_inv)
    ret = W_inv - np.dot(W_sqrt_inv,T)
    return ret, np.sum(np.log(np.diag(cho[0]))), (np.dot(cho[0],W_sqrt_inv), cho[1])


def blockdiag(xx):
    """
    Create a block diagonal matrix from a list of smaller matrices
    """
    m,n = zip(*map(lambda x:x.shape,xx))
    ret = np.zeros((np.sum(m),np.sum(n)))
    [np.add(ret[sum(m[:i]):sum(m[:i+1]),sum(n[:i]):sum(n[:i+1])],\
        x,\
        ret[sum(m[:i]):sum(m[:i+1]),sum(n[:i]):sum(n[:i+1])]) for i,x in enumerate(xx)]
    return ret


def jitchol(A,maxtries=5):
    """
    Arguments
    ---------
    A : An almost pd square matrix

    Returns
    -------
    cho_factor(K)

    Notes
    -----
    Adds jitter to K, to enforce positive-definiteness
    """

    try:
        return linalg.cho_factor(A)
    except:
        diagA = np.diag(A)
        if np.any(diagA<0.):
            raise linalg.LinAlgError, "not pd: negative diagonal elements"
        jitter= diagA.mean()*1e-6
        for i in range(1,maxtries+1):
            try:
                return linalg.cho_factor(A+np.eye(A.shape[0])*jitter)
            except:
                jitter *= 10
        raise linalg.LinAlgError,"not positive definite, even with jitter."


def pdinv(A):
    """
    Arguments
    ---------
    A : A DxD pd numpy array

    Returns
    -------
    inv : the inverse of A
    hld: 0.5* the log of the determinant of A
    """
    L = jitchol(A)
    hld = np.sum(np.log(np.diag(L[0])))
    inv = linalg.flapack.dpotri(L[0],True)[0]
    inv = np.triu(inv)+np.triu(inv,1).T
    return inv, hld


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
    invs = [linalg.flapack.dpotri(L[0],True)[0] for L in chols]
    invs = [np.triu(I)+np.triu(I,1).T for I in invs]
    return np.dstack(invs),np.array(halflogdets)

def softmax(x):
    ex = np.exp(x-x.max(1)[:,None])
    return ex/ex.sum(1)[:,np.newaxis]

def softmax_weave(X):
    """
    use weave to compute the softmax of the (rows of) matrix X. 

    uses omp to parallelise the loop

    also returns the log of the softmaxed result
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

def single_softmax(x):
    ex = np.exp(x)
    return ex/ex.sum()

def lngammad(v,D):
    return np.sum([special.gammaln((v+1.-d)/2.) for d in range(1,D+1)],0)

def ln_wishart_B(W_hld,v,D):
    return  -v*W_hld-(v*D*0.5*np.log(2.) + D*(D-1.)*0.25*np.log(np.pi) + lngammad(v,D))

def ln_dirichlet_C(a):
    return special.gammaln(a.sum())-np.sum(special.gammaln(a))

def checkgrad(ffprime,x,fprime=None,step=1e-6, tolerance = 1e-4, *args):
    """check the gradient function fprime by comparing it to a numerical estiamte from the function f"""

    if fprime:
        ffprime=lambda x : ffprime(x),fprime(x)


    #choose a random direction to step in:
    dx = step*np.sign(np.random.uniform(-1,1,x.shape))

    #evaulate around the point x
    f1,g1 = ffprime(x+dx,*args)
    f2,g1= ffprime(x-dx,*args)

    numerical_gradient = (f1-f2)/(2*dx)
    gradient = ffprime(x,*args)[1]
    ratio = (f1-f2)/(2*np.dot(dx,gradient))
    print "gradient = ",gradient
    print "numerical gradient = ",numerical_gradient
    print "ratio = ", ratio, '\n'
    sys.stdout.flush()

    if np.abs(1-ratio)>tolerance:
        print "Ratio far from unity. Testing individual gradients"
        for i in range(len(x)):
            dx = np.zeros(x.shape)
            dx[i] = step*np.sign(np.random.uniform(-1,1,x[i].shape))

            f1,g1 = ffprime(x+dx,*args)
            f2,g2 = ffprime(x-dx,*args)

            numerical_gradient = (f1-f2)/(2*dx)
            gradient = ffprime(x,*args)[1]
            print i,"th element"
            #print "gradient = ",gradient
            #print "numerical gradient = ",numerical_gradient
            ratio = (f1-f2)/(2*np.dot(dx,gradient))
            print "ratio = ",ratio,'\n'
            sys.stdout.flush()


def parse_probfile(filename):
    """
    Parse one of Peter's *.prob files into a sparse matrix
    """
    readnames = []
    index_n = []
    index_k = []
    values = []
    f = file(filename)
    n = 0 # read counter
    for l in f.readlines():
        if l[0]=='#':
            continue # line is a comment
        l = l.split()
        readnames.append(l.pop(0))
        N_alignments = int(l.pop(0))
        for a in range(N_alignments):
            index_n.append(n)
            index_k.append(int(l.pop(0)))
            values.append(float(l.pop(0)))
        assert len(l)==0 # the line should be empty now...

        n += 1 # increment read count
    f.close()

    #construct sparse matrix object
    B = sparse.construct.coo_matrix((values,(index_n,index_k)),dtype=np.float)
    return B

def sparse_softmax(S):
    expS = S.copy()
    expS.data = np.exp(expS.data)
    expS_sum = np.asarray(expS.sum(1)).flatten()
    expS.data /= expS_sum[expS.row]
    return expS

def sparse_softmax_inplace(X,Xtarget):
    """
    take two sparse matrices with the same structure, and make one softmax of the other
    """
    np.exp(X.data,Xtarget.data)
    Xt_sum = np.asarray(Xtarget.sum(1)).flatten()
    Xtarget.data /= Xt_sum[X.row]
    np.log(Xtarget.data,X.data)


