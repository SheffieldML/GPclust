import numpy as np
import tensorflow as tf
import gpflow
from gpflow._settings import settings
float_type = settings.dtypes.float_type

def tf_multiple_pdinv(A):
    """
    Arguments
    ---------
    A : A DxDxN numpy array (each A[:,:,i] is pd)

    Returns
    -------
    invs : the inverses of A
    hld: 0.5* the log of the determinants of A
    """
    A = tf.convert_to_tensor(A)
    s = tf.shape(A)
    D = s[0]
    N = s[2]
    # Reshape A so tensorflow can deal with it
    A = tf.reshape(tf.transpose(tf.reshape(A,tf.stack([D*D,N]))),tf.stack([N,D,D]))

    chols = tf.batch_cholesky(A + tf.expand_dims(tf.eye(D,dtype=float_type), 0) * 1e-6)
    #RHS = N copies of eye(D), so it is [N,D,D]
    RHS = tf.reshape(tf.tile(tf.eye(D,dtype=float_type),tf.stack([N,1])),tf.stack([N,D,D]))
    invs = tf.batch_cholesky_solve(chols,RHS)
    #Reshape back to original
    invs = tf.reshape(tf.transpose(tf.reshape(invs,tf.stack([N,D*D]))),tf.stack([D,D,N]))
    hld = tf.reduce_sum(tf.log(tf.batch_matrix_diag_part(chols)),1)
    return invs, hld


def lngammad(v, D):
    lgd = tf.reduce_sum(tf.lgamma(0.5*(v + 1.0 - tf.linspace(tf.to_double(1.), tf.cast(D, tf.float64), D))))
    return lgd

def tensor_lngammad(v, D):
    v = tf.convert_to_tensor(v)
    N = tf.shape(v)[0]

    w = tf.linspace(tf.to_double(1.), tf.cast(D, tf.float64),D)
    z = tf.tile(tf.expand_dims(w,1),tf.stack([1,N]))
    z = tf.reshape(z,tf.stack([D,N]))
    lgd = tf.reduce_sum(tf.lgamma(0.5*(v + 1.0 - z)),0)
    return lgd

def softmax(X):
    phi = tf.nn.softmax(X, name='phi')
    log_phi = tf.nn.log_softmax(X, name='log_phi')
    H = -tf.reduce_sum(tf.multiply(phi, log_phi), name='H')
    return phi, log_phi, H


def ln_dirichlet_C(a):
    return tf.lgamma(tf.reduce_sum(a)) - tf.reduce_sum(tf.lgamma(a))

if __name__ == '__main__':
    from utilities import multiple_pdinv

    D = 3; N = 2
    A = np.random.rand(D,D,N)
    A = (A.reshape(D*D,N) + np.eye(D).reshape(D*D,1)).reshape(D,D,N)
    invs, halflogdets = multiple_pdinv(A)
    tfinvs, tfhalflogdets = tf_multiple_pdinv(A)
    with tf.Session() as sess:
        res = sess.run([tfinvs,tfhalflogdets])
        assert np.allclose(res[1],halflogdets,rtol=1e-4)
        assert np.allclose(res[0],invs,rtol=1e-4)
    # lngammad
    v = 5.1
    D = 5
    lgd = lngammad(v, D)
    with tf.Session() as sess:
        res = sess.run(lgd)
        assert np.allclose(res, 0.67775756)

    v = np.asarray([ 0.22524875,  0.84573547,  0.6198872 ,  0.40215012])
    D = 2
    lgd = tensor_lngammad(v,D)
    with tf.Session() as sess:
        res = sess.run(lgd)
        assert np.allclose(res,np.asarray([ 3.45602003,  3.35178811,  2.86457503,  2.98551298]))
    '''X = np.random.randn(5, 3)
    phi, log_phi, H = softmax(X)
    with tf.Session() as sess:
        res = sess.run([phi, log_phi, H])
        print(res)

    a = 2.0*np.ones(5)
    ldc = ln_dirichlet_C(a)
    with tf.Session() as sess:
        res = sess.run(ldc)
        print(res)
    '''
