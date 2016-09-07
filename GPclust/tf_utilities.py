import numpy as np
import tensorflow as tf

def lngammad(v,D):
    lgd = tf.reduce_sum( tf.lgamma( 0.5*(v + 1.0 - tf.linspace(1.,tf.cast(D,tf.float32),D)) ) )
    return lgd


def softmax(X):
    phi = tf.nn.softmax(X,name='phi')
    log_phi = tf.nn.log_softmax(X,name='log_phi')
    H = -tf.reduce_sum( tf.mul(phi,log_phi) ,name='H')
    return phi, log_phi, H


def ln_dirichlet_C(a):
    ldc = tf.lgamma( tf.reduce_sum(a) ) - tf.reduce_sum( tf.lgamma(a) )
    return ldc

if __name__ == '__main__':
    # lngammad
    v = 5.1
    D = 5
    lgd = lngammad(v,D)
    with tf.Session() as sess:
        res = sess.run(lgd)
        assert np.isclose(res,0.67775756) 
    
    X = np.random.randn(5,3)
    phi,log_phi,H = softmax(X)
    with tf.Session() as sess:
        res = sess.run([phi,log_phi,H])
        print res
    
    a = 2.0*np.ones(5)
    ldc = ln_dirichlet_C(a)
    with tf.Session() as sess:
        res = sess.run(ldc)
        print res
