import numpy as np
import tensorflow as tf

def lngammad(v,D):
    lgd = tf.reduce_sum( tf.lgamma( 0.5*(v + 1.0 - tf.linspace(1.,tf.cast(D,tf.float32),D)) ) )
    with tf.Session() as sess:
        res = sess.run(lgd)
        return res


def softmax(X):
    phi = tf.nn.softmax(X)
    log_phi = tf.nn.log_softmax(X)
    H = -tf.reduce_sum( tf.mul(phi,log_phi) )
    with tf.Session() as sess:
        res = sess.run([phi,log_phi,H])
        return res


def ln_dirichlet_C(a):
    ldc = tf.lgamma( tf.reduce_sum(a) ) - tf.reduce_sum( tf.lgamma(a) )
    with tf.Session() as sess:
        res = sess.run(ldc)
        return res

if __name__ == '__main__':
    # lngammad
    v = 5.1
    D = 5
    lgd = lngammad(v,D)
    assert np.isclose(lgd,0.67775756) 
    
    X = np.random.randn(5,3)
    phi,log_phi,H = softmax(X)
    print phi,log_phi,H
    
    a = 2.0*np.ones(5)
    print ln_dirichlet_C(a)
