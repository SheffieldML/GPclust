import numpy as np
import tensorflow as tf
from utilities import softmax_weave
from scipy.special import gammaln

def orig_version(phi_,alpha):
    K = phi_.shape[1]
    phi, logphi, H = softmax_weave(phi_)
    phi_hat = phi.sum(0)
    Hgrad = -logphi
            
    phi_tilde_plus_hat = phi_hat[::-1].cumsum()[::-1]
            
    phi_tilde = phi_tilde_plus_hat - phi_hat
            
    A = gammaln(1. + phi_hat)
    B = gammaln(alpha + phi_tilde)
    C = gammaln(alpha + 1. + phi_tilde_plus_hat)
    D = K*(gammaln(1.+alpha) - gammaln(alpha))
    #return A.sum() + B.sum() - C.sum() + D + H
    return H

if __name__ == '__main__':
    N = 100; K = 10;
    alpha = 1.1

    logphi = np.random.randn(N,K)
    orig = orig_version(np.exp(logphi),alpha)

    phi = tf.nn.softmax(logphi)
    phi_hat = tf.reduce_sum(phi, 0)
    entropy = -tf.reduce_sum(tf.mul(phi,tf.nn.log_softmax(logphi)))
    norm = tf.expand_dims(tf.reduce_sum(phi),0)
    log_norm = tf.log(norm)
    tentropy = -tf.reduce_sum(tf.mul(phi/norm,tf.nn.log_softmax(logphi)/log_norm))
    phi_tilde_plus_hat = tf.cumsum(tf.reverse(phi_hat,[True]), reverse=True)
    phi_tilde = phi_tilde_plus_hat - phi_hat
    A = tf.lgamma(1. + phi_hat)
    B = tf.lgamma(alpha + phi_tilde)
    C = tf.lgamma(alpha + 1. + phi_tilde_plus_hat)
    D = K*tf.sub(tf.lgamma(1. + alpha),tf.lgamma(alpha))
    Z = -tf.reduce_sum(A) - tf.reduce_sum(B) + tf.reduce_sum(C) - tf.to_double(D) - entropy
    xw = tf.linspace(0.001,5.0,1000)
    w = tf.lgamma(xw)
    with tf.Session() as sess:
        res = sess.run(tentropy)
        print res, orig
    #    res = sess.run(w)
    #    print orig
    #x = np.linspace(0.001,5.0,1000)
    #y = gammaln(x)
    #z = gammaln(x.astype(np.float32))
    #import matplotlib.pyplot as plt
    #plt.semilogy(x,np.abs(y-res),'r')
    #plt.show()
