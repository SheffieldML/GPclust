import numpy as np
import tensorflow as tf
from utilities import softmax_weave
from scipy.special import gammaln

def numpy_prop_bound(phi_,K,alpha):
    phi, logphi, H = softmax_weave(phi_)
    phi_hat = phi.sum(0)
    #print phi_hat[::-1].cumsum()
    #print phi_hat[::-1].cumsum()[::-1]
    phi_tilde_plus_hat = phi_hat[::-1].cumsum()[::-1]
    phi_tilde = phi_tilde_plus_hat - phi_hat

    A = gammaln(1. + phi_hat)
    B = gammaln(alpha + phi_tilde)
    C = gammaln(alpha + 1. + phi_tilde_plus_hat)
    D = K*(gammaln(1.+alpha) - gammaln(alpha))
    #print phi_tilde_plus_hat 
    #print A.sum(), B.sum(), C.sum(), D, H

    return phi, logphi, A.sum() + B.sum() - C.sum() + D + H


if __name__ == '__main__':
    N = 100; K = 10; alpha = 5.0
    phi_ = np.random.randn(N,K)
    phi, logphi, orig = numpy_prop_bound(phi_,K,alpha)
    phi_hat = phi.sum(0)
    #print phi_hat
    #w = tf.cumsum(tf.reverse(phi_hat,[True]))
    #q = tf.cumsum(tf.reverse(phi_hat,[True]),reverse=True)
    entropy = -tf.reduce_sum(tf.mul(phi,tf.nn.log_softmax(logphi)))
    alpha = tf.to_double(alpha)
    phi_tilde_plus_hat = tf.reverse(tf.cumsum(tf.reverse(phi_hat,[True])), [True])
    #phi_tilde_plus_hat = tf.cumsum(tf.reverse(phi_hat,[True]), reverse=True)
    phi_tilde = phi_tilde_plus_hat - phi_hat
    A = tf.lgamma(1. + phi_hat)
    B = tf.lgamma(alpha + phi_tilde)
    C = tf.lgamma(alpha + 1. + phi_tilde_plus_hat)
    D = tf.to_double(K)*tf.sub(tf.lgamma(1. + alpha),tf.lgamma(alpha))
    Z = -tf.reduce_sum(A) - tf.reduce_sum(B) + tf.reduce_sum(C) - D - entropy

    with tf.Session() as sess:
        res = sess.run(Z)
        #res = sess.run([Z,tf.reduce_sum(A),tf.reduce_sum(B),tf.reduce_sum(C),D,entropy])
        #print res[1:]
        #print -res[0], orig
        print -res, orig
