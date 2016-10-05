import numpy as np
import tensorflow as tf

N = 10; K = 4; D = 2

phi = np.random.randn(N,K,D)
Z = tf.ones_like(phi,dtype=tf.float32)
W = Z[:,:,0]
with tf.Session() as sess:
    res = sess.run([W])
    print res[0].shape
