import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    #Dist = tf.sub(tf.expand_dims(Xnew,2),tf.expand_dims(self.mun,0)) 
    N = 100; D = 2; K = 7
    M = np.random.rand(N,D)
    v = np.random.rand(D,K)
    Mtf = tf.convert_to_tensor(M)
    vtf = tf.convert_to_tensor(v)
    DistTF = tf.sub(tf.expand_dims(Mtf,2),tf.expand_dims(vtf,0)) 
    Dist = M[:,:,np.newaxis]-v[np.newaxis,:,:]

    g = np.random.rand(D,D,K)
    gtf = tf.convert_to_tensor(g)
    # The following broadcast is not supported, so we have to tile and reshape g
    h = tf.tile(tf.expand_dims(g,0),tf.pack([100,1,1,1]))
    tmp = tf.reduce_sum(tf.mul(tf.expand_dims(Dist,2),h))
    print np.sum(Dist[:,:,None,:]*g[None,:,:,:])
    with tf.Session() as sess:
        res = sess.run(tmp)
        print res
