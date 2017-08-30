import numpy as np
import tensorflow as tf

class Ring2d:
    def __init__(self,name="Ring2d"):
         self.z = tf.placeholder(tf.float32, [None, 2], name='z')

    def __call__(self, z):
        with tf.variable_scope(self.name):
            z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
            z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
            v = (tf.sqrt(z1 * z1 + z2 * z2) - 2) / 0.4
            return v * v

    def mean(self,z):
        return np.mean(z,axis=1)
    def std(self,z):
        return np.std(z,axis=1)
    def measure(self,z,n):
        return np.mean(np.power(z,n),axis=1)